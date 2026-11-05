"""One cover: a song re-sung by your singer over a regenerated arrangement that keeps the original melody.

    python job.py --source song.mp3 --output cover.wav --work DIR [--lyrics-file F] [--arrangement TEXT]

Steps (each a separate process, so each model's VRAM is released before the next one loads):
  features  (CPU, beside the score step) the song's drum share, for the reference pick (features.py).
  score     SheetSage2 transcribes the song: lead sheet (melody, chords, sections), beats and downbeats.
  asr       Whisper word timings (sung stretches it missed are heard again); they place the lyric lines
            in the score's sections.
  lyrics    only when no lyrics are given (and not turned off): the song's published lyrics are looked up online
            from what was heard; a failure only leaves them out (lyrics_step.py).
  style     only when no --arrangement is given and a listener is configured: style tags written from the song's audio
            (style_words.py). Optional; a failure leaves the default words.
  plan      picks the singer's reference clip (tempo, then drum share) and writes one composite score
            (reference bars, then the song) plus section-tagged lyrics (plan.py). Timed lyrics (an LRC file given as
            --lyrics-file, or the lookup's) that fit the recording are the plan; otherwise the heard lines are.
  encode    the stretched, aligned reference clip -> YuE2 semantic tokens (real-audio tokenizer head).
  generate  YuE2 continues from those tokens as a forced prefix: the singer's voice carries into the song.
            A render that stops right after the voice prompt is sung again with a fresh seed, up to ATTEMPTS in all
            (sing()); the model stays loaded between seeds of one attempt. The last attempt may use the next-closest
            reference when the first one had to be stretched by more than SWITCH_STRETCH.
  check     optional (YUE2_COVER_LYRIC_CHECK=1): the cover's words are scored and the score is recorded. Never re-sings.
The reference part is cut off the result, so the output starts at the song's first bar; leading echo and dead air after the
voice prompt are trimmed (trim_seam). receipt.json in the work folder lists every attempt: seed, reference, length.
Any failure exits non-zero.
"""
import argparse
import atexit
import hashlib
import json
import math
import os
import secrets
import subprocess
import sys
import time
import wave
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paths  # noqa: E402

NAR_SHA = '585f303da1d5252d228d1e8ac6d4c4d11d970df9297406935cc8bdafa49cfa7e'
# Rides along every Whisper window and keeps a Chinese song in simplified script; pass '' for other languages.
CHINESE_PROMPT = '以下是一首中文歌曲的歌词。'
# About 1 render in 4 stops 1-17 s after the voice prompt with a fixed seed. The same inputs and seed give the same song,
# so a retry needs a new seed: every attempt takes a fresh one, and the receipt keeps each, so a take can be sung again.
ATTEMPTS = 3


def fresh_seeds(count=ATTEMPTS):
    return [1 + secrets.randbelow(2**31 - 2) for _ in range(count)]


# A reference stretched more than this away from its own tempo may be swapped for the next-closest on the last attempt.
SWITCH_STRETCH = 0.03
# Too short to be the song: under this share of the source (or SHORTEST_SECONDS).
SHORTEST_SHARE = 0.5
SHORTEST_SECONDS = 10
EARLY_STOP_EXIT = 4  # run_cover.py: every seed it was given stopped early
# features.py runs beside the score step; past this many seconds from the job's start the pick is by tempo only.
FEATURES_TIMEOUT = 90
CHECK_TIMEOUT = 300


def environment():
    env = {**os.environ, 'HF_HOME': str(paths.HF_HOME), 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
           'HF_HUB_DISABLE_TELEMETRY': '1', 'TOKENIZERS_PARALLELISM': 'false',
           'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4',
           'PYTHONPATH': os.pathsep.join(filter(None, (str(HERE), str(paths.TOKENIZER_SCRIPTS),
                                                       os.environ.get('PYTHONPATH', ''))))}
    return env


def step(name, command, work, timeout, log, allow=()):
    """Run one model step; a non-zero exit raises unless it is in allow (then it is returned)."""
    began = time.monotonic()
    with (work / f'{name}.log').open('wb') as out:
        # Same process group as the caller: stopping the job stops the model step too.
        proc = subprocess.run(list(map(str, command)), cwd=work, env=environment(), stdout=out,
                              stderr=subprocess.STDOUT, timeout=timeout)
    log[name] = round(time.monotonic() - began, 1)
    if proc.returncode and proc.returncode not in allow:
        tail = (work / f'{name}.log').read_text(encoding='utf-8', errors='replace')[-1500:]
        raise RuntimeError(f'YUE2_{name.upper()}_FAILED: {tail}')
    return proc.returncode


def ffmpeg(*args, timeout=180):
    subprocess.run([paths.FFMPEG, '-v', 'error', '-nostdin', '-y', *map(str, args)], check=True, timeout=timeout,
                   env=environment(), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def wav_seconds(path):
    with wave.open(str(path), 'rb') as audio:
        return audio.getnframes() / audio.getframerate()


# After the forced reference phrase the model often sings on a little of it, then leaves the tail of the reference
# section silent before the song comes in (seen: 9 s of echo then 2.5 s of dead air). The song itself never has dead air in
# its first seconds, so the cover starts after the last stretch of it there.
SEAM_WINDOW = 15.0       # seconds searched from the cut
SEAM_QUIET_DB = -60.0    # dead air: a 50 ms frame quieter than this
SEAM_QUIET_SECONDS = 0.4
SEAM_FADE = 0.03


def seam_start(audio, rate):
    """Frame index where the cover should start: after the last dead-air stretch (>= SEAM_QUIET_SECONDS of frames below
    SEAM_QUIET_DB) in the first SEAM_WINDOW seconds, or None when there is none. audio: float array, frames x channels."""
    import numpy as np
    frame = int(rate * 0.05)
    frames = min(len(audio), int(SEAM_WINDOW * rate)) // frame
    run, start = 0, None
    for i in range(frames):
        part = audio[i * frame:(i + 1) * frame]
        if 20 * np.log10(max(1e-6, float(np.sqrt(np.mean(part ** 2))))) < SEAM_QUIET_DB:
            run += 1
            continue
        if run * 0.05 >= SEAM_QUIET_SECONDS:
            start = i
        run = 0
    return None if start is None else max(0, start * frame - frame)


def trim_seam(path):
    """Start the cover after its leading echo and dead air (in place); seconds removed, 0.0 when there was none."""
    import numpy as np
    import soundfile as sf
    audio, rate = sf.read(str(path), always_2d=True, dtype='float32')
    begin = seam_start(audio, rate)
    if begin is None:
        return 0.0
    kept = audio[begin:].copy()
    fade = min(len(kept), int(SEAM_FADE * rate))
    kept[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)[:, None]
    sf.write(str(path), kept, rate, subtype='PCM_16')
    return begin / rate


def reference_audio(plan, work):
    """Stretch the singer's clip to the song's tempo and shift it so its first note sits where its score says."""
    refs = json.loads((paths.REFERENCES / 'refs.json').read_text(encoding='utf-8'))
    clip = paths.REFERENCES / next(r['path'] for r in refs if r['clip'] == plan['reference'])
    lead, samples = plan['lead_seconds'], plan['reference_samples']
    chain = [f"atempo={plan['reference_rate']:.6f}"]
    if lead >= 0:
        chain.append(f'adelay=delays={round(lead * 48000)}S:all=1')
    else:
        chain.append(f'atrim=start_sample={round(-lead * 48000)},asetpts=PTS-STARTPTS')
    chain += [f'apad=whole_len={samples}', f'atrim=end_sample={samples}']
    out = work / 'reference-aligned.flac'
    ffmpeg('-i', clip, '-af', ','.join(chain), '-ar', 48000, '-ac', 2, '-c:a', 'flac', out)
    return out


class Unsung(RuntimeError):
    """Every attempt stopped early (or came out too short): the job fails."""

    def __init__(self, attempts):
        super().__init__('YUE2_EARLY_STOP')
        self.attempts = attempts


def stretched(plan):
    return abs(plan['reference_rate'] - 1) > SWITCH_STRETCH


def sing(plan, seeds, shortest, prefixes, *, prepare, generate, finish, alternative, attempts_max=ATTEMPTS):
    """Up to attempts_max renders of the song; the first long enough wins.

    generate(plan, prefix, seeds, attempt) sings the seeds in order with one model load and returns (complete, early):
    complete is run_cover's record for the first seed that sang (None when all stopped early), early the early stops
    before it. finish(complete, plan) cuts the cover out and returns (audio_seconds, trimmed). When the reference is
    stretched, the last seed is kept back so the last attempt can use alternative(plan) (None: none close enough);
    prepare(plan) encodes a reference that has no prefix yet. Returns (complete, plan, audio_seconds, trimmed, attempts).
    Raises Unsung(attempts) when no attempt was long enough. Each attempts entry has attempt, seed, reference,
    audio_seconds, early_stop and, where it applies, too_short or fault."""
    attempts, n, current, switched = [], 0, plan, False
    while n < attempts_max:
        if n == attempts_max - 1 and not switched and stretched(current):
            switched = True
            other = alternative(current)
            if other is not None:
                current = other
                if current['reference'] not in prefixes:
                    prefixes[current['reference']] = prepare(current)
        hold = n < attempts_max - 1 and not switched and stretched(current)
        batch = seeds[n:attempts_max - 1] if hold else seeds[n:attempts_max]
        complete, early = generate(current, prefixes[current['reference']], batch, n + 1)
        for stop in early:
            n += 1
            attempts.append({'attempt': n, 'seed': stop['seed'], 'reference': current['reference'],
                             'audio_seconds': round(stop['new_frames'] / 25, 2), 'early_stop': True,
                             'seconds': stop['seconds'], **({'fault': True} if stop.get('fault') else {})})
        if complete is None:
            continue
        n += 1
        audio_seconds, trimmed = finish(complete, current)
        sung = audio_seconds >= shortest
        attempts.append({'attempt': n, 'seed': complete['seed'], 'reference': current['reference'],
                         'audio_seconds': round(audio_seconds, 2), 'early_stop': False,
                         **({} if sung else {'too_short': True})})
        if sung:
            return complete, current, audio_seconds, trimmed, attempts
    raise Unsung(attempts)


def run(args):
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    py, sage = paths.PYTHON, paths.SAGE_PYTHON
    log, began = {}, time.monotonic()
    source = work / 'source.wav'
    ffmpeg('-i', args.source, '-ac', 2, '-ar', 48000, '-c:a', 'pcm_s16le', source)
    # The song's mix measures (CPU) while SheetSage and Whisper run; plan.build reads them for the reference pick.
    features = start_features(py, source, work)
    step('score', [sage, HERE / 'score_step.py', source, work / 'score'], work, 600, log)
    from plan import build, lrc_lines, text_language
    lyrics = Path(args.lyrics_file).read_text(encoding='utf-8') if args.lyrics_file else ''
    given = lrc_lines(lyrics)
    # Timed lyrics say the song's language; Whisper's own guess from one window of a mixed-language song can be wrong.
    hint = text_language([line for _, line in given]) if given else ''
    # The score's vocal melody lets the ASR step hear again the sung stretches its whole-song pass skipped.
    step('asr', [py, HERE / 'asr_step.py', source, work / 'asr.json', args.asr_prompt, work / 'score', hint],
         work, 600, log)
    lookup = work / 'lyrics-lookup.json'
    lookup.unlink(missing_ok=True)          # never reuse the result of an earlier attempt in this folder
    if (not lyrics.strip() or given) and paths.LYRICS_LOOKUP and not args.no_lyrics_lookup:
        # No lyrics, or only timed ones: the published ones are looked up from what was heard (lyrics_step.py). Any
        # failure only leaves the lookup out; the heard lyrics are sung as they are.
        try:
            step('lyrics', [py, HERE / 'lyrics_step.py', work / 'asr.json', lookup], work, 90, log)
        except (RuntimeError, subprocess.TimeoutExpired):
            lookup.unlink(missing_ok=True)
    arrangement, style = style_for_job(args, source, work)
    t = time.monotonic()
    try:
        if features is not None:
            features.wait(timeout=max(1.0, FEATURES_TIMEOUT - (t - began)))
    except subprocess.TimeoutExpired:
        features.kill()
        features.wait()
    if features is not None and features.returncode:
        (work / 'features.json').unlink(missing_ok=True)
    log['features_wait'] = round(time.monotonic() - t, 1)
    t = time.monotonic()
    plan = build(work, lyrics, arrangement)
    log['plan'] = round(time.monotonic() - t, 1)
    import soundfile
    source_seconds = soundfile.info(str(source)).duration
    shortest = max(SHORTEST_SECONDS, SHORTEST_SHARE * source_seconds)
    folders = {plan['reference']: work}

    def prepare(chosen):
        folder = folders[chosen['reference']]
        reference = reference_audio(chosen, folder)
        prefix = folder / 'encoded/semantic-prefix.npy'
        prefix.parent.mkdir(exist_ok=True)
        name = 'encode' if folder == work else f"encode-{chosen['reference']}"
        step(name, [py, HERE / 'encode_impl.py', '--input', reference, '--input-sha256', sha(reference),
                    '--output', prefix], work, 600, log)
        return prefix

    def alternative(current):
        folder = work / 'alternative'
        try:
            other = build(work, lyrics, arrangement, exclude={current['reference']}, out=folder)
        except ValueError:  # no other reference close enough in tempo (or it does not fit this score)
            return None
        folders[other['reference']] = folder
        return other

    faults = min(ATTEMPTS, max(0, paths.FAULT_EARLY_STOPS))

    def generate(chosen, prefix, seeds, attempt):
        name, out = f'generate-{attempt}', work / f'generate-{attempt}'
        config = {'model_dir': str(paths.YUE2), 'vae_dir': str(paths.VAE),
                  'nar_path': str(paths.NAR), 'nar_sha256': NAR_SHA,
                  'request_path': str(folders[chosen['reference']] / 'request.json'),
                  'semantic_prefix_path': str(prefix), 'reference_frames': chosen['reference_frames'],
                  'output_root': str(out), 'semantic_sampling': {'max_tokens': chosen['max_tokens']}, 'ode_steps': 64,
                  'seeds': seeds, 'min_new_frames': math.ceil(shortest * 25),
                  'fault_early_stops': min(len(seeds), max(0, faults - attempt + 1))}
        (work / f'{name}-config.json').write_text(json.dumps(config, indent=1), encoding='utf-8')
        code = step(name, [py, HERE / 'run_cover.py', '--config', work / f'{name}-config.json', '--phase', 'generate'],
                    work, 2400, log, allow=(EARLY_STOP_EXIT,))
        if code == EARLY_STOP_EXIT:
            return None, json.loads((out / 'voice-prefix/early-stops.json').read_text(encoding='utf-8'))['early_stops']
        complete = json.loads((out / 'voice-prefix/complete.json').read_text(encoding='utf-8'))
        return complete, complete['early_stops']

    def finish(complete, chosen):
        audio = Path(complete['artifact_root']) / 'audio.flac'
        # The reference phrase is only the voice prompt; the cover starts at the song's first bar.
        ffmpeg('-ss', f"{chosen['reference_seconds']:.6f}", '-i', audio, '-ac', 2, '-ar', 48000, '-c:a', 'pcm_s16le',
               args.output)
        trimmed = trim_seam(args.output)
        return complete['duration_seconds'] - chosen['reference_seconds'] - trimmed, trimmed

    prefixes = {plan['reference']: prepare(plan)}
    try:
        complete, chosen, audio_seconds, trimmed, attempts = sing(
            plan, fresh_seeds(), shortest, prefixes, prepare=prepare, generate=generate, finish=finish,
            alternative=alternative)
    except Unsung as failed:
        Path(args.output).unlink(missing_ok=True)
        receipt = {**plan, 'style': style, 'steps': log, 'seconds': round(time.monotonic() - began, 1),
                   'source_seconds': round(source_seconds, 2), 'attempts': failed.attempts, 'error': 'YUE2_EARLY_STOP'}
        (work / 'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=1), encoding='utf-8')
        print(json.dumps(receipt, ensure_ascii=False))
        raise SystemExit('YUE2_EARLY_STOP')
    check = None
    if paths.LYRIC_CHECK and chosen.get('language') == 'zh':
        check = lyric_check(py, args.output, folders[chosen['reference']] / 'request.json', work, log)
    receipt = {**chosen, 'style': style, **({'lyric_check': check} if check is not None else {}),
               'steps': log, 'seconds': round(time.monotonic() - began, 1), 'trimmed_start': round(trimmed, 2),
               'audio_seconds': audio_seconds, 'truncated': complete['truncated'],
               'peak_reserved_gib': complete['peak_reserved_gib'], 'seed': complete['seed'], 'attempts': attempts,
               'source_seconds': round(source_seconds, 2)}
    (work / 'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(receipt, ensure_ascii=False))


def start_features(py, source, work):
    """Measure the song beside the score step; None when features.py cannot be started (the pick is then by tempo)."""
    try:
        out = (work / 'features.log').open('wb')
        proc = subprocess.Popen([str(py), str(HERE / 'features.py'), str(source), str(work / 'features.json')],
                                cwd=work, env=environment(), stdout=out, stderr=subprocess.STDOUT)
    except OSError:
        return None
    atexit.register(lambda: proc.poll() is None and proc.kill())
    return proc


def style_for_job(args, source, work):
    """(arrangement words, style record): the user's words as they are, else tags written from the song by the
    configured listener, else '' (the default words). See style_words.py."""
    import style_words
    listener = None
    if not (args.arrangement or '').strip() and paths.LISTENER:
        try:
            listener = style_words.load_listener(paths.LISTENER)
        except (ImportError, AttributeError) as error:
            print(f'style listener not usable, default words are used: {error}', file=sys.stderr)
    words, record = style_words.resolve(args.arrangement, source, work, listener=listener,
                                        ffmpeg=paths.FFMPEG, ffprobe=paths.FFPROBE)
    style_words.save(record, work)
    return words, record


def lyric_check(py, cover, request, work, log):
    """The cover's words scored by lyric_check.py (recorded only). None when the check could not run."""
    folder = work / 'words-check'
    try:
        step('words-separate', [py, HERE / 'lyric_check.py', 'separate', '--audio', cover, '--work', folder],
             work, CHECK_TIMEOUT, log)
        step('words', [py, HERE / 'lyric_check.py', 'hear', '--vocals', folder / 'vocals.wav', '--request', request,
                       '--asr', work / 'asr.json', '--language', 'zh', '--out', work / 'words.json'],
             work, CHECK_TIMEOUT, log)
        return json.loads((work / 'words.json').read_text(encoding='utf-8'))
    except (RuntimeError, OSError, ValueError, subprocess.SubprocessError) as error:
        log['words-error'] = str(error)[:200]
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--source', required=True, help='the song to cover (any format ffmpeg reads)')
    parser.add_argument('--output', required=True, help='the cover, 48 kHz stereo WAV')
    parser.add_argument('--work', required=True, help='a folder for this job')
    parser.add_argument('--lyrics-file', help='the lyrics, one line per sung line (or an LRC file with timestamps); '
                                              "without it, Whisper's text is sung")
    parser.add_argument('--arrangement', help='genre/arrangement words in English, e.g. "acoustic guitar folk"; '
                                              'without them: written from the song if a listener is configured')
    parser.add_argument('--asr-prompt', default=CHINESE_PROMPT, help="Whisper prompt (used for Chinese songs only)")
    parser.add_argument('--no-lyrics-lookup', action='store_true',
                        help="do not look the song's published lyrics up online "
                             '(or set YUE2_COVER_LYRICS_LOOKUP=0)')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
