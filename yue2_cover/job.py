"""One cover: a song re-sung by your singer over a regenerated arrangement that keeps the original melody.

    python job.py --source song.mp3 --output cover.wav --work DIR [--lyrics-file F] [--arrangement TEXT]

Steps (each a separate process, so each model's VRAM is released before the next one loads):
  score     SheetSage2 transcribes the song: lead sheet (melody, chords, sections), beats and downbeats.
  asr       Whisper word timings; they place the lyric lines in the score's sections.
  plan      picks the singer's reference clip nearest in tempo and writes one composite score
            (reference bars, then the song) plus section-tagged lyrics (plan.py).
  encode    the stretched, aligned reference clip -> YuE2 semantic tokens (real-audio tokenizer head).
  generate  YuE2 continues from those tokens as a forced prefix: the singer's voice carries into the song.
The reference part is cut off the result, so the output starts at the song's first bar.
Any failure exits non-zero.
"""
import argparse
import hashlib
import json
import os
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


def environment():
    env = {**os.environ, 'HF_HOME': str(paths.HF_HOME), 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
           'HF_HUB_DISABLE_TELEMETRY': '1', 'TOKENIZERS_PARALLELISM': 'false',
           'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4',
           'PYTHONPATH': os.pathsep.join(filter(None, (str(HERE), str(paths.TOKENIZER_SCRIPTS),
                                                       os.environ.get('PYTHONPATH', ''))))}
    return env


def step(name, command, work, timeout, log):
    began = time.monotonic()
    with (work / f'{name}.log').open('wb') as out:
        # Same process group as the caller: stopping the job stops the model step too.
        proc = subprocess.run(list(map(str, command)), cwd=work, env=environment(), stdout=out,
                              stderr=subprocess.STDOUT, timeout=timeout)
    log[name] = round(time.monotonic() - began, 1)
    if proc.returncode:
        tail = (work / f'{name}.log').read_text(encoding='utf-8', errors='replace')[-1500:]
        raise RuntimeError(f'YUE2_{name.upper()}_FAILED: {tail}')


def ffmpeg(*args, timeout=180):
    subprocess.run([paths.FFMPEG, '-v', 'error', '-nostdin', '-y', *map(str, args)], check=True, timeout=timeout,
                   env=environment(), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def wav_seconds(path):
    with wave.open(str(path), 'rb') as audio:
        return audio.getnframes() / audio.getframerate()


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


def run(args):
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    py, sage = paths.PYTHON, paths.SAGE_PYTHON
    log, began = {}, time.monotonic()
    source = work / 'source.wav'
    ffmpeg('-i', args.source, '-ac', 2, '-ar', 48000, '-c:a', 'pcm_s16le', source)
    step('score', [sage, HERE / 'score_step.py', source, work / 'score'], work, 600, log)
    step('asr', [py, HERE / 'asr_step.py', source, work / 'asr.json', args.asr_prompt], work, 600, log)
    from plan import build
    lyrics = Path(args.lyrics_file).read_text(encoding='utf-8') if args.lyrics_file else ''
    t = time.monotonic()
    plan = build(work, lyrics, args.arrangement or '')
    log['plan'] = round(time.monotonic() - t, 1)
    reference = reference_audio(plan, work)
    prefix = work / 'encoded/semantic-prefix.npy'
    prefix.parent.mkdir(exist_ok=True)
    step('encode', [py, HERE / 'encode_impl.py', '--input', reference, '--input-sha256', sha(reference),
                    '--output', prefix], work, 600, log)
    config = {'model_dir': str(paths.YUE2), 'vae_dir': str(paths.VAE),
              'nar_path': str(paths.NAR), 'nar_sha256': NAR_SHA,
              'request_path': str(work / 'request.json'), 'semantic_prefix_path': str(prefix),
              'reference_frames': plan['reference_frames'], 'output_root': str(work / 'generate'),
              'semantic_sampling': {'max_tokens': plan['max_tokens']}, 'ode_steps': 64}
    (work / 'generate-config.json').write_text(json.dumps(config, indent=1), encoding='utf-8')
    step('generate', [py, HERE / 'run_cover.py', '--config', work / 'generate-config.json', '--phase', 'generate'],
         work, 2400, log)
    complete = json.loads((work / 'generate/voice-prefix/complete.json').read_text(encoding='utf-8'))
    audio = Path(complete['artifact_root']) / 'audio.flac'
    # The reference phrase is only the voice prompt; the cover starts at the song's first bar.
    ffmpeg('-ss', f"{plan['reference_seconds']:.6f}", '-i', audio, '-ac', 2, '-ar', 48000, '-c:a', 'pcm_s16le', args.output)
    # Now and then the model stops right after the voice prompt (a 275 s song once came back as 1.8 s).
    sung, song = wav_seconds(args.output), wav_seconds(source)
    if sung < max(10, 0.5 * song):
        raise RuntimeError(f'YUE2_TOO_SHORT: {sung:.1f} s for a {song:.1f} s song; try another reference clip '
                           'or give the lyrics')
    receipt = {**plan, 'steps': log, 'seconds': round(time.monotonic() - began, 1),
               'audio_seconds': complete['duration_seconds'] - plan['reference_seconds'],
               'truncated': complete['truncated'], 'peak_reserved_gib': complete['peak_reserved_gib']}
    (work / 'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(receipt, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--source', required=True, help='the song to cover (any format ffmpeg reads)')
    parser.add_argument('--output', required=True, help='the cover, 48 kHz stereo WAV')
    parser.add_argument('--work', required=True, help='a folder for this job')
    parser.add_argument('--lyrics-file', help='the lyrics, one line per sung line; without it, Whisper\'s text is sung')
    parser.add_argument('--arrangement', help='genre/arrangement words, e.g. "acoustic guitar folk"')
    parser.add_argument('--asr-prompt', default=CHINESE_PROMPT, help="Whisper prompt; '' for non-Chinese songs")
    run(parser.parse_args())


if __name__ == '__main__':
    main()
