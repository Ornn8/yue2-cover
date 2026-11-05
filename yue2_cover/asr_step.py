"""Word-timed Whisper transcription of the song (openai-whisper on GPU).

    python asr_step.py SONG.wav OUT.json [PROMPT [SCORE_DIR [LANGUAGE]]]

Each 30 s window is decoded on its own (no previous-text conditioning): conditioning makes long
songs loop on one line. The text only anchors times for the given lyrics, or stands in for them
when none are given.

Whole-song decoding still skips sung lines now and then (in one pop song a whole verse after the interlude
was missing, so the model got 84 notes for 29 syllables there and stretched and rushed the words). With the
score's vocal melody at hand (SCORE_DIR, the folder score_step.py wrote), every stretch of sung notes that no real
word starts near is heard again on its own (Whisper `clip_timestamps`, same options), and the new segments take
the place of whatever the whole-song pass put there. The re-heard spans are listed under "refilled" in OUT.json.

LANGUAGE (zh, en, ja or ko) fixes Whisper's language, e.g. from timed lyrics you already have; without it the language is
heard in the most-sung 30 s of the song (the first 30 s are often an intro). The prompt is used for Chinese only.
Segments Whisper made up are dropped: far faster than anyone sings, one short unit repeated, or an echo of the prompt.
"""
import json
import re
import sys
from pathlib import Path

import paths
from plan import NOISE

PHRASE_GAP = 1.5      # a rest this long between unheard vocal notes ends a span
NEAR = 1.0            # a word starting this close to a note counts as hearing it
PAD = 0.5             # a span is heard again with this much around it
MIN_SPAN = 2.0        # shorter unheard stretches are left as they are


def heard_noise(text, seconds):
    """A segment Whisper made up rather than heard (in a 1.4 s instrumental tail it once wrote one 3-character unit 74
    times, and the plan sang it as an extra chorus line): far faster than anyone sings, or one short unit repeated over
    nearly all of it."""
    chars = re.sub(r'[\s\W_]+', '', text)
    if len(chars) >= 12 and len(chars) > 12 * max(seconds, 0.1) + 4:
        return True
    for size in range(1, 7):
        unit = chars[:size]
        if len(chars) >= 16 and unit and chars.count(unit) >= 6 and chars.count(unit) * size >= 0.8 * len(chars):
            return True
    return False


def prompt_echo(text, prompt):
    """A segment that only echoes the decoding prompt (after a long intro Whisper may write pieces of the prompt over
    and over, and the cover then sings those lines in place of the verse): nearly all of its characters are the
    prompt's, and one two-character piece of the prompt comes twice or more. Dropped like noise, so the refill hears
    the sung notes there again."""
    chars = re.sub(r'[\s\W_]+', '', text)
    base = re.sub(r'[\s\W_]+', '', prompt or '')
    if len(chars) < 4 or len(base) < 2:
        return False
    pieces = {base[i:i + 2] for i in range(len(base) - 1)}
    inside = sum(c in base for c in chars)
    return inside >= 0.8 * len(chars) and any(chars.count(piece) >= 2 for piece in pieces)


def segments_of(result, prompt=''):
    return [{'start': round(s['start'], 3), 'end': round(s['end'], 3), 'text': s['text'].strip(),
             'words': [{'w': w['word'], 's': round(w['start'], 3), 'e': round(w['end'], 3)}
                       for w in s.get('words', [])]}
            for s in result['segments']
            if not heard_noise(s['text'], s['end'] - s['start']) and not prompt_echo(s['text'], prompt)]


def sung_notes(score_dir):
    """(onset, offset) of every note of the score's vocal melody (melody_vocal.lab: onset, offset, pitch)."""
    path = Path(score_dir) / 'melody_vocal.lab'
    if not path.is_file():
        return []
    notes = []
    for line in path.read_text(encoding='utf-8').splitlines():
        fields = line.split()
        if len(fields) >= 2:
            notes.append((float(fields[0]), float(fields[1])))
    return sorted(notes)


def unheard(segments, notes):
    """Stretches of sung notes that no real word starts near, as [start, end] padded by PAD.

    Credits and subtitle lines (NOISE) are not words of the song, so they do not count as hearing anything.
    The test is note by note, not phrase by phrase: a verse sung without a breath long enough to split it was
    only half heard.
    """
    starts = sorted(w['s'] for s in segments if not any(x in s['text'] for x in NOISE) for w in s['words'])
    spans = []
    for a, b in notes:
        if any(a - NEAR <= t <= b + NEAR for t in starts):
            continue
        if spans and a - spans[-1][1] < PHRASE_GAP:
            spans[-1][1] = max(spans[-1][1], b)
        else:
            spans.append([a, b])
    return [[max(0.0, a - PAD), b + PAD] for a, b in spans if b - a >= MIN_SPAN]


def merge_heard(segments, again, spans):
    """The whole-song segments, except that inside a re-heard span (by segment midpoint) the new ones take over.

    What the whole pass put there (a credit line where a verse was sung) gives way.
    """
    def inside(segment):
        middle = (segment['start'] + segment['end']) / 2
        return any(a <= middle < b for a, b in spans)

    return sorted([s for s in segments if not inside(s)] + [s for s in again if inside(s)],
                  key=lambda s: s['start'])


def clamp_spans(spans, length):
    """Spans end where the audio does. A span ending past the end (a song cut mid-phrase) sends Whisper's
    clip_timestamps round and round until the step times out; one starting within half a second of the end is dropped."""
    return [[a, min(b, length)] for a, b in spans if a < length - 0.5]


def language_of(model, source, notes):
    """The song's language, heard in its most sung 30 s (its first 30 s are often an intro with no voice)."""
    import whisper
    audio = whisper.load_audio(str(source))
    starts = [a for a, _ in notes]
    begin = max(starts, key=lambda s: sum(1 for a in starts if s <= a < s + 30)) if starts else 0.0
    clip = whisper.pad_or_trim(audio[int(begin * 16000):int(begin * 16000) + 30 * 16000])
    _, probs = model.detect_language(whisper.log_mel_spectrogram(clip, model.dims.n_mels).to(model.device))
    return max(probs, key=probs.get)


def hear_again(model, source, segments, notes, options, length=None):
    """(segments, spans): the stretches of sung notes the whole-song pass missed, decoded again on their own."""
    spans = unheard(segments, notes)
    if length is not None:
        spans = clamp_spans(spans, length)
    if not spans:
        return segments, []
    again = segments_of(model.transcribe(str(source), clip_timestamps=[t for span in spans for t in span], **options),
                        options.get('initial_prompt') or '')
    return merge_heard(segments, again, spans), [[round(a, 2), round(b, 2)] for a, b in spans]


def main():
    source, out = Path(sys.argv[1]), Path(sys.argv[2])
    prompt = sys.argv[3] if len(sys.argv) > 3 else ''
    score_dir = sys.argv[4] if len(sys.argv) > 4 else None
    hint = sys.argv[5] if len(sys.argv) > 5 else ''
    import whisper
    model = whisper.load_model(paths.WHISPER, device='cuda')
    language = hint if hint in ('zh', 'en', 'ja', 'ko') else language_of(
        model, source, sung_notes(score_dir) if score_dir else [])
    # The simplified-Chinese prompt rides along every window, which keeps Whisper out of traditional script; only for a
    # Chinese song: on an English one it can make Whisper hear subtitle credits and nothing of the song.
    prompt = prompt if language == 'zh' else ''
    options = dict(word_timestamps=True, condition_on_previous_text=False, temperature=0, language=language,
                   initial_prompt=prompt or None, carry_initial_prompt=bool(prompt), verbose=None)
    result = model.transcribe(str(source), **options)
    segments, refilled = segments_of(result, prompt), []
    if score_dir:
        length = len(whisper.load_audio(str(source))) / 16000 - 0.05
        segments, refilled = hear_again(model, source, segments, sung_notes(score_dir), options, length)
    out.write_text(json.dumps({'language': result.get('language'), 'segments': segments, 'refilled': refilled},
                              ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'language': result.get('language'), 'segments': len(segments), 'refilled': refilled}))


if __name__ == '__main__':
    main()
