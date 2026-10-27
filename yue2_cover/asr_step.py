"""Word-timed Whisper transcription of the song (openai-whisper on GPU).

    python asr_step.py SONG.wav OUT.json [PROMPT [SCORE_DIR]]

Each 30 s window is decoded on its own (no previous-text conditioning): conditioning makes long
songs loop on one line. The text only anchors times for the given lyrics, or stands in for them
when none are given.

Whole-song decoding still skips sung lines now and then (in one pop song a whole verse after the interlude
was missing, so the model got 84 notes for 29 syllables there and stretched and rushed the words). With the
score's vocal melody at hand (SCORE_DIR, the folder score_step.py wrote), every stretch of sung notes that no real
word starts near is heard again on its own (Whisper `clip_timestamps`, same options), and the new segments take
the place of whatever the whole-song pass put there. The re-heard spans are listed under "refilled" in OUT.json.
"""
import json
import sys
from pathlib import Path

import paths
from plan import NOISE

PHRASE_GAP = 1.5      # a rest this long between unheard vocal notes ends a span
NEAR = 1.0            # a word starting this close to a note counts as hearing it
PAD = 0.5             # a span is heard again with this much around it
MIN_SPAN = 2.0        # shorter unheard stretches are left as they are


def segments_of(result):
    return [{'start': round(s['start'], 3), 'end': round(s['end'], 3), 'text': s['text'].strip(),
             'words': [{'w': w['word'], 's': round(w['start'], 3), 'e': round(w['end'], 3)}
                       for w in s.get('words', [])]}
            for s in result['segments']]


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


def hear_again(model, source, segments, notes, options):
    """(segments, spans): the stretches of sung notes the whole-song pass missed, decoded again on their own."""
    spans = unheard(segments, notes)
    if not spans:
        return segments, []
    again = segments_of(model.transcribe(str(source), clip_timestamps=[t for span in spans for t in span], **options))
    return merge_heard(segments, again, spans), [[round(a, 2), round(b, 2)] for a, b in spans]


def main():
    source, out = Path(sys.argv[1]), Path(sys.argv[2])
    prompt = sys.argv[3] if len(sys.argv) > 3 else ''
    score_dir = sys.argv[4] if len(sys.argv) > 4 else None
    import whisper
    model = whisper.load_model(paths.WHISPER, device='cuda')
    # The simplified-Chinese prompt rides along every window, which keeps Whisper out of traditional script.
    options = dict(word_timestamps=True, condition_on_previous_text=False, temperature=0,
                   initial_prompt=prompt or None, carry_initial_prompt=bool(prompt), verbose=None)
    result = model.transcribe(str(source), **options)
    segments, refilled = segments_of(result), []
    if score_dir:
        segments, refilled = hear_again(model, source, segments, sung_notes(score_dir), options)
    out.write_text(json.dumps({'language': result.get('language'), 'segments': segments, 'refilled': refilled},
                              ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'language': result.get('language'), 'segments': len(segments), 'refilled': refilled}))


if __name__ == '__main__':
    main()
