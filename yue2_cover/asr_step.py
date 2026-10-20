"""Word-timed Whisper transcription of the song (openai-whisper on GPU).

Each 30 s window is decoded on its own (no previous-text conditioning): conditioning makes long
songs loop on one line. The text only anchors times for the given lyrics, or stands in for them
when none are given.
"""
import json
import sys
from pathlib import Path

import paths


def main():
    source, out = Path(sys.argv[1]), Path(sys.argv[2])
    prompt = sys.argv[3] if len(sys.argv) > 3 else ''
    import whisper
    model = whisper.load_model(paths.WHISPER, device='cuda')
    # The simplified-Chinese prompt rides along every window, which keeps Whisper out of traditional script.
    result = model.transcribe(str(source), word_timestamps=True, condition_on_previous_text=False,
                              temperature=0, initial_prompt=prompt or None, carry_initial_prompt=bool(prompt),
                              verbose=None)
    segments = [{'start': round(s['start'], 3), 'end': round(s['end'], 3), 'text': s['text'].strip(),
                 'words': [{'w': w['word'], 's': round(w['start'], 3), 'e': round(w['end'], 3)}
                           for w in s.get('words', [])]}
                for s in result['segments']]
    out.write_text(json.dumps({'language': result.get('language'), 'segments': segments},
                              ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'language': result.get('language'), 'segments': len(segments)}))


if __name__ == '__main__':
    main()
