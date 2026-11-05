"""Can the finished cover's words be heard? An optional check that only records a score; nothing here changes the audio.

The same plan sung twice can come out with 22 and 18 of 24 lines audible, so a take's intelligibility is worth measuring.
job.py runs this after a cover when YUE2_COVER_LYRIC_CHECK=1 and writes the score into receipt.json ("lyric_check");
it never sings the song again (a production setup once did, and switched that off). Chinese plans only.

    lyric_check.py separate --audio TAKE.wav --work DIR              -> DIR/vocals.wav  (needs demucs: pip install demucs)
    lyric_check.py hear --vocals V.wav --request request.json --asr asr.json --language zh --out SCORE.json

hear: Whisper on the lead vocal, twice (with the Chinese prompt, its echoes dropped, and without); a planned line is
heard when some heard stretch of about its length matches it at ratio >= LINE_RATIO, lines taken in order. share =
heard lines / planned lines; source_share = the same for the source mix's own Whisper text (asr.json), since a line
nobody can hear in the original is not the cover's fault. passed = share >= PASS_SHARE or
share >= PASS_OF_SOURCE * source_share. Numbers only; it prints no lyrics.

Known weakness: tuned on Mandarin only. Other languages need their own thresholds (see the README's help-wanted list)."""
import argparse
import difflib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paths  # noqa: E402

PROMPT = '以下是一首中文歌曲的歌词。'
UNIT = re.compile(r"[一-鿿]|[가-힣]|[ぁ-ゖァ-ヺ][ぁぃぅぇぉゃゅょゎァィゥェォャュョヮ]?ー?|[A-Za-z][A-Za-z']*")
LINE_RATIO = 0.5
PASS_SHARE = 0.85        # takes we measured: good ones 0.89-0.98, ones a listener found wrong 0.75-0.79
PASS_OF_SOURCE = 0.9     # ... or this share of what can be heard in the original itself


def plain(text):
    return ''.join(UNIT.findall(text)).lower()


def planned_lines(request):
    """The target's lines: every lyric block after the reference blocks ([Verse] and an optional empty [Interlude])."""
    blocks = request['lyrics'].strip().split('\n\n')
    rest = blocks[1:]
    if rest and rest[0].strip().lower() == '[interlude]':
        rest = rest[1:]
    return [l.strip() for b in rest for l in b.splitlines()[1:] if plain(l)]


AHEAD = 3                # a line is looked for no further on than this many lines' worth of heard characters


def heard_share(lines, texts):
    """Share of lines heard in order: each line is looked for from where the line before it was found, no further on
    than AHEAD lines' length; found, the search goes on after it. In order, not anywhere in the song: a song's chorus
    sung clearly once would otherwise count for all three choruses (0.85 anywhere, 0.44 in order in one test).
    The best of the heard texts (with and without the prompt) counts."""
    def ordered(heard):
        pos, hits = 0, 0
        for k, line in enumerate(lines):
            a = plain(line)
            n = len(a)
            reach = pos + sum(len(plain(l)) for l in lines[k:k + AHEAD]) + 2 * AHEAD
            best = None
            for i in range(pos, max(pos, min(len(heard), reach) - max(1, n - 1)) + 1):
                for size in (n - 1, n, n + 1):
                    if size <= 0:
                        continue
                    m = difflib.SequenceMatcher(None, a, heard[i:i + size], autojunk=False)
                    if m.quick_ratio() >= LINE_RATIO and m.ratio() >= LINE_RATIO:
                        best = i + size
                        break
                if best is not None:
                    break
            if best is not None:
                hits, pos = hits + 1, best
            else:  # sung but not heard as itself: move on by about half its length, the take goes on meanwhile
                pos = min(len(heard), pos + n // 2)
        return hits
    if not lines:
        return 0.0
    return max(ordered(heard) for heard in texts) / len(lines) if texts else 0.0


def separate(args):
    """The lead vocal of the take, by demucs (any vocal separator works: hear only needs a vocals file)."""
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, '-m', 'demucs', '--two-stems', 'vocals', '-o', str(work / 'demucs'), str(args.audio)],
                   check=True)
    found = sorted((work / 'demucs').rglob('vocals.*'))
    if not found:
        raise SystemExit('YUE2_LYRIC_CHECK_NO_VOCALS')
    shutil.copyfile(found[0], work / 'vocals.wav')
    print(work / 'vocals.wav')


def hear(args):
    from asr_step import prompt_echo
    import whisper
    request = json.loads(Path(args.request).read_text(encoding='utf-8'))
    lines = planned_lines(request)
    model = whisper.load_model(paths.WHISPER, device='cuda')
    texts = []
    for prompt in ((PROMPT, None) if args.language == 'zh' else (None,)):
        result = model.transcribe(str(args.vocals), language=args.language or None, initial_prompt=prompt,
                                  carry_initial_prompt=bool(prompt), condition_on_previous_text=False, temperature=0,
                                  verbose=None)
        texts.append(''.join(plain(s['text']) for s in result['segments']
                             if not (prompt and prompt_echo(s['text'], prompt))))
    share = heard_share(lines, texts)
    source = 0.0
    if args.asr and Path(args.asr).is_file():
        asr = json.loads(Path(args.asr).read_text(encoding='utf-8'))
        source = heard_share(lines, [''.join(plain(s['text']) for s in asr.get('segments', []))])
    passed = share >= PASS_SHARE or share >= PASS_OF_SOURCE * source
    score = {'lyric_lines': len(lines), 'lyric_share': round(share, 3), 'source_share': round(source, 3),
             'lyric_passed': bool(passed)}
    Path(args.out).write_text(json.dumps(score), encoding='utf-8')
    print(json.dumps(score))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='mode', required=True)
    a = sub.add_parser('separate')
    a.add_argument('--audio', required=True)
    a.add_argument('--work', required=True)
    b = sub.add_parser('hear')
    b.add_argument('--vocals', required=True)
    b.add_argument('--request', required=True)
    b.add_argument('--asr')
    b.add_argument('--language', default='zh')
    b.add_argument('--out', required=True)
    args = parser.parse_args()
    (separate if args.mode == 'separate' else hear)(args)


if __name__ == '__main__':
    main()
