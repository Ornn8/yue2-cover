"""Add one clip of your singer to the reference library (refs.json + <id>/score.abc + clips/<id>.<ext>).

    python tools/add_reference.py --clip clip.flac --lyrics-file clip-lyrics.txt --id singer-ballad-01

A good clip: 25-40 s of the singer's own recording (vocals over the accompaniment, as in a released song),
starting just before a phrase, no long silence. Add clips at several tempos: each cover uses the clip nearest the
song's tempo and stretches it at most 1.18x, so a library spanning ~60-120 BPM covers most songs.

SheetSage2 transcribes the clip; the clip's tempo comes from its beats, its first sung note from the vocal melody.
The score keeps one section (labelled verse) and drops trailing bars that are only rests. Check the printed
numbers; --true-bpm and --first-note-seconds override the measured ones.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = HERE.parent / 'yue2_cover'
sys.path.insert(0, str(CODE))
import paths  # noqa: E402
from abc_tools import parse_abc  # noqa: E402


def lab_starts(path):
    return [float(line.split()[0]) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def beat_tempo(beats):
    """Least-squares seconds per beat over all beats; steadier than the median gap for short clips."""
    n = len(beats)
    if n < 4:
        raise SystemExit('Too few beats in the clip; use a longer clip or pass --true-bpm')
    mx, my = (n - 1) / 2, sum(beats) / n
    slope = sum((i - mx) * (t - my) for i, t in enumerate(beats)) / sum((i - mx) ** 2 for i in range(n))
    return 60 / slope


def one_section(text):
    """The library format: a single `% verse` section, no trailing rest-only bars."""
    lines = text.splitlines()
    head = []
    while lines and not lines[0].startswith('% '):
        head.append(lines.pop(0))
    groups = []
    for line in lines:
        if line.startswith('% '):
            continue
        if line.startswith('V: Vocal'):
            groups.append([])
        if not groups:
            raise SystemExit('Unexpected score layout near: ' + line[:60])
        groups[-1].append(line)

    def rests_only(group):
        music = [line for line in group if not re.match(r'^[A-Z]:', line)]
        return all(re.fullmatch(r'[zZ0-9|\s]*', re.sub(r'"[^"]*"', '', line)) for line in music)

    while len(groups) > 1 and rests_only(groups[-1]):
        groups.pop()
    return '\n'.join(head + ['% verse'] + [line for g in groups for line in g]) + '\n'


def seconds(path):
    out = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', str(path)],
                         capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--clip', required=True, type=Path)
    parser.add_argument('--lyrics-file', required=True, type=Path, help='the lines sung in the clip, one per line')
    parser.add_argument('--id', required=True, help='a name for the clip: letters, digits, - and _')
    parser.add_argument('--true-bpm', type=float)
    parser.add_argument('--first-note-seconds', type=float)
    parser.add_argument('--score-dir', type=Path, help='an existing SheetSage2 output folder for this clip')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', args.id):
        raise SystemExit('--id: letters, digits, ".", "-" and "_" only')
    lyrics = [line.strip() for line in args.lyrics_file.read_text(encoding='utf-8').splitlines() if line.strip()]
    with tempfile.TemporaryDirectory() as tmp:
        score_dir = args.score_dir or Path(tmp) / 'score'
        if not args.score_dir:
            env = {**os.environ, 'PYTHONPATH': str(CODE), 'HF_HOME': str(paths.HF_HOME)}
            subprocess.run([paths.SAGE_PYTHON, str(CODE / 'score_step.py'), str(args.clip), str(score_dir)], env=env,
                           check=True)
        abc = score_dir / 'target-score.abc'
        text = one_section((abc if abc.exists() else score_dir / 'score.abc').read_text(encoding='utf-8'))
        score = parse_abc(text)
        notes = score.voices['Vocal'].notes
        if not notes:
            raise SystemExit('No sung notes found in the clip')
        vocal = score_dir / 'melody_vocal.lab'
        onsets = lab_starts(vocal if vocal.exists() else score_dir / 'melody_full.lab')
        entry = {
            'clip': args.id, 'path': f'clips/{args.id}{args.clip.suffix.lower()}', 'score_bpm': score.bpm,
            'true_bpm': round(args.true_bpm or beat_tempo(lab_starts(score_dir / 'beat.lab')), 3),
            'first_note_seconds': args.first_note_seconds if args.first_note_seconds is not None else onsets[0],
            'first_note_quarter': str(Fraction(notes[0][0])), 'seconds': round(seconds(args.clip), 2),
            'score_quarters': str(Fraction(score.voices['Vocal'].time)), 'lyrics': lyrics,
        }
    root = paths.REFERENCES
    (root / args.id).mkdir(parents=True, exist_ok=True)
    (root / args.id / 'score.abc').write_text(text, encoding='utf-8')
    (root / 'clips').mkdir(exist_ok=True)
    shutil.copyfile(args.clip, root / entry['path'])
    index = root / 'refs.json'
    refs = json.loads(index.read_text(encoding='utf-8')) if index.exists() else []
    refs = [r for r in refs if r['clip'] != args.id] + [entry]
    index.write_text(json.dumps(refs, ensure_ascii=False, indent=1), encoding='utf-8')
    end = (Fraction(entry['score_quarters']) - Fraction(entry['first_note_quarter'])) * 60 / Fraction(str(entry['true_bpm']))
    print(json.dumps({**entry, 'score_ends_seconds': round(entry['first_note_seconds'] + float(end), 2)},
                     ensure_ascii=False, indent=1))
    if entry['first_note_seconds'] + float(end) > entry['seconds'] + 1.5:
        print('Note: the score runs past the end of the clip; a shorter score or a longer clip aligns better.')


if __name__ == '__main__':
    main()
