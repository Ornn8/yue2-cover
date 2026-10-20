"""CPU-only checks: lyric cleanup, the reference-library score format and the beat tempo fit.

    python tests/test_plan_and_tools.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'yue2_cover'), str(ROOT / 'tools')]
from plan import lyric_lines  # noqa: E402
from add_reference import beat_tempo, one_section  # noqa: E402
from abc_tools import parse_abc  # noqa: E402


def asr(*texts):
    return {'segments': [{'text': t} for t in texts]}


# Whisper's credits, subtitle lines and stray English over a Chinese song's interlude are not sung.
lines, source = lyric_lines('', asr('作词 作曲 作曲 作曲', '该怎么走 我却只剩回头。', 'Zither Harp', '牵手和分手来自同一双手',
                                   '字幕志愿者 某某'))
assert source == 'asr' and lines == ['该怎么走 我却只剩回头', '牵手和分手来自同一双手'], lines
# English songs keep their lines; the given lyrics are never filtered.
assert lyric_lines('', asr('I will always love you', 'oh 宝贝', 'and I'))[0] == ['I will always love you', 'oh 宝贝', 'and I']
assert lyric_lines('第一句\nZither Harp', asr('x'))[0] == ['第一句', 'Zither Harp']

# A clip's score keeps one section and drops trailing bars that are only rests.
SCORE = '''X:1
T:
M:4/4
L:1/16
Q:1/4=60
V: Vocal clef=treble name="Vocal Melody" snm="Vocal"
V: Ins clef=treble name="Ins Melody" snm="Inst."
K:C
% verse
V: Vocal
z8"C"z4G4|"C"c6c2c2z2e4|"G"d4z2c2d2z2e2dc|"C"c4z2c2"C7"e4g4|
V: Ins
Z4|
% chorus
V: Vocal
"F"aaz8z2a4|"C"g4z2e2e2z2c4|"G"d6c2"E7"d2z2e2dc|"Fmaj7"c4z2A2"G7"A4G4|
V: Ins
Z4|
V: Vocal
"G7"z2z12z2|
V: Ins
Z|
'''
text = one_section(SCORE)
assert text.count('% ') == 1 and '% verse' in text and '"G7"z2z12z2' not in text
score = parse_abc(text)
assert score.voices['Vocal'].time == 32 and score.bpm == 60

# Tempo from beats: a steady 0.5 s beat with jitter is 120 BPM.
beats = [0.3 + 0.5 * i + (0.02 if i % 2 else -0.02) for i in range(40)]
assert abs(beat_tempo(beats) - 120) < 0.5
print('plan and tool tests passed')
