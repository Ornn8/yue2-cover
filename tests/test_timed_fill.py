"""The timed fill (plan.timed_fill, plan.load_timed): a published line no heard line matched is sung at its published
time when those times fit this recording. No network, no GPU. Every lyric line here is a placeholder made up for the
tests.

    python -m pytest -q
"""
import json
import math
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'yue2_cover')]
import plan  # noqa: E402

# Six placeholder lines of 8 syllables (Chinese) and of 5 (English), published four seconds apart from 10 s on.
ZH = [f'第{n}行歌词在这里' for n in '一二三四五六']
EN = [f'line {n} of the song' for n in ('one', 'two', 'three', 'four', 'five', 'six')]
LINES = ZH
STRANGER = '多出来的一句占位歌词'      # a heard line that no published line is like


def published(lines, start=10.0, step=4.0):
    return [(start + i * step, line) for i, line in enumerate(lines)]


# The melody: a note every half second from 10 s to 38 s; the first and last second it covers.
ONSETS = [10 + 0.5 * j for j in range(57)]
BOUNDS = (9.7, 38.5)


def sung(entries):
    """(lines, lyric, times, reliable) of lines sung at the given [(start second, text)], a syllable every 0.3 s, all
    heard reliably."""
    lines = [text for _, text in entries]
    lyric = [(u.lower(), li) for li, text in enumerate(lines) for u in plan.UNIT.findall(text)]
    times = [start + 0.3 * k for start, text in entries for k, _ in enumerate(plan.UNIT.findall(text))]
    return lines, lyric, times, [True] * len(lyric)


def fill(heard, timed, onsets=ONSETS, bounds=BOUNDS):
    return plan.timed_fill(*sung(heard), timed, onsets, bounds)


def times_of(result, li):
    """The times of one line's syllables in a timed_fill result."""
    return [t for (_, l), t in zip(result[1], result[2]) if l == li]


def without(entries, *gone):
    return [e for i, e in enumerate(entries) if i not in gone]


# --- the three cases: filled, a different recording, no notes ----------------------------------------------------

@pytest.mark.parametrize('lines', [ZH, EN])
def test_a_line_never_heard_is_sung_at_its_published_time_on_the_melodys_notes(lines):
    timed = published(lines)
    heard = without(timed, 2)                        # the third line was never heard
    result = fill(heard, timed)
    out_lines, lyric, times, reliable, filled = result
    n = len(plan.UNIT.findall(lines[2]))
    assert filled == 1 and out_lines == lines
    # Published at 18 s; the melody's notes there are 18.0, 18.5, 19.0 ...: one note for each syllable.
    assert times_of(result, 2) == [18.0 + 0.5 * j for j in range(n)]
    assert [u for u, l in lyric if l == 2] == [u.lower() for u in plan.UNIT.findall(lines[2])]
    # Every other line keeps its syllables, times and reliability; only the new ones are unreliable.
    kept = sung(heard)
    assert [(u, t, r) for (u, l), t, r in zip(lyric, times, reliable) if l != 2] == \
        [(u, t, r) for (u, _), t, r in zip(kept[1], kept[2], kept[3])]
    assert [r for (_, l), r in zip(lyric, reliable) if l == 2] == [False] * n
    assert sorted(l for _, l in lyric) == [l for _, l in lyric]          # the lines are numbered in order


def test_a_misheard_line_is_replaced_by_the_line_published_at_that_time():
    timed = published(LINES)
    misheard = '第三行歌词在这理'                  # one character off: no heard line matches the published line
    heard = [*timed[:2], (18.0, misheard), *timed[3:]]
    result = fill(heard, timed)
    assert result[4] == 1 and result[0] == LINES and misheard not in result[0]
    assert times_of(result, 2) == [18.0 + 0.5 * j for j in range(8)]


def test_a_line_may_start_on_a_note_just_before_its_time_and_replaces_what_was_heard_over_that_note():
    timed = published(LINES)
    short = '占位的一句'                           # heard over 16.6-17.8 s, a note short of the published 18 s
    heard = [*timed[:2], (16.6, short), *timed[3:]]
    onsets = sorted([*ONSETS, 17.75])               # the melody's note for the line starts 0.25 s early
    result = fill(heard, timed, onsets)
    assert result[4] == 1 and result[0] == LINES
    assert times_of(result, 2) == [17.75, 18.0, 18.5, 19.0, 19.5, 20.0, 20.5, 21.0]
    assert fill(heard, timed)[0] == [*LINES[:2], short, LINES[2], *LINES[3:]]   # with no such note it is kept


def test_heard_lines_nothing_was_published_over_stay():
    timed = published(LINES)
    heard = [*without(timed, 2), (32.5, STRANGER)]   # after the last published line
    out_lines, _, _, _, filled = fill(heard, timed)
    assert filled == 1 and out_lines == [*LINES, STRANGER]


def test_a_recording_a_few_seconds_later_than_the_published_times_fills_at_its_own_times():
    timed = published(LINES)
    heard = [(t + 2.5, text) for t, text in without(timed, 2)]
    result = fill(heard, timed)
    assert result[4] == 1 and result[0] == LINES
    assert times_of(result, 2) == [20.5 + 0.5 * j for j in range(8)]    # 18 s + 2.5 s


def test_the_first_line_of_a_repeated_chorus_that_was_not_heard_the_first_time_is_filled():
    chorus = [(10.0, LINES[0]), (30.0, LINES[0])]
    timed = sorted([*chorus, *[(14.0 + 4 * i, text) for i, text in enumerate(LINES[1:5])], (34.0, LINES[5])])
    heard = without(timed, 0)                        # only the second time was heard
    result = fill(heard, timed)
    assert result[4] == 1 and result[0] == [LINES[0], *LINES[1:5], LINES[0], LINES[5]]
    assert times_of(result, 0) == [10.0 + 0.5 * j for j in range(8)]


def test_a_differently_timed_recording_is_left_unchanged():
    timed = published(LINES)
    live = [(t * 1.2, text) for t, text in without(timed, 2)]      # another recording: slower, drifting from the lyrics
    expected = sung(live)
    result = plan.timed_fill(*expected, timed, ONSETS, BOUNDS)
    assert result == (*expected, 0)


def test_the_published_times_are_trusted_only_with_four_matched_lines():
    timed = published(LINES)
    few = sung(timed[:3])
    assert plan.timed_fill(*few, timed, ONSETS, BOUNDS) == (*few, 0)       # three agree perfectly: still not enough
    four = fill(timed[:4], timed)
    assert four[4] == 2 and four[0] == LINES                              # four are: the last two lines come in


def test_most_matched_lines_must_start_close_to_their_published_times():
    timed = published(LINES)
    heard = without(timed, 2)

    def drifted(*seconds):
        return [(t + d, text) for d, (t, text) in zip(seconds, heard)]

    assert fill(drifted(2, 0, 0, 0, 2), timed)[4] == 1       # 3 of 5 start on time: 60% is enough
    assert fill(drifted(2, 2, 0, 0, 4), timed)[4] == 0       # 2 of 5 are within a second of the median offset
    assert fill(drifted(0, 1.5, 3, 4.5, 6), timed)[4] == 0   # the offset grows line by line, as in a different tempo


def test_no_line_is_filled_where_the_melody_has_no_notes():
    timed = published(LINES)
    misheard = '第三行歌词在这理'
    heard = [*timed[:2], (18.0, misheard), *timed[3:]]
    rest = [on for on in ONSETS if not 16.5 < on < 22]               # a rest in the melody over 17-22 s
    unchanged = sung(heard)
    assert fill(heard, timed, rest) == (*unchanged, 0)               # what was heard there stays
    assert fill(heard, timed, [])[4] == 0                            # no notes at all


def test_a_line_needs_a_note_for_half_its_syllables_and_is_then_spread_over_its_time():
    timed = published(LINES)
    heard = without(timed, 2)
    few = [on for on in ONSETS if not 17.7 < on < 21.85] + [18.0, 19.0, 20.0]       # 3 notes for 8 syllables
    assert fill(heard, timed, sorted(few))[4] == 0
    half = sorted(few + [21.0])                                      # 4 notes for 8 syllables: enough
    result = fill(heard, timed, half)
    assert result[4] == 1 and result[0] == LINES
    # Not on the four notes: spread evenly from 18 s to where the line ends (21.85 s, before the next line).
    assert times_of(result, 2) == pytest.approx([18 + (21.85 - 18) * j / 7 for j in range(8)])


def test_a_line_beyond_the_end_of_the_melody_is_not_sung():
    timed = published(LINES)
    heard = timed[:5]                                                # the last line, published at 30 s, was never heard
    assert fill(heard, timed)[4] == 1
    assert fill(heard, timed, ONSETS, (9.7, 29.0))[4] == 0           # the melody is taken to end before it


# --- load_timed ------------------------------------------------------------------------------------------------

def write_lookup(work, value):
    (work / 'lyrics-lookup.json').write_text(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False),
                                             encoding='utf-8', newline='\n')


def test_load_timed_gives_the_seconds_and_text_of_the_lines_that_have_a_time(tmp_path):
    write_lookup(tmp_path, {'song': {'id': 7}, 'score': 0.8, 'lines': [
        {'t': 1.5, 'text': ZH[0]}, {'text': ZH[1]}, {'t': '3', 'text': ZH[2]}, {'t': 7, 'text': ZH[3]}]})
    assert plan.load_timed(tmp_path) == [(1.5, ZH[0]), (7.0, ZH[3])]
    assert plan.load_lookup(tmp_path)[0] == ZH[:4]                   # the published lines are still all of them


def test_load_timed_skips_times_that_are_not_numbers(tmp_path):
    write_lookup(tmp_path, '{"song": {"id": 7}, "lines": [{"t": NaN, "text": "a"}, {"t": Infinity, "text": "b"}, '
                           '{"t": 2, "text": "c"}]}')
    assert plan.load_timed(tmp_path) == [(2.0, 'c')]
    assert all(math.isfinite(t) for t, _ in plan.load_timed(tmp_path))


@pytest.mark.parametrize('value', [None, '{broken', '[]', {'song': None, 'why': 'no close lyrics'},
                                   {'song': {'id': 7}, 'lines': []}, {'song': {'id': 7}}, {'song': 'x', 'lines': 3},
                                   {'song': {'id': 7}, 'lines': [{'t': 1.0}]}])
def test_load_timed_is_empty_when_there_is_nothing_usable(tmp_path, value):
    if value is not None:
        write_lookup(tmp_path, value)
    assert plan.load_timed(tmp_path) == []


# --- plan.build ------------------------------------------------------------------------------------------------

class FakeScore:
    """A score with a note every half second from 10 s to 38 s; only what build() asks of the real one."""

    def __init__(self, score_dir):
        self.text = '% verse\n% chorus\n'
        self.score = types.SimpleNamespace(bpm=100)
        self.vocal = types.SimpleNamespace(notes=[(on, 60, 0.5) for on in ONSETS])

    def seconds(self, q):
        return float(q)

    def sections(self):
        return [('verse', 0.0, 40.0), ('chorus', 40.0, float('inf'))]


def heard_segment(text, start):
    return {'start': start, 'end': start + 0.3 * len(text), 'text': text,
            'words': [{'w': ch, 's': start + 0.3 * k, 'e': start + 0.3 * k + 0.25} for k, ch in enumerate(text)]}


def build_with(monkeypatch, work, heard, lyrics=''):
    monkeypatch.setattr(plan, 'Score', FakeScore)
    monkeypatch.setattr(plan, 'load_references', lambda: [{'clip': 'ref-1', 'lyrics': ['reference line'], 'true_bpm': 100}])
    monkeypatch.setattr(plan, 'choose_reference', lambda bpm, refs: (refs[0], 1.0))
    monkeypatch.setattr(plan, 'reference_plan', lambda ref, text: (
        '% verse\n% verse\n% chorus\n', {'rate': 1.0, 'lead_seconds': 0.0, 'samples': 48000, 'frames': 25,
                                          'boundary_quarters': '0', 'composite_seconds': 100.0}))
    monkeypatch.setattr(plan, 'held_tails', lambda *args: [])
    asr = {'language': 'zh', 'segments': [heard_segment(text, start) for start, text in heard]}
    (work / 'asr.json').write_text(json.dumps(asr, ensure_ascii=False), encoding='utf-8', newline='\n')
    write_lookup(work, {'song': {'id': 22, 'name': 'x', 'artist': 'y'}, 'score': 0.9,
                        'lines': [{'t': t, 'text': text} for t, text in published(LINES)]})
    result = plan.build(work, lyrics, '')
    request = json.loads((work / 'request.json').read_text(encoding='utf-8'))
    return result, [l for l in request['lyrics'].splitlines() if l and not l.startswith('[')][1:]   # minus the reference


def test_build_sings_a_line_nobody_heard_at_its_published_time(monkeypatch, tmp_path):
    heard = without(published(LINES), 2)
    result, sung_lines = build_with(monkeypatch, tmp_path, heard)
    assert result['timed_lines'] == 1 and result['lines'] == 6 and result['lyric_source'] == 'asr+lookup:0'
    assert sung_lines == LINES                                       # the missing line is back, in its place
    assert json.loads((tmp_path / 'plan.json').read_text(encoding='utf-8'))['timed_lines'] == 1


def test_build_leaves_another_recording_as_it_was_heard(monkeypatch, tmp_path):
    heard = [(t * 1.2, text) for t, text in without(published(LINES), 2)]
    result, sung_lines = build_with(monkeypatch, tmp_path, heard)
    assert result['timed_lines'] == 0 and result['lines'] == 5 and sung_lines == without(LINES, 2)


def test_build_never_fills_around_the_users_own_lyrics(monkeypatch, tmp_path):
    heard = without(published(LINES), 2)
    result, sung_lines = build_with(monkeypatch, tmp_path, heard, lyrics='\n'.join(without(LINES, 2)) + '\n')
    assert result['lyric_source'] == 'user' and result['timed_lines'] == 0 and sung_lines == without(LINES, 2)
