"""Timed lyrics as the plan (plan.lrc_*, line_starts, place_timed), language detection, the ASR noise filters and the
tie fix. No GPU, no network. Every lyric line here is a placeholder made up for the tests.

    python -m pytest -q
"""
import sys
from fractions import Fraction
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'yue2_cover')]
import asr_step  # noqa: E402
import job  # noqa: E402
import plan  # noqa: E402

ZH = [f'第{n}行歌词在这里' for n in '一二三四五六']
EN = [f'line {n} of the song' for n in ('one', 'two', 'three', 'four', 'five', 'six')]


def timed(lines, start=10.0, step=4.0):
    return [(start + i * step, line) for i, line in enumerate(lines)]


def heard_units(entries, shift=0.0):
    """What Whisper's words look like: every syllable of a line, 0.3 s apart, from the line's time + shift."""
    return [(u.lower(), t + shift + 0.3 * k) for t, line in entries for k, u in enumerate(plan.UNIT.findall(line))]


class FakeScore:
    """A melody with a note every half second from 10 s to 38 s (quarter = second)."""

    class vocal:
        notes = [(Fraction(10) + Fraction(j, 2), 60, Fraction(1, 2)) for j in range(57)]

    def seconds(self, q):
        return float(q)


# --- which language a text is sung in ----------------------------------------------------------------------------

@pytest.mark.parametrize('lines, expected', [
    (ZH, 'zh'), (EN, 'en'), (['さくらが咲いたよ', 'きゃりーぱみゅぱみゅ'], 'ja'), (['사랑해요 우리 함께', '노래를 불러요'], 'ko'),
    ([], 'other'), (['1234 !!'], 'other')])
def test_the_language_comes_from_the_sung_units(lines, expected):
    assert plan.text_language(lines) == expected


def test_whispers_guess_is_only_the_last_word():
    assert plan.text_language([], 'ja') == 'ja' and plan.text_language(['1234'], 'en') == 'en'
    assert plan.text_language(ZH, 'en') == 'zh'


def test_a_mostly_english_song_with_a_few_chinese_characters_is_english():
    assert plan.text_language(['hello world 你', 'we are the stars tonight']) == 'en'


def test_the_style_line_names_the_sung_language():
    assert plan.style_for('ja', '').startswith('Japanese female vocal')
    assert plan.style_for('ko', 'k-pop').startswith('Korean female vocal') and 'k-pop' in plan.style_for('ko', 'k-pop')


# --- timed lyrics given as the lyrics ------------------------------------------------------------------------------

def test_lyrics_with_four_or_more_timestamps_are_timed_lyrics():
    text = '\n'.join(f'[00:{int(t):02d}.00]{line}' for t, line in timed(ZH[:4]))
    assert plan.lrc_lines(text) == timed(ZH[:4])
    assert plan.lrc_lines('\n'.join(text.splitlines()[:3])) == []              # too few to trust
    assert plan.lrc_lines('\n'.join(ZH)) == [] and plan.lrc_lines('') == []    # plain lyrics are not timed lyrics


# --- do the timed lyrics fit this recording ---------------------------------------------------------------------------

def test_timed_lyrics_heard_two_seconds_late_fit_at_a_shift_of_two():
    lrc = timed(ZH)
    fit = plan.lrc_fit(lrc, heard_units(lrc, 2.0))
    assert fit['shift'] == 2.0 and fit['spread'] == 0.0
    assert fit['on_time'] == fit['heard'] == fit['lines'] == 6 and fit['share'] == 1.0
    assert all(abs(s - 2.0) < 0.01 for _, s in fit['anchors'])


def test_a_pause_this_version_adds_moves_only_the_lines_after_it():
    lrc = timed(ZH)
    units = heard_units(lrc[:3], 1.0) + heard_units(lrc[3:], 3.0)
    fit = plan.lrc_fit(lrc, units)
    mapped = plan.mapped_by(fit['anchors'])
    assert mapped(10.0) == pytest.approx(11.0, abs=0.01) and mapped(30.0) == pytest.approx(33.0, abs=0.01)


def test_timed_lyrics_of_another_song_do_not_fit():
    lrc = timed(ZH)
    other = timed([f'完全不同的{n}首歌曲歌词' for n in '一二三四五六'])
    assert plan.lrc_fit(lrc, heard_units(other)) is None


def test_the_cover_share_tells_a_cut_matched_to_the_wrong_part_from_the_right_one():
    lrc = timed(ZH)
    right = {'segments': [{'start': t, 'end': t + 2.4, 'text': line} for t, line in lrc]}
    fit = plan.lrc_fit(lrc, heard_units(lrc), asr=right)
    assert fit['cover'] == 1.0
    wrong = {'segments': [{'start': t, 'end': t + 2.4, 'text': f'别的歌里唱的第{n}句话'} for (t, _), n in zip(lrc, '一二三四五六')]}
    assert plan.lrc_fit(lrc, heard_units(lrc), asr=wrong)['cover'] == 0.0


def test_a_line_repeated_in_the_song_votes_for_the_shift_of_every_place_it_was_heard():
    lrc = timed([ZH[0], ZH[1], ZH[2], ZH[3], ZH[0]])
    units = heard_units(lrc, 5.0)
    pairs = plan.line_starts(lrc, units)
    assert [round(h - t, 1) for t, h in pairs] == [5.0] * 5


def test_the_shift_found_from_the_melody_puts_line_starts_on_phrase_starts():
    lrc = timed(ZH)
    phrases = [t + 3.0 for t, _ in lrc]
    assert plan.note_shift([t for t, _ in lrc], phrases) == pytest.approx(3.0, abs=0.05)


# --- timed lines placed on the melody ------------------------------------------------------------------------------

def test_each_timed_line_takes_the_notes_inside_its_own_time():
    lines, lyric, times, reliable = plan.place_timed(timed(ZH), lambda t: t, FakeScore(), disjoint=True)
    assert lines == ZH and len(lyric) == 6 * 8 == len(times)
    assert times[0] == pytest.approx(10.0, abs=0.01) and times == sorted(times)
    assert sum(reliable) == 6                        # a line's start is known, inside it the melody decides
    firsts = [times[k] for k in range(len(lyric)) if k == 0 or lyric[k][1] != lyric[k - 1][1]]
    assert all(abs(a - t) < 0.5 for a, (t, _) in zip(firsts, timed(ZH)))


def test_a_line_with_no_notes_in_its_time_is_left_out():
    lrc = timed(ZH) + [(120.0, 'a line far past the melody')]
    lines, _, _, _ = plan.place_timed(lrc, lambda t: t, FakeScore(), disjoint=True)
    assert lines == ZH


# --- the made-up and echoed segments Whisper writes -------------------------------------------------------------------

def test_a_unit_written_over_and_over_in_a_short_stretch_is_noise_not_a_lyric():
    assert asr_step.heard_noise('你好吗' * 74, 1.4)
    assert asr_step.heard_noise('啊' * 20, 8.0)                          # one short unit over nearly all of it
    assert not asr_step.heard_noise(ZH[0], 3.0) and not asr_step.heard_noise('', 1.0)
    assert asr_step.heard_noise('今天我们一起走过了很长很长的路', 0.5)       # 15 characters in half a second


def test_a_segment_that_only_echoes_the_prompt_is_dropped():
    prompt = job.CHINESE_PROMPT
    assert asr_step.prompt_echo('唱歌曲的歌曲的歌曲', prompt)
    assert asr_step.prompt_echo('歌词的歌词的歌词', prompt)
    assert not asr_step.prompt_echo(ZH[0], prompt) and not asr_step.prompt_echo('唱歌', prompt)
    assert not asr_step.prompt_echo('whatever', '')


def test_segments_keep_real_lines_and_drop_the_noise():
    result = {'segments': [
        {'start': 1.0, 'end': 4.0, 'text': ZH[0], 'words': []},
        {'start': 5.0, 'end': 6.4, 'text': '你好吗' * 74, 'words': []},
        {'start': 7.0, 'end': 10.0, 'text': '唱歌曲的歌曲的歌曲', 'words': []}]}
    assert [s['text'] for s in asr_step.segments_of(result, job.CHINESE_PROMPT)] == [ZH[0]]
    assert len(asr_step.segments_of(result)) == 2                         # without a prompt there is no echo to find


def test_a_span_past_the_end_of_the_audio_is_cut_at_the_end():
    assert asr_step.clamp_spans([[10.0, 20.0], [28.0, 40.0], [29.8, 35.0]], 30.0) == [[10.0, 20.0], [28.0, 30.0]]


# --- the tie that used to stop the plan ---------------------------------------------------------------------------------

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
c4c4e4g4|"C"c4-c4c4z4|
V: Ins
Z2|
'''


def test_a_held_tail_is_tied_to_the_note_before_it():
    assert plan.tie_notes(SCORE, [Fraction(1)]).splitlines()[10] == 'c4-c4e4g4|"C"c4-c4c4z4|'


def test_a_note_the_transcription_already_tied_does_not_stop_the_plan():
    # The note before the onset is the tail token of a tied note; its own start is no note start (this was a KeyError).
    assert plan.tie_notes(SCORE, [Fraction(6)]).splitlines()[10] == 'c4c4e4g4|"C"c4-c4-c4z4|'


def test_ties_between_different_pitches_are_refused():
    with pytest.raises(ValueError, match='YUE2_TIE_MISMATCH'):
        plan.tie_notes(SCORE, [Fraction(2)])
