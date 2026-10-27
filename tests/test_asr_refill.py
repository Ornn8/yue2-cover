"""Sung stretches the whole-song Whisper pass missed are heard again (asr_step.py). No GPU, no Whisper model:
whisper is replaced by a fake whose answers the tests choose.

    python -m pytest -q
"""
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'yue2_cover')]
import asr_step  # noqa: E402
import job  # noqa: E402


def segment(text, *starts):
    """A segment in asr_step's own format, with one word starting at each of `starts`."""
    return {'start': starts[0], 'end': starts[-1] + 0.3, 'text': text,
            'words': [{'w': 'w', 's': s, 'e': s + 0.3} for s in starts]}


def whisper_segment(text, *starts):
    """The same in the shape openai-whisper returns."""
    return {'start': starts[0], 'end': starts[-1] + 0.3, 'text': ' ' + text,
            'words': [{'word': 'w', 'start': s, 'end': s + 0.3} for s in starts]}


# One long phrase sung without a breath: 40 half-second notes from 10 s to 30 s.
PHRASE = [(10 + 0.5 * i, 10.5 + 0.5 * i) for i in range(40)]
# Whisper heard its first and last seconds only.
FIRST_HALF = segment('first half', 10, 11, 12, 13, 14)
LAST_HALF = segment('last half', 26, 27, 28, 29, 29.5)
MISSED = [[15.0, 25.0]]       # notes 15.5..24.5 have no word within 1 s, padded by 0.5 s


# --- which stretches were not heard ---------------------------------------------------------------------------

def test_a_fully_missed_stretch_inside_a_long_phrase_is_found():
    assert asr_step.unheard([FIRST_HALF, LAST_HALF], PHRASE) == MISSED


def test_a_heard_stretch_is_not_found():
    everywhere = segment('whole phrase', *range(10, 30))
    assert asr_step.unheard([everywhere], PHRASE) == []
    assert asr_step.unheard([], []) == []


def test_credit_and_subtitle_segments_do_not_count_as_hearing():
    credits = segment('作词 作曲 字幕志愿者', 16, 18, 20, 22, 24)
    assert asr_step.unheard([FIRST_HALF, credits, LAST_HALF], PHRASE) == MISSED
    # The same word starts in a real lyric line do hear the stretch.
    sung = segment('some lyric line', 16, 18, 20, 22, 24)
    assert asr_step.unheard([FIRST_HALF, sung, LAST_HALF], PHRASE) == []


def test_unheard_stretches_shorter_than_two_seconds_are_left_alone():
    assert asr_step.unheard([], [(10.0, 10.5), (10.5, 11.5)]) == []
    assert asr_step.unheard([], [(10.0, 12.0)]) == [[9.5, 12.5]]


def test_notes_less_than_1_5_seconds_apart_form_one_stretch_and_a_longer_rest_splits_it():
    assert asr_step.unheard([], [(10.0, 11.5), (12.5, 14.0)]) == [[9.5, 14.5]]
    assert asr_step.unheard([], [(10.0, 12.0), (13.5, 15.5)]) == [[9.5, 12.5], [13.0, 16.0]]


def test_a_stretch_never_starts_before_the_song():
    assert asr_step.unheard([], [(0.2, 3.0)]) == [[0.0, 3.5]]


def test_sung_notes_come_from_the_vocal_melody_file_in_time_order(tmp_path):
    assert asr_step.sung_notes(tmp_path) == []
    (tmp_path / 'melody_vocal.lab').write_text('3.0 4.0 60\n\n1.0 2.5 62\n', encoding='utf-8', newline='\n')
    assert asr_step.sung_notes(tmp_path) == [(1.0, 2.5), (3.0, 4.0)]


# --- merging the re-heard segments ----------------------------------------------------------------------------

def test_reheard_segments_take_the_place_of_what_the_whole_pass_put_in_the_span():
    before = segment('before', 1, 2)
    credit = segment('作词 作曲', 16, 17)               # a credit line where a verse was sung
    after = segment('after', 30, 31)
    new = segment('new verse', 16, 18, 20, 22)
    new_late = segment('new verse, end', 23, 24)
    outside = segment('outside the span', 40, 41)        # clip decoding can reach past a span; that part is dropped
    merged = asr_step.merge_heard([after, before, credit], [outside, new_late, new], MISSED)
    assert [s['text'] for s in merged] == ['before', 'new verse', 'new verse, end', 'after']


def test_a_segment_is_judged_by_its_midpoint():
    straddling = {'start': 24.0, 'end': 27.0, 'text': 'straddling', 'words': []}    # midpoint 25.5: outside
    on_the_edge = {'start': 24.0, 'end': 26.0, 'text': 'on the edge', 'words': []}  # midpoint 25.0: the span ends there
    inside = {'start': 23.0, 'end': 25.0, 'text': 'inside', 'words': []}            # midpoint 24.0: inside
    again = [dict(straddling, text='again straddling'), dict(on_the_edge, text='again edge'),
             dict(inside, text='again inside')]
    merged = asr_step.merge_heard([straddling, on_the_edge, inside], again, MISSED)
    assert [s['text'] for s in merged] == ['again inside', 'straddling', 'on the edge']


# --- the Whisper calls ----------------------------------------------------------------------------------------

class FakeModel:
    """Stands in for a loaded Whisper model: the whole-song call and the clip call get different answers."""

    def __init__(self, whole, clips=()):
        self.whole, self.clips, self.calls = whole, list(clips), []

    def transcribe(self, source, **options):
        self.calls.append((source, options))
        return {'language': 'zh', 'segments': self.clips if 'clip_timestamps' in options else self.whole}


def test_hear_again_decodes_only_the_missed_spans_with_the_same_options():
    options = dict(word_timestamps=True, condition_on_previous_text=False, temperature=0, verbose=None)
    model = FakeModel([], clips=[whisper_segment('new verse', 16, 18, 20, 22)])
    segments, spans = asr_step.hear_again(model, 'song.wav', [FIRST_HALF, LAST_HALF], PHRASE, options)
    assert spans == MISSED
    assert [s['text'] for s in segments] == ['first half', 'new verse', 'last half']
    assert segments[1]['words'][0] == {'w': 'w', 's': 16, 'e': 16.3}
    (source, called), = model.calls
    assert source == 'song.wav'
    assert called.pop('clip_timestamps') == [15.0, 25.0]      # flat: start, end, start, end ...
    assert called == options


def test_hear_again_does_not_decode_when_every_note_was_heard():
    model = FakeModel([])
    whole = [segment('whole phrase', *range(10, 30))]
    segments, spans = asr_step.hear_again(model, 'song.wav', whole, PHRASE, {})
    assert segments == whole and spans == [] and model.calls == []


# --- the step as the job runs it ------------------------------------------------------------------------------

def run_step(monkeypatch, tmp_path, model, *extra):
    monkeypatch.setitem(sys.modules, 'whisper', types.SimpleNamespace(load_model=lambda name, device: model))
    out = tmp_path / 'asr.json'
    monkeypatch.setattr(sys, 'argv', ['asr_step.py', str(tmp_path / 'song.wav'), str(out), 'prompt', *extra])
    asr_step.main()
    return json.loads(out.read_text(encoding='utf-8'))


def whole_song_model():
    return FakeModel([whisper_segment('first half', 10, 11, 12, 13, 14), whisper_segment('last half', 26, 27, 28, 29, 29.5)],
                     clips=[whisper_segment('new verse', 16, 18, 20, 22)])


def score_folder(tmp_path):
    score = tmp_path / 'score'
    score.mkdir()
    (score / 'melody_vocal.lab').write_text(''.join(f'{a} {b} 60\n' for a, b in PHRASE), encoding='utf-8', newline='\n')
    return score


def test_the_step_hears_the_missed_stretch_again_and_records_it(monkeypatch, tmp_path, capsys):
    model = whole_song_model()
    result = run_step(monkeypatch, tmp_path, model, str(score_folder(tmp_path)))
    assert [s['text'] for s in result['segments']] == ['first half', 'new verse', 'last half']
    assert result['refilled'] == MISSED and result['language'] == 'zh'
    assert json.loads(capsys.readouterr().out) == {'language': 'zh', 'segments': 3, 'refilled': MISSED}
    whole_options, clip_options = model.calls[0][1], dict(model.calls[1][1])
    assert clip_options.pop('clip_timestamps') == [15.0, 25.0]
    assert clip_options == whole_options                       # same prompt, same no-conditioning, same word times
    assert whole_options['initial_prompt'] == 'prompt' and whole_options['carry_initial_prompt'] is True
    assert whole_options['condition_on_previous_text'] is False and whole_options['word_timestamps'] is True


def test_without_a_score_folder_or_vocal_melody_it_is_the_whole_song_pass_only(monkeypatch, tmp_path):
    model = whole_song_model()
    result = run_step(monkeypatch, tmp_path, model)
    assert [s['text'] for s in result['segments']] == ['first half', 'last half'] and result['refilled'] == []
    assert len(model.calls) == 1
    empty = tmp_path / 'empty'
    empty.mkdir()
    model = whole_song_model()
    result = run_step(monkeypatch, tmp_path, model, str(empty))
    assert result['refilled'] == [] and len(model.calls) == 1


def test_the_job_gives_the_asr_step_the_score_folder(monkeypatch, tmp_path):
    class Stop(Exception):
        pass

    steps = {}

    def fake_step(name, command, work, timeout, log):
        steps[name] = [str(part) for part in command]
        if name == 'asr':
            raise Stop

    monkeypatch.setattr(job, 'ffmpeg', lambda *args, **kwargs: None)
    monkeypatch.setattr(job, 'step', fake_step)
    work = tmp_path / 'work'
    args = types.SimpleNamespace(source='song.mp3', output=str(tmp_path / 'cover.wav'), work=str(work),
                                 lyrics_file=None, arrangement=None, asr_prompt='prompt')
    with pytest.raises(Stop):
        job.run(args)
    assert steps['asr'][-2:] == ['prompt', str(work / 'score')]
    assert steps['score'][-1] == str(work / 'score')           # the folder score_step.py writes melody_vocal.lab to
