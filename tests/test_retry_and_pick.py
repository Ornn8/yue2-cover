"""The retry loop (job.sing), the leading dead-air trim, the reference pick (plan.pick_reference, tempo_view) and the
song features. No GPU, no network; every name and number here is made up for the tests.

    python -m pytest -q
"""
import json
import sys
from fractions import Fraction
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'yue2_cover')]
import job  # noqa: E402
import plan  # noqa: E402


# --- job.sing: up to three attempts, fresh seeds ----------------------------------------------------------------

def a_plan(reference='clip-a', rate=1.0):
    return {'reference': reference, 'reference_rate': rate}


class Run:
    """generate/finish/prepare/alternative fakes; `results` is what each generate call returns, in order."""

    def __init__(self, results, lengths=None, other=None):
        self.results, self.lengths = list(results), list(lengths or [])
        self.calls, self.prepared, self.other = [], [], other

    def generate(self, plan, prefix, seeds, attempt):
        self.calls.append((plan['reference'], list(seeds), attempt))
        sung, early = self.results.pop(0)
        stops = [{'seed': s, 'new_frames': 50, 'seconds': 1.0} for s in early]
        return ({'seed': sung} if sung is not None else None), stops

    def finish(self, complete, plan):
        return self.lengths.pop(0), 0.0

    def prepare(self, plan):
        self.prepared.append(plan['reference'])
        return Path(f"{plan['reference']}.npy")

    def alternative(self, plan):
        return self.other

    def sing(self, plan, seeds=(11, 22, 33), shortest=100):
        return job.sing(plan, list(seeds), shortest, {plan['reference']: Path('a.npy')}, prepare=self.prepare,
                        generate=self.generate, finish=self.finish, alternative=self.alternative)


def test_the_first_long_enough_take_is_delivered_with_one_receipt_row():
    run = Run([(11, [])], lengths=[150.0])
    complete, chosen, seconds, trimmed, attempts = run.sing(a_plan())
    assert complete['seed'] == 11 and seconds == 150.0
    assert attempts == [{'attempt': 1, 'seed': 11, 'reference': 'clip-a', 'audio_seconds': 150.0, 'early_stop': False}]
    assert run.calls == [('clip-a', [11, 22, 33], 1)]       # all three seeds are handed over: one model load


def test_early_stops_before_a_take_are_listed_and_count_as_attempts():
    run = Run([(33, [11, 22])], lengths=[150.0])
    _, _, _, _, attempts = run.sing(a_plan())
    assert [(a['attempt'], a['seed'], a['early_stop']) for a in attempts] == [(1, 11, True), (2, 22, True), (3, 33, False)]


def test_a_take_that_is_too_short_is_marked_and_the_next_seed_is_sung():
    run = Run([(11, []), (22, [])], lengths=[3.0, 150.0])
    complete, _, _, _, attempts = run.sing(a_plan())
    assert complete['seed'] == 22
    assert attempts[0]['too_short'] is True and 'too_short' not in attempts[1]
    assert [call[2] for call in run.calls] == [1, 2]


def test_when_every_attempt_fails_the_job_gets_all_three_rows():
    run = Run([(None, [11, 22, 33])])
    with pytest.raises(job.Unsung) as error:
        run.sing(a_plan())
    assert [a['seed'] for a in error.value.attempts] == [11, 22, 33] and str(error.value) == 'YUE2_EARLY_STOP'


def test_a_stretched_reference_holds_the_last_seed_back_for_the_next_closest_clip():
    run = Run([(None, [11, 22]), (33, [])], lengths=[150.0], other=a_plan('clip-b', 1.01))
    complete, chosen, _, _, attempts = run.sing(a_plan(rate=1.10))
    assert run.calls == [('clip-a', [11, 22], 1), ('clip-b', [33], 3)]       # the third seed rides on the other clip
    assert run.prepared == ['clip-b'] and chosen['reference'] == 'clip-b'
    assert [a['reference'] for a in attempts] == ['clip-a', 'clip-a', 'clip-b']


def test_without_a_closer_clip_the_last_attempt_keeps_the_same_reference():
    run = Run([(None, [11, 22]), (33, [])], lengths=[150.0], other=None)
    _, chosen, _, _, _ = run.sing(a_plan(rate=1.10))
    assert chosen['reference'] == 'clip-a' and run.prepared == []


def test_a_reference_within_three_percent_is_not_stretched():
    assert not job.stretched(a_plan(rate=1.029)) and job.stretched(a_plan(rate=1.031)) and job.stretched(a_plan(rate=0.96))


def test_fresh_seeds_are_valid_and_differ():
    seeds = job.fresh_seeds()
    assert len(seeds) == job.ATTEMPTS and all(0 < s < 2**31 for s in seeds)
    assert job.fresh_seeds() != seeds


# --- the dead air after the voice prompt ------------------------------------------------------------------------

def test_the_cover_starts_after_the_last_dead_air_stretch_in_its_first_seconds():
    np = pytest.importorskip('numpy')
    rate = 8000
    loud = lambda seconds: 0.3 * np.sin(np.linspace(0, 440 * 6.28 * seconds, int(rate * seconds)))
    audio = np.concatenate([loud(1.0), np.zeros(int(rate * 0.6)), loud(5.0)]).astype('float32')[:, None]
    begin = job.seam_start(audio, rate)
    assert begin is not None and abs(begin / rate - 1.55) < 0.06           # one frame (50 ms) before the sound returns
    assert job.seam_start(loud(5.0).astype('float32')[:, None], rate) is None   # no dead air: nothing is cut
    short_gap = np.concatenate([loud(1.0), np.zeros(int(rate * 0.2)), loud(5.0)]).astype('float32')[:, None]
    assert job.seam_start(short_gap, rate) is None                         # a 0.2 s rest is music, not dead air


# --- tempo views: half-time and double-time readings --------------------------------------------------------------

def clip(name, bpm, drums=None, seconds=30):
    return {'clip': name, 'true_bpm': bpm, 'score_bpm': round(bpm), 'first_note_quarter': '2', 'seconds': seconds}


def test_a_song_inside_every_clips_stretch_range_is_read_as_it_is():
    refs = [clip('a', 70), clip('b', 100)]
    view, reading = plan.tempo_view(80, refs)
    assert reading == 'direct' and view == refs


def test_a_beat_tracker_reading_twice_the_tempo_is_read_at_half_time():
    refs = [clip('a', 70), clip('b', 100)]
    view, reading = plan.tempo_view(158, refs)           # really a ~79 bpm ballad
    assert reading == 'half-time' and {r['clip'] for r in view} == {'a', 'b'}
    assert all(r['tempo_factor'] == 2 for r in view) and view[0]['true_bpm'] == 140
    view, reading = plan.tempo_view(38, refs)            # and the other way round
    assert reading == 'double-time' and view[0]['tempo_factor'] == 0.5


def test_a_tempo_no_reading_fits_is_clamped_to_the_largest_stretch():
    view, reading = plan.tempo_view(300, [clip('a', 70)])
    assert reading == 'clamped' and all(r['clamped'] for r in view)
    ref, rate = plan.choose_reference(300, view)
    assert rate == pytest.approx(plan.MAX_STRETCH)


def test_a_clip_counted_at_half_time_has_its_score_rescaled():
    text = 'X:1\nT:\nM:4/4\nL:1/16\nQ:1/4=60\nV: Vocal\n'
    twice = plan.scale_score(text, 2).splitlines()
    assert twice[3] == 'L:1/8' and twice[4] == 'Q:1/4=120' and twice[2] == 'M:4/2'
    half = plan.scale_score(text, 0.5).splitlines()
    assert half[3] == 'L:1/32' and half[4] == 'Q:1/4=30' and half[2] == 'M:4/8'


def test_a_clip_with_nothing_left_after_exclusions_is_out_of_range():
    with pytest.raises(ValueError, match='YUE2_TEMPO_OUT_OF_RANGE'):
        plan.choose_reference(80, [clip('a', 80)], exclude={'a'})


# --- the reference pick: tempo within 3 %, then the closest drum share ------------------------------------------

def measured(**drums):
    return {name: {'drums': value} for name, value in drums.items()}


def test_with_one_clip_in_the_tempo_band_the_pick_is_by_tempo():
    refs = [clip('a', 80), clip('b', 100)]
    ref, record = plan.pick_reference(81, refs, {'drums': 0.2}, measured(a=0.05, b=0.2))
    assert ref['clip'] == 'a' and record['rule'] == 'tempo' and record['pick'] == 'a'


def test_among_clips_within_three_percent_the_closest_drum_share_wins():
    refs = [clip('a', 80), clip('b', 81), clip('c', 79.5), clip('far', 100)]
    ref, record = plan.pick_reference(80, refs, {'drums': 0.18}, measured(a=0.05, b=0.17, c=0.10, far=0.18))
    assert ref['clip'] == 'b' and record['rule'] == 'tempo+drums'
    assert [row['clip'] for row in record['candidates']] == ['a', 'b', 'c']       # 'far' is outside the band
    assert record['song_drums'] == 0.18


def test_clips_within_the_tie_margin_are_picked_at_random():
    refs = [clip('a', 80), clip('b', 80.5), clip('c', 79.5)]
    seen = []

    def choice(options):
        seen.append([r['clip'] for r in options])
        return options[-1]

    ref, record = plan.pick_reference(80, refs, {'drums': 0.10}, measured(a=0.10, b=0.11, c=0.20), choice=choice)
    assert seen == [['a', 'b']] and ref['clip'] == 'b' and record['rule'] == 'tempo+drums+random'
    assert record['tied'] == ['a', 'b']


def test_a_missing_measure_falls_back_to_the_nearest_tempo():
    refs = [clip('a', 80), clip('b', 81)]
    ref, record = plan.pick_reference(80.2, refs, None, measured(a=0.1, b=0.1))                  # song not measured
    assert record['rule'] == 'tempo' and ref['clip'] == 'a'
    ref, record = plan.pick_reference(80.2, refs, {'drums': 0.1}, measured(a=0.1))               # b not measured
    assert record['rule'] == 'tempo'


def test_an_excluded_clip_is_never_picked():
    refs = [clip('a', 80), clip('b', 81), clip('c', 79.5)]
    ref, record = plan.pick_reference(80, refs, {'drums': 0.1}, measured(a=0.1, b=0.2, c=0.3), exclude={'a'})
    assert ref['clip'] == 'b' and 'a' not in [row['clip'] for row in record['candidates']]


def test_reference_features_are_read_from_the_library_and_missing_means_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(plan.paths, 'REFERENCES', tmp_path)
    assert plan.load_reference_features() == {}
    (tmp_path / 'features.json').write_text(json.dumps({'a': {'drums': 0.1}}), encoding='utf-8')
    assert plan.load_reference_features() == {'a': {'drums': 0.1}}
    (tmp_path / 'features.json').write_text('{broken', encoding='utf-8')
    assert plan.load_reference_features() == {}


def test_song_features_are_none_when_missing_or_broken(tmp_path):
    assert plan.song_features(tmp_path) is None
    (tmp_path / 'features.json').write_text('[1]', encoding='utf-8')
    assert plan.song_features(tmp_path) is None
    (tmp_path / 'features.json').write_text('{"drums": 0.2}', encoding='utf-8')
    assert plan.song_features(tmp_path) == {'drums': 0.2}


# --- features.py ---------------------------------------------------------------------------------------------------

def test_clicks_have_a_higher_drum_share_than_a_sustained_tone(tmp_path):
    np = pytest.importorskip('numpy')
    pytest.importorskip('librosa')
    sf = pytest.importorskip('soundfile')
    import features
    rate, seconds = 22050, 6
    t = np.arange(rate * seconds) / rate
    tone = (0.3 * np.sin(2 * np.pi * 220 * t)).astype('float32')
    clicks = np.zeros_like(tone)
    clicks[:: rate // 4] = 0.9
    clicks = np.convolve(clicks, np.hanning(40), mode='same').astype('float32')
    sf.write(str(tmp_path / 'tone.wav'), tone, rate)
    sf.write(str(tmp_path / 'clicks.wav'), clicks, rate)
    quiet, busy = features.measure(tmp_path / 'tone.wav'), features.measure(tmp_path / 'clicks.wav')
    assert busy['drums'] > quiet['drums'] + 0.2
    assert set(quiet) >= {'seconds', 'drums', 'hf4k_db', 'centroid_hz', 'low150_db', 'onsets_per_s', 'lra'}


def test_a_new_reference_stores_its_measures_beside_the_library(monkeypatch, tmp_path):
    sys.path.insert(0, str(ROOT / 'tools'))
    import add_reference
    import features
    monkeypatch.setattr(features, 'measure', lambda path: {'drums': 0.12})
    add_reference.measure_clip(tmp_path / 'x.flac', 'x', tmp_path)
    add_reference.measure_clip(tmp_path / 'y.flac', 'y', tmp_path)
    assert json.loads((tmp_path / 'features.json').read_text(encoding='utf-8')) == {'x': {'drums': 0.12}, 'y': {'drums': 0.12}}
    monkeypatch.setattr(features, 'measure', lambda path: 1 / 0)          # a failure only leaves the measures out
    add_reference.measure_clip(tmp_path / 'z.flac', 'z', tmp_path)
    assert 'z' not in json.loads((tmp_path / 'features.json').read_text(encoding='utf-8'))
