"""The published-lyrics lookup (lyrics_step.py) and the correction it feeds (plan.py). No network, no GPU: the HTTP
helpers are replaced by fakes, and any real request fails the test. Every lyric line here is invented for the tests.

    python -m pytest -q
"""
import io
import json
import os
import subprocess
import sys
import time
import types
import urllib.error
import urllib.parse
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'yue2_cover')]
import job  # noqa: E402
import lyrics_step  # noqa: E402
import paths  # noqa: E402
import plan  # noqa: E402

# An invented "published" song: ten lines, the first sung again as the seventh (a chorus).
PUBLISHED = ['青石桥边落满了灯笼花', '远处的钟声慢慢地散开', '风把旧信吹到了河对岸', '我数着星星等一场大雾',
             '木窗外的猫还没有睡醒', '谁在街角留下半杯清茶', '青石桥边落满了灯笼花', '纸船漂过了第七个路口',
             '月亮借我一盏小小的灯', '明天的雨请你慢慢地下']
# Lines that are in no song.
UNRELATED = ['今晚的饭菜特别丰盛啊', '数学老师在黑板上画圆圈', '公交车站旁边有只小狗', '雨后的操场积了一层水']


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError('the tests never use the network')

    monkeypatch.setattr(lyrics_step.urllib.request, 'urlopen', refuse)


def flat(groups):
    return [line for _, new in groups for line in new]


def lrc_of(lines, start=20.0, step=4.0):
    """Timed lyrics as the service serves them: credits first, then one stamped line each."""
    head = '[00:00.00] 作词 : 某人\n[00:00.50]作曲 : 某人\n'
    body = ''.join(f'[{int((start + i * step) // 60):02d}:{(start + i * step) % 60:05.2f}]{line}\n'
                   for i, line in enumerate(lines))
    return head + body


# --- plan.corrected: misheard, split, merged and unrelated lines ----------------------------------------------

def test_misheard_lines_take_the_published_words():
    heard = ['青石桥边落满了灯龙花', '远处的钟声慢慢散开', '我数着星星等一场大雾']   # a wrong character; one syllable short; right
    groups, changed = plan.corrected(heard, PUBLISHED)
    assert flat(groups) == [PUBLISHED[0], PUBLISHED[1], PUBLISHED[3]]
    assert [g[0] for g in groups] == [[0], [1], [2]] and changed == 2    # the right line is not a change


def test_a_published_line_heard_as_two_halves_is_one_line_again():
    heard = ['远处的钟声慢慢散开', '风把旧信吹到', '了和对岸']
    groups, changed = plan.corrected(heard, PUBLISHED)
    assert groups[1] == ([1, 2], [PUBLISHED[2]]) and changed == 2


def test_two_published_lines_heard_as_one_become_two_lines():
    groups, changed = plan.corrected(['木窗外的猫还没睡醒谁在街角留下半杯清茶'], PUBLISHED)
    assert groups == [([0], [PUBLISHED[4], PUBLISHED[5]])] and changed == 1


def test_a_line_that_is_in_no_song_stays_as_heard():
    heard = ['青石桥边落满了灯笼花', UNRELATED[0], UNRELATED[1], '我数着星星等一场大雾']
    groups, _ = plan.corrected(heard, PUBLISHED)
    assert flat(groups) == [PUBLISHED[0], UNRELATED[0], UNRELATED[1], PUBLISHED[3]]
    assert [g[0] for g in groups] == [[0], [1], [2], [3]]


def test_a_line_half_right_is_close_enough_a_line_less_right_is_not():
    half = '青石桥边落下几片冷雨'                   # 5 of 10 characters: CLOSE, whatever came before
    assert plan.similarity(half, PUBLISHED[0]) == pytest.approx(0.5)
    assert flat(plan.corrected([half], PUBLISHED)[0]) == [PUBLISHED[0]]
    less = '青石桥下流过几片冷雨'                   # 3 of 10: below CLOSE and NEXT
    assert plan.similarity(less, PUBLISHED[0]) < plan.NEXT
    assert flat(plan.corrected([PUBLISHED[9], less], PUBLISHED)[0]) == [PUBLISHED[9], less]


def test_a_weak_match_counts_only_right_after_the_previous_match():
    weak = '风吧旧心推倒了何岸'                     # 4 of 9 characters: below CLOSE, above NEXT
    assert plan.NEXT <= plan.similarity(weak, PUBLISHED[2]) < plan.CLOSE
    # Right after the line before it in the song: it takes the published words.
    assert flat(plan.corrected([PUBLISHED[1], weak], PUBLISHED)[0]) == [PUBLISHED[1], PUBLISHED[2]]
    # Alone, or after a line that is in no song: it stays as heard.
    assert flat(plan.corrected([weak], PUBLISHED)[0]) == [weak]
    assert flat(plan.corrected([UNRELATED[2], weak], PUBLISHED)[0]) == [UNRELATED[2], weak]


def test_a_much_longer_published_line_is_never_forced_into_a_short_heard_one():
    short = '风把旧信吹'                           # half a line: 10 / 5 syllables is beyond LENGTHS
    assert flat(plan.corrected([short], PUBLISHED)[0]) == [short]


def test_the_lines_of_an_english_song_are_corrected_too():
    published = ['paper boats drift down the river', 'past the sleeping lantern town', 'wait by the old clock tower']
    merged, _ = plan.corrected(['paper boats drift down the rivers past the sleeping lantern town'], published)
    assert merged == [([0], published[:2])]
    halves, _ = plan.corrected(['paper boats drift', 'down the rivers'], published)
    assert halves == [([0, 1], [published[0]])]
    # The words of two published lines are counted as the words they are (11 against 15 heard), not run together.
    ad_libs, _ = plan.corrected(['paper boats drift down the river past the sleeping lantern town oh oh oh oh'], published)
    assert ad_libs == [([0], published[:2])]


# --- plan.retimed: place first, correct second ----------------------------------------------------------------

def placed(heard, starts):
    """The syllables of the heard lines with times 0.3 s apart from each line's start, all reliable."""
    lyric = [(u, li) for li, line in enumerate(heard) for u in plan.UNIT.findall(line)]
    times = [starts[li] + 0.3 * k for li, line in enumerate(heard) for k, _ in enumerate(plan.UNIT.findall(line))]
    return lyric, times, [True] * len(lyric)


def test_retimed_keeps_the_times_of_a_line_with_the_same_syllable_count():
    heard = ['远处的钟声慢慢地撒开']                 # one wrong character, same length
    lyric, times, reliable = placed(heard, [10])
    lines, new_lyric, new_times, new_reliable = plan.retimed(plan.corrected(heard, PUBLISHED)[0], lyric, times, reliable)
    assert lines == [PUBLISHED[1]] and new_times == times and new_reliable == reliable
    assert [u for u, _ in new_lyric] == plan.UNIT.findall(PUBLISHED[1])


def test_retimed_spreads_a_line_with_another_syllable_count_over_its_span():
    heard = ['远处的钟声慢慢地撒开', '青石桥边落满灯笼花']        # the second is one syllable short
    lyric, times, reliable = placed(heard, [10, 50])
    lines, new_lyric, new_times, new_reliable = plan.retimed(plan.corrected(heard, PUBLISHED)[0], lyric, times, reliable)
    assert lines == [PUBLISHED[1], PUBLISHED[0]] and len(new_lyric) == 20
    assert new_times[:10] == times[:10] and all(new_reliable[:10])           # same syllables: same times
    assert new_times[10] == pytest.approx(50) and new_times[-1] == pytest.approx(times[-1])
    assert new_times[10:] == sorted(new_times[10:]) and not any(new_reliable[10:])   # spread over its span, evenly
    assert new_lyric[-1] == ('花', 1)


def test_repeated_choruses_stay_where_they_were_sung():
    # Placing first and correcting second: both choruses take the same published line, each in its own time span.
    heard = ['青石桥边落满了灯龙花', '我数着星星等一场大雾', '青石桥边落满灯笼花']
    lyric, times, reliable = placed(heard, [10, 30, 50])
    lines, new_lyric, new_times, _ = plan.retimed(plan.corrected(heard, PUBLISHED)[0], lyric, times, reliable)
    assert lines == [PUBLISHED[0], PUBLISHED[3], PUBLISHED[0]]
    first = {li: next(t for (_, l), t in zip(new_lyric, new_times) if l == li) for li in range(3)}
    assert first == {0: pytest.approx(10), 1: pytest.approx(30), 2: pytest.approx(50)}


def test_a_group_of_two_halves_keeps_every_syllable_time_when_the_count_matches():
    heard = ['风把旧信吹到', '了和对岸']
    lyric, times, reliable = placed(heard, [10, 14])
    lines, _, new_times, new_reliable = plan.retimed(plan.corrected(heard, PUBLISHED)[0], lyric, times, reliable)
    assert lines == [PUBLISHED[2]] and new_times == times and all(new_reliable)


# --- plan.load_lookup -----------------------------------------------------------------------------------------

def write_lookup(work, value):
    (work / 'lyrics-lookup.json').write_text(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False),
                                             encoding='utf-8', newline='\n')


def test_load_lookup_gives_the_published_lines_and_the_song(tmp_path):
    write_lookup(tmp_path, {'song': {'id': 7, 'name': 'x', 'artist': 'y'}, 'score': 0.8,
                            'lines': [{'t': 1.0, 'text': PUBLISHED[0]}, {'t': 5.0, 'text': PUBLISHED[1]}]})
    assert plan.load_lookup(tmp_path) == (PUBLISHED[:2], {'id': 7, 'score': 0.8})


@pytest.mark.parametrize('value', [None, '{broken', '[]', {'song': None, 'why': 'no close lyrics'},
                                   {'song': {'id': 7}, 'lines': []}, {'song': {'id': 7}}, {'song': 'x', 'lines': 3}])
def test_load_lookup_leaves_the_heard_lyrics_when_there_is_nothing_usable(tmp_path, value):
    if value is not None:
        write_lookup(tmp_path, value)
    assert plan.load_lookup(tmp_path) == (None, None)


# --- lyrics_step: parsing and matching ------------------------------------------------------------------------

def test_timed_lyrics_drop_credits_and_expand_repeated_stamps():
    lrc = ('[00:00.00] 作词 : 某人\n[00:00.20]演唱: 某人\n[00:00.30]【invented notice】\n[00:00.35](instrumental)\n'
           f'[00:00.39]{PUBLISHED[0]}\n[01:04.14][00:48.36]{PUBLISHED[1]}\n[01:30.00]\n[01:31.00]...\n')
    parsed = lyrics_step.parse_lrc(lrc)
    assert [line for _, line in parsed] == [PUBLISHED[0], PUBLISHED[1], PUBLISHED[1]]
    assert [t for t, _ in parsed] == pytest.approx([0.39, 48.36, 64.14])


def test_heard_lines_skip_credits_and_fragments_and_split_at_punctuation():
    asr = {'segments': [{'text': '作词 某人 编曲 某人'}, {'text': f'{PUBLISHED[0]}，{PUBLISHED[1]}。 ok'}, {'text': '好'},
                        {'text': f'{PUBLISHED[2]} {PUBLISHED[3]}'}]}
    assert lyrics_step.heard_lines(asr) == PUBLISHED[:4]


def test_overlap_is_the_share_of_heard_lines_with_a_close_published_line():
    heard = ['青石桥边落满了灯龙花', '我数着星星等一场大雾', UNRELATED[0]]   # a misheard line, a right one, a stranger
    assert lyrics_step.overlap(heard, PUBLISHED) == pytest.approx(2 / 3)
    assert lyrics_step.overlap(UNRELATED, PUBLISHED) == 0.0
    assert lyrics_step.overlap([], PUBLISHED) == 0.0 and lyrics_step.overlap(heard, []) == 0.0


def test_overlap_is_judged_line_by_line_so_repeated_choruses_do_not_matter():
    heard = [PUBLISHED[0], PUBLISHED[3], PUBLISHED[0], PUBLISHED[0], PUBLISHED[9], PUBLISHED[0]]
    assert lyrics_step.overlap(heard, PUBLISHED) == 1.0


# --- lyrics_step.lookup with the network faked ----------------------------------------------------------------

HEARD = ['青石桥边落满了灯龙花', '风把旧信吹到了河对岸', '我数着星星等一场大雾', '远处的钟声慢慢散开']


def asr_of(*lines):
    return {'language': 'zh', 'segments': [{'text': '，'.join(lines[i:i + 2])} for i in range(0, len(lines), 2)]}


class Service:
    """A fake of the two endpoints: a search answer (song ids) and a lyric text per song id."""

    def __init__(self, songs, lyrics, monkeypatch):
        self.songs, self.lyrics, self.posts, self.gets = songs, lyrics, [], []
        monkeypatch.setattr(lyrics_step, 'post', self.post)
        monkeypatch.setattr(lyrics_step, 'get', self.get)

    def post(self, url, data, deadline):
        self.posts.append((url, data))
        return {'result': {'songs': [{'id': i, 'name': f'song {i}', 'ar': [{'name': 'a'}, {'name': 'b'}]}
                                     for i in self.songs]}}

    def get(self, url, deadline):
        self.gets.append(url)
        text = self.lyrics[int(urllib.parse.parse_qs(urllib.parse.urlparse(url).query)['id'][0])]
        return {'lrc': {'lyric': text} if text is not None else None}


def test_lookup_keeps_the_candidate_most_heard_lines_match(monkeypatch):
    service = Service([11, 22], {11: lrc_of(UNRELATED), 22: lrc_of(PUBLISHED)}, monkeypatch)
    found = lyrics_step.lookup(asr_of(*HEARD))
    assert found['song'] == {'id': 22, 'name': 'song 22', 'artist': 'a,b'} and found['score'] == 1.0
    assert [line['text'] for line in found['lines']] == PUBLISHED          # credits dropped, in time order
    assert found['lines'][0] == {'t': 20.0, 'text': PUBLISHED[0]} and found['lines'][1]['t'] == 24.0
    assert all(url == lyrics_step.SEARCH and data['type'] == 1006 for url, data in service.posts)
    assert [g.split('?')[0] for g in service.gets] == ['https://music.163.com/api/song/lyric'] * 2


def test_lookup_searches_the_three_longest_distinct_heard_lines(monkeypatch):
    service = Service([22], {22: lrc_of(PUBLISHED)}, monkeypatch)
    lines = ['风把旧信吹到了', PUBLISHED[0], PUBLISHED[0], '月亮借我一盏小小的灯光', PUBLISHED[3], '木窗外的猫还没有睡醒啊', '纸船漂过了']
    lyrics_step.lookup(asr_of(*lines))
    assert [data['s'] for _, data in service.posts] == ['月亮借我一盏小小的灯光', '木窗外的猫还没有睡醒啊', PUBLISHED[0]]
    service = Service([22], {22: lrc_of(PUBLISHED)}, monkeypatch)
    lyrics_step.lookup(asr_of(PUBLISHED[0], PUBLISHED[0], PUBLISHED[0], PUBLISHED[3]))
    assert [data['s'] for _, data in service.posts] == [PUBLISHED[0], PUBLISHED[3]]


def test_lookup_rejects_a_song_that_fewer_than_half_the_heard_lines_match(monkeypatch):
    heard = [PUBLISHED[0], UNRELATED[0], UNRELATED[1], UNRELATED[2]]
    Service([22], {22: lrc_of(PUBLISHED)}, monkeypatch)
    found = lyrics_step.lookup(asr_of(*heard))
    assert found == {'song': None, 'why': 'no close lyrics', 'best_score': 0.25, 'candidates': 1}
    # Exactly half is enough.
    found = lyrics_step.lookup(asr_of(PUBLISHED[0], PUBLISHED[3], UNRELATED[0], UNRELATED[1]))
    assert found['song']['id'] == 22 and found['score'] == 0.5


def test_lookup_rejects_candidates_that_are_other_songs(monkeypatch):
    Service([11, 12], {11: lrc_of(UNRELATED), 12: lrc_of(UNRELATED[::-1])}, monkeypatch)
    found = lyrics_step.lookup(asr_of(*HEARD))
    assert found['song'] is None and found['why'] == 'no close lyrics' and found['candidates'] == 2
    assert found['best_score'] < lyrics_step.ACCEPT and 'lines' not in found


def test_lookup_skips_candidates_without_timed_lyrics_and_reports_none_found(monkeypatch):
    Service([11, 12], {11: None, 12: '[ti:only a title]\nno stamps here'}, monkeypatch)
    assert lyrics_step.lookup(asr_of(*HEARD)) == {'song': None, 'why': 'no close lyrics', 'best_score': None,
                                                  'candidates': 2}
    Service([], {}, monkeypatch)
    assert lyrics_step.lookup(asr_of(*HEARD))['candidates'] == 0


def test_lookup_does_not_search_when_nothing_was_heard(monkeypatch):
    service = Service([22], {22: lrc_of(PUBLISHED)}, monkeypatch)
    assert lyrics_step.lookup({'segments': [{'text': '作词 某人'}, {'text': '好'}]}) == {'song': None, 'why': 'nothing heard'}
    assert service.posts == [] and service.gets == []


def test_lookup_stops_when_its_time_budget_is_spent(monkeypatch):
    service = Service([22], {22: lrc_of(PUBLISHED)}, monkeypatch)
    monkeypatch.setattr(lyrics_step, 'BUDGET', -1)
    assert lyrics_step.lookup(asr_of(*HEARD))['song'] is None
    assert service.posts == [] and service.gets == []


# --- lyrics_step.main: a failure only leaves the lookup out ---------------------------------------------------

def run_main(monkeypatch, tmp_path, asr):
    source, out = tmp_path / 'asr.json', tmp_path / 'lyrics-lookup.json'
    if asr is not None:
        source.write_text(json.dumps(asr, ensure_ascii=False), encoding='utf-8', newline='\n')
    monkeypatch.setattr(sys, 'argv', ['lyrics_step.py', str(source), str(out)])
    lyrics_step.main()
    return json.loads(out.read_text(encoding='utf-8'))


def test_main_writes_the_found_song_and_prints_it_without_the_lines(monkeypatch, tmp_path, capsys):
    Service([22], {22: lrc_of(PUBLISHED)}, monkeypatch)
    written = run_main(monkeypatch, tmp_path, asr_of(*HEARD))
    assert written['song']['id'] == 22 and len(written['lines']) == 10
    printed = json.loads(capsys.readouterr().out)
    assert printed['song']['id'] == 22 and 'lines' not in printed


@pytest.mark.parametrize('error', [urllib.error.URLError('offline'), TimeoutError('slow'), ValueError('not json')])
def test_main_writes_no_song_when_the_service_fails(monkeypatch, tmp_path, capsys, error):
    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(lyrics_step, 'post', fail)
    assert run_main(monkeypatch, tmp_path, asr_of(*HEARD)) == {'song': None, 'why': type(error).__name__}
    assert json.loads(capsys.readouterr().out)['song'] is None


def test_main_writes_no_song_when_the_asr_file_is_missing(monkeypatch, tmp_path):
    assert run_main(monkeypatch, tmp_path, None) == {'song': None, 'why': 'FileNotFoundError'}


# --- the two HTTP helpers -------------------------------------------------------------------------------------

def fake_urlopen(monkeypatch, answer):
    seen = {}

    def urlopen(request, timeout):
        seen['request'], seen['timeout'] = request, timeout
        return io.BytesIO(json.dumps(answer).encode())

    monkeypatch.setattr(lyrics_step.urllib.request, 'urlopen', urlopen)
    return seen


def test_post_sends_a_form_to_the_search_endpoint(monkeypatch):
    seen = fake_urlopen(monkeypatch, {'result': {}})
    data = {'s': PUBLISHED[0], 'type': 1006, 'limit': 5, 'offset': 0}
    assert lyrics_step.post(lyrics_step.SEARCH, data, time.monotonic() + 100) == {'result': {}}
    request = seen['request']
    assert request.get_method() == 'POST' and request.full_url == 'https://music.163.com/api/cloudsearch/pc'
    assert urllib.parse.parse_qs(request.data.decode()) == {'s': [PUBLISHED[0]], 'type': ['1006'], 'limit': ['5'],
                                                           'offset': ['0']}
    assert request.get_header('Referer') == 'https://music.163.com/'
    assert request.get_header('Content-type') == 'application/x-www-form-urlencoded'
    assert seen['timeout'] == lyrics_step.TIMEOUT


def test_get_fetches_the_timed_lyrics_of_one_song(monkeypatch):
    seen = fake_urlopen(monkeypatch, {'lrc': {'lyric': 'x'}})
    assert lyrics_step.get(lyrics_step.LYRIC.format(22), time.monotonic() + 100) == {'lrc': {'lyric': 'x'}}
    request = seen['request']
    assert request.get_method() == 'GET' and request.full_url == 'https://music.163.com/api/song/lyric?id=22&lv=1'


def test_requests_never_wait_longer_than_the_time_left(monkeypatch):
    seen = fake_urlopen(monkeypatch, {})
    lyrics_step.get(lyrics_step.LYRIC.format(1), time.monotonic() + 2.5)
    assert 1.0 <= seen['timeout'] <= 2.5
    lyrics_step.get(lyrics_step.LYRIC.format(1), time.monotonic() - 5)
    assert seen['timeout'] == 1.0


# --- the switch: environment variable and flag ----------------------------------------------------------------

@pytest.mark.parametrize('value, expected', [(None, True), ('', True), ('1', True), ('yes', True), ('on', True),
                                             ('0', False), ('false', False), ('FALSE', False), (' no ', False),
                                             ('off', False)])
def test_the_environment_variable_turns_the_lookup_off_only_when_asked(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv('YUE2_COVER_LYRICS_LOOKUP', raising=False)
    else:
        monkeypatch.setenv('YUE2_COVER_LYRICS_LOOKUP', value)
    assert paths.flag('YUE2_COVER_LYRICS_LOOKUP') is expected


@pytest.mark.parametrize('value, expected', [(None, 'True'), ('1', 'True'), ('0', 'False'), ('off', 'False')])
def test_the_lookup_is_on_by_default_and_the_variable_is_read_at_import(value, expected):
    env = {k: v for k, v in os.environ.items() if k != 'YUE2_COVER_LYRICS_LOOKUP'}
    if value is not None:
        env['YUE2_COVER_LYRICS_LOOKUP'] = value
    done = subprocess.run([sys.executable, '-c', 'import paths; print(paths.LYRICS_LOOKUP)'], cwd=ROOT / 'yue2_cover',
                          env=env, capture_output=True, text=True, timeout=60, check=True)
    assert done.stdout.strip() == expected


class Stop(Exception):
    pass


def run_job(monkeypatch, tmp_path, *, lyrics=None, flag=False, enabled=True, step_error=None, stale=False):
    """Run job.run up to plan.build; return the names of the steps that ran and the work folder."""
    work, ran = tmp_path / 'work', []
    work.mkdir(exist_ok=True)
    if stale:
        write_lookup(work, {'song': {'id': 1}, 'lines': [{'t': 1, 'text': PUBLISHED[0]}]})

    def fake_step(name, command, work_dir, timeout, log):
        ran.append((name, [str(part) for part in command], timeout))
        if name == 'lyrics':
            if step_error:
                raise step_error
            write_lookup(work, {'song': None})

    def fake_build(work_dir, lyrics_text, arrangement):
        raise Stop

    monkeypatch.setattr(job, 'ffmpeg', lambda *args, **kwargs: None)
    monkeypatch.setattr(job, 'step', fake_step)
    monkeypatch.setattr(plan, 'build', fake_build)
    monkeypatch.setattr(paths, 'LYRICS_LOOKUP', enabled)
    lyrics_file = None
    if lyrics is not None:
        lyrics_file = tmp_path / 'lyrics.txt'
        lyrics_file.write_text(lyrics, encoding='utf-8', newline='\n')
    args = types.SimpleNamespace(source='song.mp3', output=str(tmp_path / 'cover.wav'), work=str(work),
                                 lyrics_file=str(lyrics_file) if lyrics_file else None, arrangement=None,
                                 asr_prompt='prompt', no_lyrics_lookup=flag)
    with pytest.raises(Stop):
        job.run(args)
    return ran, work


def test_the_job_looks_the_lyrics_up_when_none_are_given(monkeypatch, tmp_path):
    ran, work = run_job(monkeypatch, tmp_path)
    assert [name for name, _, _ in ran] == ['score', 'asr', 'lyrics']
    name, command, timeout = ran[-1]
    assert command[1:] == [str(Path(job.__file__).resolve().parent / 'lyrics_step.py'), str(work / 'asr.json'),
                           str(work / 'lyrics-lookup.json')] and timeout == 90


def test_a_blank_lyrics_file_is_no_lyrics(monkeypatch, tmp_path):
    ran, _ = run_job(monkeypatch, tmp_path, lyrics='  \n')
    assert [name for name, _, _ in ran] == ['score', 'asr', 'lyrics']


def test_the_flag_skips_the_lookup(monkeypatch, tmp_path):
    assert [n for n, _, _ in run_job(monkeypatch, tmp_path, flag=True)[0]] == ['score', 'asr']


def test_the_environment_switch_skips_the_lookup(monkeypatch, tmp_path):
    assert [n for n, _, _ in run_job(monkeypatch, tmp_path, enabled=False)[0]] == ['score', 'asr']


def test_given_lyrics_skip_the_lookup(monkeypatch, tmp_path):
    ran, _ = run_job(monkeypatch, tmp_path, lyrics=PUBLISHED[0] + '\n')
    assert [n for n, _, _ in ran] == ['score', 'asr']


@pytest.mark.parametrize('error', [RuntimeError('YUE2_LYRICS_FAILED'), subprocess.TimeoutExpired('lyrics_step.py', 90)])
def test_a_failed_lookup_only_leaves_the_lookup_out(monkeypatch, tmp_path, error):
    ran, work = run_job(monkeypatch, tmp_path, step_error=error, stale=True)    # the job still reached plan.build
    assert [n for n, _, _ in ran] == ['score', 'asr', 'lyrics'] and not (work / 'lyrics-lookup.json').exists()


def test_an_earlier_attempts_lookup_is_never_reused(monkeypatch, tmp_path):
    _, work = run_job(monkeypatch, tmp_path, flag=True, stale=True)
    assert not (work / 'lyrics-lookup.json').exists()


def test_the_command_line_flag(monkeypatch):
    seen = []
    monkeypatch.setattr(job, 'run', seen.append)
    for extra, expected in (([], False), (['--no-lyrics-lookup'], True)):
        monkeypatch.setattr(sys, 'argv', ['job.py', '--source', 's.mp3', '--output', 'o.wav', '--work', 'w', *extra])
        job.main()
        assert seen[-1].no_lyrics_lookup is expected


# --- plan.build: the lookup file, placing first, and the receipt field ---------------------------------------

class FakeScore:
    """A score with a verse (0-40 s) and a chorus (from 40 s); only what build() asks of the real one."""

    def __init__(self, score_dir):
        self.text = '% verse\n% chorus\n'
        self.score = types.SimpleNamespace(bpm=100)
        self.vocal = types.SimpleNamespace(notes=[(0, 60, 4), (100, 60, 4)])

    def seconds(self, q):
        return float(q)

    def sections(self):
        return [('verse', 0.0, 40.0), ('chorus', 40.0, float('inf'))]


def heard_segment(text, start):
    return {'start': start, 'end': start + 0.3 * len(text), 'text': text,
            'words': [{'w': ch, 's': start + 0.3 * k, 'e': start + 0.3 * k + 0.25} for k, ch in enumerate(text)]}


def build_with(monkeypatch, tmp_path, lookup=None, lyrics=''):
    monkeypatch.setattr(plan, 'Score', FakeScore)
    monkeypatch.setattr(plan, 'load_references', lambda: [{'clip': 'ref-1', 'lyrics': ['reference line'], 'true_bpm': 100}])
    monkeypatch.setattr(plan, 'choose_reference', lambda bpm, refs: (refs[0], 1.0))
    monkeypatch.setattr(plan, 'reference_plan', lambda ref, text: (
        '% verse\n% verse\n% chorus\n', {'rate': 1.0, 'lead_seconds': 0.0, 'samples': 48000, 'frames': 25,
                                          'boundary_quarters': '0', 'composite_seconds': 100.0}))
    monkeypatch.setattr(plan, 'held_tails', lambda *args: [])
    # The chorus is heard twice, wrongly each time: in the verse (10 s) and in the chorus (50 s).
    asr = {'language': 'zh', 'segments': [heard_segment('青石桥边落满了灯龙花', 10), heard_segment(PUBLISHED[3], 20),
                                          heard_segment('青石桥边落满灯笼花', 50)]}
    (tmp_path / 'asr.json').write_text(json.dumps(asr, ensure_ascii=False), encoding='utf-8', newline='\n')
    if lookup is not None:
        write_lookup(tmp_path, lookup)
    result = plan.build(tmp_path, lyrics, '')
    return result, json.loads((tmp_path / 'request.json').read_text(encoding='utf-8'))['lyrics']


FOUND = {'song': {'id': 22, 'name': 'x', 'artist': 'y'}, 'score': 0.9,
         'lines': [{'t': 20 + 4 * i, 'text': line} for i, line in enumerate(PUBLISHED)]}


def test_build_corrects_the_heard_lines_in_their_own_sections(monkeypatch, tmp_path):
    result, text = build_with(monkeypatch, tmp_path, FOUND)
    assert result['lyric_source'] == 'asr+lookup:2' and result['lines'] == 3
    assert result['lyrics_song'] == {'id': 22, 'score': 0.9}                    # it reaches receipt.json via job.py
    # The chorus line is in the verse where it was first sung and again in the chorus where it was sung again.
    assert text == (f'[Verse]\nreference line\n\n[Verse]\n{PUBLISHED[0]}\n{PUBLISHED[3]}\n\n[Chorus]\n{PUBLISHED[0]}\n')


def test_build_without_a_found_song_sings_what_was_heard(monkeypatch, tmp_path):
    for n, lookup in enumerate((None, {'song': None, 'why': 'no close lyrics'}, '{broken')):
        work = tmp_path / f'work{n}'
        work.mkdir()
        result, text = build_with(monkeypatch, work, lookup)
        assert result['lyric_source'] == 'asr' and 'lyrics_song' not in result
        assert '青石桥边落满了灯龙花' in text and '青石桥边落满灯笼花' in text


def test_build_never_changes_the_users_own_lyrics(monkeypatch, tmp_path):
    result, text = build_with(monkeypatch, tmp_path, FOUND, lyrics=f'{PUBLISHED[0]}\n{PUBLISHED[3]}\n{PUBLISHED[0]}\n')
    assert result['lyric_source'] == 'user' and 'lyrics_song' not in result
