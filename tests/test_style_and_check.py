"""Style words (style_words.py) and the optional lyric check (lyric_check.py). No network, no GPU, no model: the listener is a
fake function. Every lyric line here is a placeholder made up for the tests.

    python -m pytest -q
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'yue2_cover')]
import lyric_check  # noqa: E402
import style_words  # noqa: E402


# --- cleaning what a model answers -----------------------------------------------------------------------------------

def test_a_tag_line_is_kept_as_it_is():
    assert style_words.clean_style('emotional piano ballad, soft strings, slow tempo, melancholic') == \
        'emotional piano ballad, soft strings, slow tempo, melancholic'


def test_voice_words_are_removed_because_the_singer_is_fixed_elsewhere():
    answer = 'upbeat synth pop, female vocals, driving drums, breathy, mid-tempo, singer-songwriter feel'
    assert style_words.clean_style(answer) == 'upbeat synth pop, driving drums, mid-tempo'


def test_labels_quotes_and_code_fences_around_the_line_are_stripped():
    assert style_words.clean_style('```\nStyle: "warm folk, acoustic guitar, gentle"\n```') == \
        'warm folk, acoustic guitar, gentle'


def test_tags_are_unique_and_the_line_is_cut_at_the_limit():
    many = ', '.join(f'tag number {i} here' for i in range(40))
    result = style_words.clean_style(many + ', Tag Number 1 Here')
    assert len(result) <= style_words.MAX_STYLE_CHARS and len(set(result.split(', '))) == len(result.split(', '))


@pytest.mark.parametrize('answer', [None, 42, '', '   ', 'I cannot help with that.', 'jazz', '抒情, 钢琴, 慢板, 温暖',
                                    'Sorry, I cannot listen to audio.'])
def test_an_answer_that_is_not_a_tag_line_is_refused(answer):
    assert style_words.clean_style(answer) is None


# --- the excerpt ---------------------------------------------------------------------------------------------------------

def test_a_short_song_is_sent_whole_and_a_long_one_around_its_first_chorus():
    assert style_words.excerpt_window(40.0) == (0.0, 40.0)
    assert style_words.excerpt_window(200.0) == (56.0, 40.0)
    assert style_words.excerpt_window(50.0) == (10.0, 40.0)               # never past the end
    assert style_words.excerpt_window(100.0)[0] == 28.0


class FakeMedia:
    """ffprobe/ffmpeg stand-ins: the probe says the song is `seconds` long, ffmpeg writes some bytes."""

    def __init__(self, seconds=200.0, size=5000):
        self.seconds, self.size, self.commands = seconds, size, []

    def __call__(self, command, **kwargs):
        self.commands.append(command)
        if command[0] == 'ffprobe':
            return subprocess.CompletedProcess(command, 0, stdout=f'{self.seconds}\n')
        Path(command[-1]).write_bytes(b'x' * self.size)
        return subprocess.CompletedProcess(command, 0, stdout='')


# --- who decides the words -------------------------------------------------------------------------------------------------

def never(*args):
    raise AssertionError('the listener must not be called')


def test_the_users_own_words_are_used_exactly_and_nothing_is_listened_to(tmp_path):
    words, record = style_words.resolve('  acoustic guitar folk ', tmp_path / 'song.wav', tmp_path, listener=never,
                                        run=FakeMedia())
    assert words == '  acoustic guitar folk ' and record == {'style_source': 'user', 'text': 'acoustic guitar folk'}


def test_without_a_listener_the_default_words_are_used(tmp_path):
    assert style_words.resolve('', tmp_path / 'song.wav', tmp_path, run=FakeMedia()) == ('', {'style_source': 'default'})


def test_a_listener_writes_the_words_from_the_excerpt(tmp_path):
    heard = []

    def listener(audio, mime, prompt):
        heard.append((len(audio), mime, 'style tags' in prompt))
        return 'dream pop, shimmering guitars, mid-tempo, wistful, hazy production'

    media = FakeMedia(seconds=200.0)
    words, record = style_words.resolve('', tmp_path / 'song.wav', tmp_path, listener=listener, run=media)
    assert words == 'dream pop, shimmering guitars, mid-tempo, wistful, hazy production'
    assert record['style_source'] == 'auto' and record['excerpt'] == {'start': 56.0, 'seconds': 40.0, 'song_seconds': 200.0}
    assert heard == [(5000, 'audio/mp3', True)]
    assert not (tmp_path / 'style-excerpt.mp3').exists()                  # the excerpt is not kept


@pytest.mark.parametrize('failure', [RuntimeError('model down'), TimeoutError('slow'), KeyError('choices')])
def test_a_failing_listener_never_fails_the_cover(tmp_path, failure):
    def listener(audio, mime, prompt):
        raise failure

    words, record = style_words.resolve('', tmp_path / 'song.wav', tmp_path, listener=listener, run=FakeMedia())
    assert words == '' and record['style_source'] == 'default' and type(failure).__name__ in record['why']


def test_an_answer_that_is_not_tags_leaves_the_default_words(tmp_path):
    words, record = style_words.resolve('', tmp_path / 'song.wav', tmp_path, listener=lambda *a: 'Sorry, I cannot listen.',
                                        run=FakeMedia())
    assert words == '' and record['style_source'] == 'default' and 'tag line' in record['why']


def test_an_excerpt_that_is_empty_or_too_big_is_refused(tmp_path):
    for size in (0, style_words.MAX_EXCERPT_BYTES + 1):
        _, record = style_words.resolve('', tmp_path / 'song.wav', tmp_path, listener=never, run=FakeMedia(size=size))
        assert record['style_source'] == 'default' and 'STYLE_AUDIO_INVALID' in record['why']


def test_the_listener_is_named_as_module_and_function():
    assert style_words.load_listener('') is None and style_words.load_listener('nothing') is None
    assert style_words.load_listener('listener_openai:listen').__name__ == 'listen'


def test_the_example_listener_reads_everything_from_the_environment(monkeypatch):
    import listener_openai
    seen = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, limit):
            return json.dumps({'choices': [{'message': {'content': 'lo-fi hip hop, mellow keys, slow'}}]}).encode()

    class Opener:
        def open(self, request, timeout):
            seen['url'], seen['auth'], seen['body'] = request.full_url, request.get_header('Authorization'), json.loads(request.data)
            return Response()

    monkeypatch.setattr(listener_openai, 'build_opener', lambda *handlers: Opener())
    monkeypatch.setenv('YUE2_COVER_LISTENER_URL', 'https://listener.invalid/v1/')
    monkeypatch.setenv('YUE2_COVER_LISTENER_KEY', 'test-key')
    monkeypatch.setenv('YUE2_COVER_LISTENER_MODEL', 'any-audio-model')
    assert listener_openai.listen(b'abc', 'audio/mp3', 'prompt') == 'lo-fi hip hop, mellow keys, slow'
    assert seen['url'] == 'https://listener.invalid/v1/chat/completions' and seen['auth'] == 'Bearer test-key'
    assert seen['body']['model'] == 'any-audio-model'
    assert seen['body']['messages'][0]['content'][0]['input_audio']['format'] == 'mp3'


# --- the lyric check -------------------------------------------------------------------------------------------------------

LINES = ['青石桥边落满花', '远处钟声慢慢响', '风把旧信吹过河', '我数星星等天亮']


def request_of(lines):
    reference = '[Verse]\nreference line'
    return {'lyrics': reference + '\n\n[Chorus]\n' + '\n'.join(lines) + '\n'}


def test_the_planned_lines_are_those_after_the_reference_blocks():
    assert lyric_check.planned_lines(request_of(LINES)) == LINES
    with_interlude = {'lyrics': '[Verse]\nreference line\n\n[Interlude]\n\n[Chorus]\n' + '\n'.join(LINES) + '\n'}
    assert lyric_check.planned_lines(with_interlude) == LINES


def test_every_line_heard_in_order_scores_one():
    assert lyric_check.heard_share(LINES, [''.join(lyric_check.plain(l) for l in LINES)]) == 1.0


def test_a_line_that_is_not_there_is_not_heard():
    heard = ''.join(lyric_check.plain(l) for l in LINES[:2] + ['完全听不出来的别的话'] + LINES[3:])
    assert lyric_check.heard_share(LINES, [heard]) == 0.75


def test_a_chorus_sung_clearly_once_does_not_count_for_all_its_repeats():
    lines = LINES[:2] + LINES[:2] + LINES[:2]                  # the chorus three times in the plan
    heard = ''.join(lyric_check.plain(l) for l in LINES[:2])   # but only one clear time in the take
    assert lyric_check.heard_share(lines, [heard]) == pytest.approx(2 / 6)


def test_the_better_of_the_heard_texts_counts_and_no_lines_scores_zero():
    good, bad = ''.join(lyric_check.plain(l) for l in LINES), '乱七八糟的东西'
    assert lyric_check.heard_share(LINES, [bad, good]) == 1.0
    assert lyric_check.heard_share([], [good]) == 0.0 and lyric_check.heard_share(LINES, []) == 0.0
