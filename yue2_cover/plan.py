"""Plan one YuE2 cover from the song's own score: reference choice, composite score, lyric sections, held tails.

The recipe (each rule fixed a failure we heard; see the README):
- lyrics carry one native `[Label]` block per `% label` section of the score, empty for instrumental ones;
- the singer's reference clip is the nearest tempo in the library, stretched as little as possible and
  shifted so its first sung note lands where its score puts it;
- a line with more notes than syllables gets its line-end same-pitch pair tied (SheetSage splits held
  notes, and YuE re-sings each attack); every other note stays as transcribed;
- with no lyrics given, lines Whisper misheard take the words of the song's published lyrics when a lookup found
  them (lyrics_step.py); the heard lines are placed in time first, so repeated choruses stay where they were sung.
  A published line Whisper never heard is sung at its published time when those times fit this recording.
"""
from __future__ import annotations

import bisect
import difflib
import json
import math
import re
import secrets
import statistics
from fractions import Fraction
from pathlib import Path

import paths
from abc_tools import parse_abc
from compose_score import compose

HAN = re.compile(r'[一-鿿]')
# A sung unit: a Chinese character (Japanese kanji too), a Korean syllable block, a Japanese kana with the small kana it
# joins, or a Latin-script word.
UNIT = re.compile(r"[一-鿿]|[가-힣]|[ぁ-ゖァ-ヺ][ぁぃぅぇぉゃゅょゎァィゥェォャュョヮ]?ー?|[A-Za-z][A-Za-z']*")
KANA = re.compile(r'[ぁ-ゖァ-ヺ]')
HANGUL = re.compile(r'[가-힣]')
LATIN = re.compile(r'[A-Za-z]')
NOISE = ('作词', '作曲', '编曲', '混音', '母带', '字幕', '独播', '剧场', '词曲', '订阅', '点赞', '转发', '打赏', 'Amara')
DEFAULT_ARRANGEMENT = 'emotional C-pop ballad'
MAX_STRETCH = 1.18


def style_for(language, arrangement):
    """Fixed singer description; the caller's arrangement words replace only the genre slot."""
    arrangement = re.sub(r'[\[\]\r\n\t]+', ' ', arrangement or '').strip()
    arrangement = re.sub(r'\s+', ' ', arrangement)[:200] or DEFAULT_ARRANGEMENT
    singer = {'zh': 'Mandarin female vocal', 'en': 'English female vocal', 'ja': 'Japanese female vocal',
              'ko': 'Korean female vocal'}.get(language, 'female vocal')
    return (f'{singer}, clear gentle young female voice, {arrangement}, '
            'faithful original vocal melody and phrasing, natural studio recording')


class Score:
    """The target transcription with bar times from SheetSage's downbeats."""

    def __init__(self, score_dir: Path):
        self.text = (score_dir / 'target-score.abc').read_text(encoding='utf-8')
        self.score = parse_abc(self.text)
        self.vocal = self.score.voices['Vocal']
        self.bars = self.vocal.bars
        downs = [float(line.split()[0]) for line in (score_dir / 'downbeat.lab').read_text().splitlines() if line.strip()]
        if len(downs) != len(self.bars):
            # Bars are the score's own grid; fall back to the nominal tempo where SheetSage's counts disagree.
            spb = 60 / self.score.bpm
            first = downs[0] if downs else 0.0
            downs = [first + float(bar[0]) * spb for bar in self.bars]
        self.downs = downs
        if not self.vocal.notes:
            raise ValueError('YUE2_NO_VOCAL')

    def seconds(self, q):
        bars, downs = self.bars, self.downs
        for i, (start, length, _) in enumerate(bars):
            if start <= q < start + length or i == len(bars) - 1:
                nxt = downs[i + 1] if i + 1 < len(downs) else downs[i] + (downs[i] - downs[i - 1] if i else float(length) * 60 / self.score.bpm)
                return downs[i] + float((q - start) / length) * (nxt - downs[i])

    def sections(self):
        """[(label, t0, t1)] in score order, from the `% label` comments and the Vocal bar counts."""
        found, voice = [], None
        for line in self.text.splitlines():
            if line.startswith('% '):
                found.append([line[2:], 0])
            elif line.startswith('V:'):
                voice = line.split()[1]
            elif voice == 'Vocal' and found and not re.match(r'^[A-Z]:', line):
                found[-1][1] += sum(int(m.group(1) or 1) if m else 1
                                    for m in (re.fullmatch(r'\s*Z([2-4])?\s*', b) for b in line[:-1].split('|')))
        out, b = [], 0
        for label, n in found:
            q1 = self.bars[b + n - 1][0] + self.bars[b + n - 1][1]
            out.append((label, self.downs[b], self.seconds(q1) if b + n < len(self.bars) else float('inf')))
            b += n
        if b != len(self.bars):
            raise ValueError('YUE2_SECTION_BARS')
        return out


def load_references():
    return json.loads((paths.REFERENCES / 'refs.json').read_text(encoding='utf-8'))


# A song whose detected tempo is outside MAX_STRETCH of every clip (one song detected at 158 bpm, a ~79 bpm
# ballad; clips span 60.7-127.3) is read at half or double time first: beat trackers are often off by 2x, and a clip
# counted at twice (or half) its own tempo lines up with the song's grid without any audio change beyond the usual
# stretch. Only when no reading fits is the nearest clip used with its stretch clamped to MAX_STRETCH.
# The reading goes into plan.json and the receipt (reference_pick.tempo_reading).
READINGS = (('half-time', 2), ('double-time', 0.5))


def counted(ref, factor):
    """A clip entry counted at factor x its own tempo (its score is rescaled in reference_plan)."""
    return {**ref, 'true_bpm': ref['true_bpm'] * factor, 'score_bpm': round(ref['score_bpm'] * factor),
            'first_note_quarter': str(Fraction(ref['first_note_quarter']) * Fraction(factor)), 'tempo_factor': factor}


def tempo_view(target_bpm, refs, exclude=()):
    """(clip entries to pick from, reading): the clips as they are when one fits the song's tempo; else the half- or
    double-time reading with the smallest stretch that fits; else every reading, to be clamped ('clamped')."""
    refs = [r for r in refs if r['clip'] not in set(exclude)]
    if not refs:
        raise ValueError('YUE2_TEMPO_OUT_OF_RANGE')

    def fits(view):
        return [r for r in view if abs(math.log(target_bpm / r['true_bpm'])) <= math.log(MAX_STRETCH)]

    if fits(refs):
        return refs, 'direct'
    options = [(name, [counted(r, factor) for r in refs]) for name, factor in READINGS]
    fitting = [(min(abs(math.log(target_bpm / r['true_bpm'])) for r in fits(view)), name, view)
               for name, view in options if fits(view)]
    if fitting:
        _, name, view = min(fitting, key=lambda x: x[0])
        return view, name
    view = refs + [r for _, v in options for r in v]
    return [{**r, 'clamped': True} for r in view], 'clamped'


def scale_score(text, factor):
    """A clip's score counted at factor x its tempo: every duration and meter times factor, Q times factor."""
    lines = text.splitlines()
    unit = Fraction(1, int(lines[3].split('/')[1])) * Fraction(factor)
    if unit.numerator != 1:
        raise ValueError('YUE2_TEMPO_READING_UNIT')
    lines[3] = f'L:1/{unit.denominator}'
    lines[4] = f"Q:1/4={max(1, round(int(lines[4].split('=')[1]) * factor))}"
    for i, line in enumerate(lines):
        if line.startswith('M:'):
            n, d = (int(x) for x in line[2:].split('/'))
            meter = Fraction(n, d) * Fraction(factor)
            lines[i] = f'M:{n}/{d // 2}' if factor == 2 and d > 1 else (
                f'M:{n}/{d * 2}' if factor == 0.5 else f'M:{meter.numerator}/{meter.denominator}')
    return '\n'.join(lines) + '\n'


def choose_reference(target_bpm, refs, exclude=()):
    """Nearest tempo on a log scale; the stretch is the one source of timbre risk we control.

    exclude: clips already tried (a retry after early stops may take the next-closest one)."""
    refs = [r for r in refs if r['clip'] not in set(exclude)]
    if not refs:
        raise ValueError('YUE2_TEMPO_OUT_OF_RANGE')
    best = min(refs, key=lambda r: abs(math.log(target_bpm / r['true_bpm'])))
    rate = target_bpm / best['true_bpm']
    if not 1 / MAX_STRETCH <= rate <= MAX_STRETCH and not best.get('clamped'):
        raise ValueError('YUE2_TEMPO_OUT_OF_RANGE')
    return best, min(max(rate, 1 / MAX_STRETCH), MAX_STRETCH)


# Reference pick: tempo first, then the clip whose own drum amount is closest to the song's
# (a cover's backing drums follow its reference clip, study r = 0.66-0.75; the reference sets nothing else measurable),
# picked at random among clips that match about equally well. features.py measures both on one scale.
TEMPO_BAND = 1.03   # every clip within 3% of the song's tempo is a tempo match (job.SWITCH_STRETCH: not "stretched")
DRUM_TIE = 0.015    # drum shares this close to the best match count as the same (songs span 0.04-0.22)


def load_reference_features():
    """{clip: features.measure(clip)} written when the library was installed; {} when absent (pick by tempo only)."""
    try:
        return json.loads((paths.REFERENCES / 'features.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}


def pick_reference(target_bpm, refs, song, measured, exclude=(), choice=secrets.choice):
    """(clip entry, record). The nearest tempo (choose_reference) unless several clips are within TEMPO_BAND of the
    song's tempo: then the closest drum amount to the song's, at random among those within DRUM_TIE of the best.
    song: features.json of the song (None: not measured); measured: load_reference_features(). The record (candidates,
    their stretch and drums, the rule, the pick) goes into plan.json and so into the receipt."""
    nearest, _ = choose_reference(target_bpm, refs, exclude)
    band = [r for r in refs if r['clip'] not in set(exclude)
            and abs(math.log(target_bpm / r['true_bpm'])) <= math.log(TEMPO_BAND)]
    song_drums = (song or {}).get('drums')
    rows = [{'clip': r['clip'], 'rate': round(target_bpm / r['true_bpm'], 4),
             'drums': (measured.get(r['clip']) or {}).get('drums')} for r in band]
    record = {'song_drums': song_drums, 'candidates': rows}
    if len(band) < 2 or not isinstance(song_drums, (int, float)) or any(row['drums'] is None for row in rows):
        record.update(rule='tempo', pick=nearest['clip'])
        return nearest, record
    best = min(abs(row['drums'] - song_drums) for row in rows)
    tied = [r for r, row in zip(band, rows) if abs(row['drums'] - song_drums) <= best + DRUM_TIE]
    ref = tied[0] if len(tied) == 1 else choice(tied)
    record.update(rule='tempo+drums' + ('+random' if len(tied) > 1 else ''), tied=[r['clip'] for r in tied],
                  pick=ref['clip'])
    return ref, record


def song_features(work):
    """features.json the job measured from the song beside the score step; None when missing or broken."""
    try:
        value = json.loads((Path(work) / 'features.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def reference_plan(ref, target_text):
    """Composite score plus the exact audio edit that aligns the clip to its own score at the target tempo."""
    ref_text = (paths.REFERENCES / ref['clip'] / 'score.abc').read_text(encoding='utf-8')
    if ref.get('tempo_factor'):
        ref_text = scale_score(ref_text, ref['tempo_factor'])
    target_bpm = parse_abc(target_text).bpm
    rate = target_bpm / ref['true_bpm']
    if ref.get('clamped'):
        rate = min(max(rate, 1 / MAX_STRETCH), MAX_STRETCH)
    joined, _, _, invariant, timing = compose(ref_text, target_text, reference_audio_seconds=ref['seconds'])
    if not invariant['pass']:
        raise ValueError('YUE2_COMPOSE_INVARIANT')
    # First sung note: where the score puts it at the target tempo vs where the stretched audio has it.
    lead = float(Fraction(ref['first_note_quarter'])) * 60 / target_bpm - ref['first_note_seconds'] / rate
    return joined, {'rate': rate, 'lead_seconds': lead, 'samples': timing['reference_samples_48khz'],
                    'frames': timing['reference_codec_frames'], 'boundary_quarters': timing['reference_boundary_quarters'],
                    'composite_seconds': timing['composite_seconds']}


def asr_units(asr):
    """(unit, start) from Whisper words; credits and subtitle hallucinations dropped."""
    out = []
    for seg in asr['segments']:
        text = seg['text']
        units = UNIT.findall(text)
        if any(x in text for x in NOISE) or (len(units) > 20 and len(set(units)) < 0.3 * len(units)):
            continue
        for w in seg['words']:
            wu = UNIT.findall(w['w'])
            for j, u in enumerate(wu):
                out.append((u.lower(), w['s'] + j * (w['e'] - w['s']) / len(wu)))
    return out


def plain(text):
    return ''.join(UNIT.findall(text)).lower()


def similarity(a, b):
    return difflib.SequenceMatcher(None, plain(a), plain(b), autojunk=False).ratio()


CLOSE = 0.45         # a heard line this close to a published one takes its words
NEXT = 0.4           # ... or this close when it is the published line right after the previous match
LENGTHS = (0.7, 1.45)  # published/heard syllables: a much longer one is two lines heard as one, never forced in


def corrected(lines, published):
    """The heard lines with the words of the published lyrics, as groups [(heard line indexes, new lines)]: each line
    (or two heard lines that are one published line) is matched to the closest published line or pair of consecutive
    lines, and takes its words when close and of a like length. The heard structure stays: a verse the singer left out
    is not put back."""
    pool = [(k, [p]) for k, p in enumerate(published)] + [(k, [a, b]) for k, (a, b) in enumerate(zip(published, published[1:]))]

    def best(text, after):
        n = len(UNIT.findall(text))
        scored = [(similarity(text, ' '.join(c)), k == after, k, c) for k, c in pool
                  if n and LENGTHS[0] <= len(UNIT.findall(' '.join(c))) / n <= LENGTHS[1]]
        return max(scored, key=lambda x: (x[0], x[1]), default=(0.0, False, None, None))

    def taken(found):
        return found[3] is not None and (found[0] >= CLOSE or found[0] >= NEXT and found[1])

    groups, i, changed, after = [], 0, 0, None
    while i < len(lines):
        one = best(lines[i], after)
        if i + 1 < len(lines):
            two = best(lines[i] + ' ' + lines[i + 1], after)
            if taken(two) and two[0] >= max(one[0], best(lines[i + 1], after)[0]) + 0.1:
                groups.append(([i, i + 1], list(two[3])))
                changed += 1
                after = two[2] + len(two[3])
                i += 2
                continue
        if taken(one):
            groups.append(([i], list(one[3])))
            changed += one[3] != [lines[i]]
            after = one[2] + len(one[3])
        else:
            groups.append(([i], [lines[i]]))
            after = None
        i += 1
    return groups, changed


def retimed(groups, lyric, times, reliable):
    """New lines, units, times and reliability: each group keeps its heard lines' own time span. A line with as many
    syllables as it was heard with keeps every syllable's time; otherwise its syllables are spread over the span."""
    lines, out_lyric, out_times, out_reliable = [], [], [], []
    for heard, new in groups:
        idx = [k for k, (_, li) in enumerate(lyric) if li in heard]
        units = [u.lower() for line in new for u in UNIT.findall(line)]
        same = len(units) == len(idx)
        t0, t1 = min(times[k] for k in idx), max(times[k] for k in idx)
        n = 0
        for line in new:
            for u in UNIT.findall(line):
                out_lyric.append((u.lower(), len(lines)))
                out_times.append(times[idx[n]] if same else t0 + (t1 - t0) * n / max(1, len(units) - 1))
                out_reliable.append(reliable[idx[n]] if same else False)
                n += 1
            lines.append(line)
    return lines, out_lyric, out_times, out_reliable


SYNC_LINES = 4          # matched lines needed before the published times are trusted for this recording
SYNC_SECONDS = 1.0      # ... and most of them must start this close to where the published lyrics put them
SYNC_SHARE = 0.6        # ... "most" being this share
FILL_MARGIN = 0.15      # a filled line starts this long after the line before it and ends this long before the next
SECONDS_PER_SYLLABLE = 1.0  # longest a filled line is spread over, per syllable (+ FILL_MARGIN)


def timed_fill(lines, lyric, times, reliable, timed, onsets, bounds):
    """Lines Whisper never heard, or heard as nothing like the song, sung at the published lyrics' own times when
    those are timed to this very recording: most lines that matched start where the published lyrics say, give or
    take a second (a chorus's first line Whisper never heard would otherwise go unsung, its notes empty). Each
    published line no heard line matched comes in at its own time if it falls between the matched lines around it and
    the melody has notes there, on those notes, in place of whatever was heard over that time; other heard lines stay.
    Another recording of the song (live, a cover) is not timed like the published lyrics, and nothing changes.
    `timed` is [(seconds, text)], `onsets` the seconds the melody's notes start and `bounds` the first and last second
    it covers. Returns lines, lyric, times, reliable and how many lines were filled."""
    spans = []
    for li in range(len(lines)):
        at = [times[k] for k, (_, l) in enumerate(lyric) if l == li]
        spans.append((min(at), max(at)))
    matched, offsets = {}, []
    for li, line in enumerate(lines):
        same = [k for k, (_, text) in enumerate(timed) if plain(text) == plain(line)]
        if same:  # a repeated line is the repeat nearest in time
            k = min(same, key=lambda k: abs(spans[li][0] - timed[k][0]))
            matched[li] = k
            offsets.append(spans[li][0] - timed[k][0])
    unchanged = (lines, lyric, times, reliable, 0)
    if len(offsets) < SYNC_LINES:
        return unchanged
    offset = statistics.median(offsets)
    if sum(abs(x - offset) <= SYNC_SECONDS for x in offsets) < SYNC_SHARE * len(offsets):
        return unchanged
    due = [(t + offset, text) for t, text in timed]
    used = set(matched.values())
    windows, start = [], bounds[0]       # the stretches between one matched line and the next
    for li in sorted(matched):
        windows.append((start, spans[li][0]))
        start = max(start, spans[li][1])
    windows.append((start, bounds[1]))
    filled = []
    for lo, hi in windows:
        # Before the first sung line the window opens SYNC_SECONDS ahead of the melody's first note: published times
        # run a few tenths early, and a first line due 0.45 s before the first note was never let in.
        floor = bounds[0] - SYNC_SECONDS if lo == bounds[0] else lo + FILL_MARGIN
        for k, (t, text) in enumerate(due):
            if k in used or not floor <= t <= hi - 0.5:
                continue
            n = len(UNIT.findall(text))
            following = next((u for u, _ in due[k + 1:] if u > t), hi)
            end = min(hi, following, t + SECONDS_PER_SYLLABLE * n + FILL_MARGIN) - FILL_MARGIN
            notes = [on for on in onsets if max(lo, t - 0.3) < on <= end]
            if n and len(notes) >= max(1, n // 2):
                # On the melody's own notes when it has one for each syllable, else spread over the line's time.
                at = notes[:n] if len(notes) >= n else [t + (end - t) * j / max(1, n - 1) for j in range(n)]
                filled.append((min(t, at[0]), end, text, at))
    if not filled:
        return unchanged
    entries = [(t, text, [(u.lower(), x, False) for u, x in zip(UNIT.findall(text), at)]) for t, _, text, at in filled]
    for li, line in enumerate(lines):
        a, b = spans[li]
        if li in matched or not any(a <= end and b >= t for t, end, _, _ in filled):
            entries.append((a, line, [(u, times[k], reliable[k]) for k, (u, l) in enumerate(lyric) if l == li]))
    entries.sort(key=lambda e: e[0])
    out_lines, out_lyric, out_times, out_reliable = [], [], [], []
    for _, line, units in entries:
        for u, t, ok in units:
            out_lyric.append((u, len(out_lines)))
            out_times.append(t)
            out_reliable.append(ok)
        out_lines.append(line)
    return out_lines, out_lyric, out_times, out_reliable, len(filled)


def load_lookup(work: Path):
    """(published lines, {'id', 'score'}) from work/lyrics-lookup.json (lyrics_step.py), or (None, None): no file, a
    corrupt file, or no song found all leave the heard lyrics as they are."""
    path = work / 'lyrics-lookup.json'
    if not path.is_file():
        return None, None
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
        song = value.get('song')
        published = [line['text'] for line in value.get('lines') or []] if song else []
    except (ValueError, AttributeError, KeyError, TypeError):
        return None, None
    return (published, {'id': song.get('id'), 'score': value.get('score')}) if published else (None, None)


def load_timed(work: Path):
    """[(seconds, text)] of the published lines that carry a time, from the file load_lookup reads; [] wherever
    load_lookup finds nothing usable."""
    if not load_lookup(work)[0]:
        return []
    value = json.loads((work / 'lyrics-lookup.json').read_text(encoding='utf-8'))
    return [(float(line['t']), line['text']) for line in value['lines']
            if isinstance(line.get('t'), (int, float)) and math.isfinite(line['t'])]


def text_language(lines, heard=None):
    """zh, ja, ko, en or other, from the sung units themselves (Whisper's guess only as the last word)."""
    units = [u for line in lines for u in UNIT.findall(line)]
    if not units:
        return heard if heard in ('zh', 'en', 'ja', 'ko') else 'other'
    kana, hangul = sum(bool(KANA.match(u)) for u in units), sum(bool(HANGUL.match(u)) for u in units)
    han, latin = sum(bool(HAN.match(u)) for u in units), sum(bool(LATIN.match(u)) for u in units)
    if kana >= 0.15 * len(units):
        return 'ja'
    if hangul and hangul >= max(han, latin):
        return 'ko'
    if han >= 0.5 * len(units):
        return 'zh'
    if latin >= 0.5 * len(units):
        return 'en'
    return heard if heard in ('zh', 'en', 'ja', 'ko') else 'other'


def lrc_lines(lyrics):
    """[(seconds, line)] when the lyrics given are timed lyrics (LRC, the original's as the client found them), else []."""
    from lyrics_step import parse_lrc
    timed = parse_lrc(lyrics or '')
    return timed if len(timed) >= 4 else []


LRC_SHIFT = 20.0    # the recording may run this far ahead of or behind the published times (an MV's intro)
LRC_NEAR = 0.4      # a published line start this close to a phrase's first note lines up with it


def phrase_starts(onsets, ends, gap=0.25):
    return [on for k, on in enumerate(onsets) if k == 0 or on - ends[k - 1] >= gap]


def note_shift(starts, phrases):
    """The shift that puts the most published line starts on the first note of a sung phrase."""
    import bisect

    def fit(shift):
        total = 0.0
        for t in starts:
            x = t + shift
            k = bisect.bisect_left(phrases, x)
            near = min((abs(x - phrases[j]) for j in (k - 1, k) if 0 <= j < len(phrases)), default=LRC_NEAR)
            total += max(0.0, 1 - near / LRC_NEAR)
        return total
    steps = int(LRC_SHIFT * 20)
    return max(range(-steps, steps + 1), key=lambda d: (fit(d / 20), -abs(d))) / 20


def lrc_place(timed, units, score):
    """The original's timed lyrics placed on this recording: one shift for the whole song, from the words Whisper heard
    when enough of them line up (as in timed_fill), else from the melody (line starts on phrase starts); each line's
    units go on the notes inside its own time. A line with no notes there (another version's part) is left out.
    Returns lines, lyric, times, reliable and how the shift was found."""
    notes = score.vocal.notes
    onsets = [float(score.seconds(n[0])) for n in notes]
    ends = [float(score.seconds(n[0] + n[2])) for n in notes]
    lines = [text for _, text in timed]
    lyric, heard_at, heard = place_lines(lines, units, score)
    offsets = []
    for li, (t, _) in enumerate(timed):
        first = next((k for k, (_, l) in enumerate(lyric) if l == li), None)
        if first is not None and heard[first]:
            offsets.append(heard_at[first] - t)
    shift, how = None, 'notes'
    if len(offsets) >= SYNC_LINES:
        middle = statistics.median(offsets)
        if sum(abs(x - middle) <= SYNC_SECONDS for x in offsets) >= SYNC_SHARE * len(offsets):
            shift, how = middle, 'heard'
    if shift is None:
        shift = note_shift([t for t, _ in timed], phrase_starts(onsets, ends))
    return (*place_timed(timed, lambda t: t + shift, score), f'{how}:{shift:+.2f}')


SPREAD_SECONDS = 0.3


def place_timed(timed, mapped, score, disjoint=False, sung=()):
    """Each timed line's units on the melody notes inside its own time on this recording (mapped(t): the line's start
    here); a line with no notes there is left out. Returns lines, lyric, times, reliable.
    disjoint: a line's notes end where the next line's may begin (its start - 0.3 s). Without it (the foreign-song path
    as it was) a line whose first note the line before had already taken was left out: 4 of one song's 24 lines.
    sung: published times of lines heard on time; such a line stays even where the score has no notes for it, spread
    over its own time (one song had a short line sung over rests: SheetSage wrote rests under it; left out, the interlude block
    was empty and the next block began with them, 3.8 s off)."""
    notes = score.vocal.notes
    onsets = [float(score.seconds(n[0])) for n in notes]
    ends = [float(score.seconds(n[0] + n[2])) for n in notes]
    out_lines, out_lyric, out_times, out_reliable = [], [], [], []
    last = ends[-1] if ends else 0.0
    for k, (t, text) in enumerate(timed):
        words = UNIT.findall(text)
        start = mapped(t)
        following = mapped(timed[k + 1][0]) if k + 1 < len(timed) else last + 0.5
        end = min(following - (0.2 if disjoint else 0.0), start + SECONDS_PER_SYLLABLE * len(words) + 1.0) - 0.1
        span = [on for on in onsets if start - 0.3 <= on < end and not (disjoint and out_times and on <= out_times[-1])]
        if words and not span and t in sung and not (out_times and start <= out_times[-1]):
            span = [start]
        elif not words or not span or (out_times and span[0] <= out_times[-1]):
            continue
        if len(span) >= len(words):  # spread over the line's notes: a word of several syllables takes several
            at = [span[j * len(span) // len(words)] for j in range(len(words))]
        else:
            # More syllables than notes: spread to the line's end; with disjoint lines only SPREAD_SECONDS a syllable
            # past its notes, so its last syllable (which decides its section) is not put at the next line's start.
            stop = min(end, max(span[-1], span[0] + SPREAD_SECONDS * (len(words) - 1))) if disjoint else end
            at = [span[0] + (stop - span[0]) * j / max(1, len(words) - 1) for j in range(len(words))]
        for j, (u, a) in enumerate(zip(words, at)):
            out_lyric.append((u.lower(), len(out_lines)))
            out_times.append(a)
            out_reliable.append(j == 0)  # the line's start is known; inside it the melody decides
        out_lines.append(text)
    return out_lines, out_lyric, out_times, out_reliable


# Timed lyrics sung as they are.
# A Chinese song used to keep the lines Whisper heard and only swap in the published words, so every mishearing of the
# line structure (an echo of the prompt, two lines heard as one, a line never heard) reached the singer. When timed
# lyrics fit this recording - most of their lines are heard within SYNC_SECONDS of where their times say, after one
# shift (and a tempo scale if this version runs faster or slower) - their own lines and times are the plan, and Whisper
# only checks the fit and fills a sung line they lack. Candidates in order: the client's LRC (found by the song's
# title and length), then the lookup's (found from the heard words); none fits: the heard lines as before.
MATCH_HEARD = 0.75       # share of the heard lines that start where their neighbours' shift says (one song 19/19)
MATCH_LINES = 0.3        # ... and at least this share of the LRC's lines due inside this recording's melody (a cut
                         # of the song, a 33 s chorus clip, is due only its own lines)
MATCH_RATIO = 0.6        # a heard stretch this close to a line is that line heard
SEARCH_WIDE_MAX = 600.0  # the song's shift is voted on by every place a line was heard, this far either side
SEARCH_NEAR = 6.0        # each line is then looked for this far either side of where that shift puts it
MATCH_COVER = 0.75       # share of the heard lines the timed lyrics must have at that time
NEIGHBOURS = 2           # a line's shift is checked against this many heard lines on each side


def line_starts(timed, units, score=None):
    """[(published time, heard start)] for the lines heard. The song's shift is the one most lines agree on, every
    place each line was heard counted (a cut of the song, or a repeated chorus line, then still votes for the right
    place); each line then takes the heard stretch most like it (ratio >= MATCH_RATIO, then the nearest to where it
    is due) within SEARCH_NEAR of that; matches that cross are dropped. Line by line, not one diff over the song: a
    whole-song diff of one song tied its second chorus to the third (+130 s)."""
    heard = [u for u, _ in units]
    at = [t for _, t in units]

    def found(text, lo=None, hi=None):
        a = plain(text)
        n = len(a)
        if not n:
            return []
        i0 = 0 if lo is None else bisect.bisect_left(at, lo)
        i1 = len(at) if hi is None else bisect.bisect_right(at, hi)
        out = []
        for i in range(i0, i1):
            for size in range(max(1, n - 2), n + 3):
                m = difflib.SequenceMatcher(None, a, ''.join(heard[i:i + size]), autojunk=False)
                if m.quick_ratio() >= MATCH_RATIO and m.ratio() >= MATCH_RATIO:
                    out.append((m.ratio(), at[i]))
        return out

    votes = {}
    for k, (t, text) in enumerate(timed):
        for _, h in found(text, t - SEARCH_WIDE_MAX, t + SEARCH_WIDE_MAX):
            votes.setdefault(round(h - t), set()).add(k)
    if not votes:
        return []
    shift = max(votes, key=lambda d: (len(votes.get(d - 1, set()) | votes[d] | votes.get(d + 1, set())), -abs(d)))
    near = []
    for t, text in timed:
        options = found(text, t + shift - SEARCH_NEAR, t + shift + SEARCH_NEAR)
        if options:
            near.append((t, max(options, key=lambda o: (round(o[0], 3), -abs(o[1] - t - shift)))[1]))
    # Longest chain with the heard starts in the published order.
    best, back = [1] * len(near), [-1] * len(near)
    for j in range(len(near)):
        for i in range(j):
            if near[i][1] < near[j][1] and best[i] + 1 > best[j]:
                best[j], back[j] = best[i] + 1, i
    j = max(range(len(near)), key=lambda k: best[k], default=None)
    chain = []
    while j is not None and j >= 0:
        chain.append(near[j])
        j = back[j]
    return chain[::-1]


def heard_cover(timed, mapped, asr):
    """Share of the heard lines (segments of 4+ syllables) that the timed lyrics have at that time: the mapped lines
    starting from 2 s before to 1 s after the segment, read together, at ratio >= 0.5. Low when Whisper heard words
    the timed lyrics do not have there (another part of the song: a cut matched to the song's first lines)."""
    due = [(mapped(t), text) for t, text in timed]
    total = covered = 0
    for seg in asr.get('segments', []):
        text = seg.get('text', '')
        if len(UNIT.findall(text)) < 4 or any(x in text for x in NOISE):
            continue
        total += 1
        there = ''.join(line for t, line in due if seg['start'] - 2.0 <= t <= seg['end'] + 1.0)
        here = plain(text)
        if there and _contained(here, plain(there)) >= 0.5:
            covered += 1
    return covered / total if total else 0.0


def _contained(a, b):
    """How much of a is found in b, in order (matching characters / len(a))."""
    if not a:
        return 0.0
    return sum(m.size for m in difflib.SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks()) / len(a)


def lrc_fit(timed, units, score=None, asr=None):
    """How timed lyrics fit this recording: {'shift' (median), 'spread', 'on_time', 'heard', 'lines', 'share',
    'anchors'}. A heard line is on time when its shift is within SYNC_SECONDS of its NEIGHBOURS' median: one shift for
    the song, or a few (an edit or a pause this version adds, a slightly different tempo), but never line by line.
    anchors: [(published time, shift)] of the on-time lines, which mapped() follows. None when too little was heard."""
    pairs = line_starts(timed, units, score)
    if len(pairs) < SYNC_LINES:
        return None
    offsets = [h - t for t, h in pairs]
    anchors = []
    for k, (t, _) in enumerate(pairs):
        near = offsets[max(0, k - NEIGHBOURS):k + NEIGHBOURS + 1]
        if abs(offsets[k] - statistics.median(near)) <= SYNC_SECONDS:
            anchors.append((t, round(offsets[k], 2)))
    shifts = [o for _, o in anchors] or offsets
    due = len(timed)
    if score is not None and score.vocal.notes:
        notes = score.vocal.notes
        lo, hi = float(score.seconds(notes[0][0])) - 1.0, float(score.seconds(notes[-1][0] + notes[-1][2])) + 1.0
        mapped = mapped_by(anchors) if anchors else (lambda t: t + statistics.median(shifts))
        due = sum(lo <= mapped(t) <= hi for t, _ in timed) or len(timed)
    cover = round(heard_cover(timed, mapped_by(anchors), asr), 3) if asr is not None and anchors else None
    return {'shift': round(statistics.median(shifts), 2), 'spread': round(max(shifts) - min(shifts), 2),
            'on_time': len(anchors), 'heard': len(pairs), 'lines': len(timed), 'due': due,
            'share': round(len(anchors) / due, 3), 'cover': cover, 'anchors': anchors}


MAP_NEAREST = 3          # a line's shift: the median of the on-time lines this many nearest to it


def mapped_by(anchors):
    """t -> its time on this recording: shifted by the median shift of the MAP_NEAREST on-time lines nearest to it.
    A median, not the nearest line's own shift: Whisper's word times wobble by a second (one song: neighbours at +1.1
    and +2.2 s), and one line moved on its own lands where the melody has no notes and drops out; three lines still
    follow a real jump (one song's +2 s pause) from the first line after it."""
    def mapped(t):
        near = sorted(anchors, key=lambda a: (abs(a[0] - t), a[0]))[:MAP_NEAREST]
        return t + statistics.median(o for _, o in near)
    return mapped


def heard_times(lines, lyric, times, reliable, units):
    """Each placed line's syllables take the times Whisper heard them at, where it heard them inside the line's own
    stretch (from 1 s before its start to the next line's start), in order. The held-tail ties are decided from these
    times; spread evenly over the line's notes they tied the wrong pairs (one song: 7 of 8 line-end ties
    moved, and the take sang 27-31 of 54 lines against 41 with the heard lines' times)."""
    heard_at = [t for _, t in units]
    times, reliable = list(times), list(reliable)
    firsts = [next(k for k, (_, l) in enumerate(lyric) if l == li) for li in range(len(lines))]
    for li in range(len(lines)):
        idx = [k for k, (_, l) in enumerate(lyric) if l == li]
        lo = times[idx[0]] - 1.0
        hi = times[firsts[li + 1]] if li + 1 < len(lines) else times[idx[-1]] + 3.0
        i0, i1 = bisect.bisect_left(heard_at, lo), bisect.bisect_left(heard_at, hi)
        heard = [u for u, _ in units[i0:i1]]
        found = difflib.SequenceMatcher(None, [lyric[k][0] for k in idx], heard, autojunk=False)
        pairs = [(a + j, i0 + b + j) for a, b, n in found.get_matching_blocks() for j in range(n)]
        if len(pairs) < max(2, len(idx) // 2):
            continue  # heard too little of it here: the notes keep it
        for a, b in pairs:
            times[idx[a]], reliable[idx[a]] = units[b][1], True
        # Unheard syllables between heard ones: evenly between their neighbours (the line stays in order).
        known = [(a, times[idx[a]]) for a, _ in pairs]
        for (a0, t0), (a1, t1) in zip(known, known[1:]):
            for a in range(a0 + 1, a1):
                times[idx[a]] = t0 + (t1 - t0) * (a - a0) / (a1 - a0)
    return times, reliable


EXTRA_MIN = 4            # syllables a heard line needs before it is added to the timed lyrics


def heard_extra(lines, lyric, times, reliable, asr, units, score, published):
    """Sung lines the timed lyrics lack (a repeat they leave out, an ad-lib): a heard line of EXTRA_MIN+ syllables,
    mostly heard as itself, on melody notes, overlapping no placed line. It takes a published line's words when one is
    close. Returns lines, lyric, times, reliable and how many were added."""
    try:
        heard, _ = lyric_lines('', asr)
    except ValueError:
        return lines, lyric, times, reliable, 0
    h_lyric, h_times, h_reliable = place_lines(heard, units, score)
    spans = []
    for li in range(len(lines)):
        idx = [k for k, (_, l) in enumerate(lyric) if l == li]
        spans.append((min(times[k] for k in idx) - 0.5, max(times[k] for k in idx) + 0.5))
    onsets = [float(score.seconds(n[0])) for n in score.vocal.notes]
    extra = []
    for li, text in enumerate(heard):
        idx = [k for k, (_, l) in enumerate(h_lyric) if l == li]
        n = len(idx)
        if n < EXTRA_MIN or sum(h_reliable[k] for k in idx) < 0.6 * n:
            continue
        a, b = h_times[idx[0]], h_times[idx[-1]]
        if any(a <= hi and b >= lo for lo, hi in spans) or sum(a - 0.3 <= on <= b + 0.3 for on in onsets) < n // 2:
            continue
        close = max(published, key=lambda p: similarity(text, p), default=None)
        words = close if close and similarity(text, close) >= CLOSE and \
            LENGTHS[0] <= len(UNIT.findall(close)) / n <= LENGTHS[1] else text
        extra.append((a, b, words))
        spans.append((a - 0.5, b + 0.5))
    if not extra:
        return lines, lyric, times, reliable, 0
    entries = [(min(times[k] for k, (_, l) in enumerate(lyric) if l == li), line, li) for li, line in enumerate(lines)]
    entries += [(a, words, (a, b)) for a, b, words in extra]
    entries.sort(key=lambda e: e[0])
    out_lines, out_lyric, out_times, out_reliable = [], [], [], []
    for _, line, key in entries:
        if isinstance(key, tuple):
            found = UNIT.findall(line)
            for j, u in enumerate(found):
                out_lyric.append((u.lower(), len(out_lines)))
                out_times.append(key[0] + (key[1] - key[0]) * j / max(1, len(found) - 1))
                out_reliable.append(j == 0)
        else:
            for k, (u, l) in enumerate(lyric):
                if l == key:
                    out_lyric.append((u, len(out_lines)))
                    out_times.append(times[k])
                    out_reliable.append(reliable[k])
        out_lines.append(line)
    return out_lines, out_lyric, out_times, out_reliable, len(extra)


def lyric_lines(lyrics, asr):
    """The given lines; without any, Whisper's own segments become the lyrics."""
    lines = [re.sub(r'\s+', ' ', l).strip() for l in (lyrics or '').splitlines()]
    lines = [l for l in lines if UNIT.search(l) and not re.fullmatch(r'\[.*\]', l)]
    if lines:
        return lines, 'user'
    lines = []
    for seg in asr['segments']:
        text = seg['text'].strip()
        if UNIT.search(text) and not any(x in text for x in NOISE):
            lines.extend(p.strip() for p in re.split(r'[，。,.!?！？]', text) if UNIT.search(p))
    # In a Chinese song Whisper hears stray English over instrumental parts ("Zither Harp" in an interlude)
    # and the line would be sung there; a mostly Chinese song keeps only lines with Chinese.
    if sum(len(HAN.findall(l)) for l in lines) >= 0.5 * sum(len(UNIT.findall(l)) for l in lines):
        lines = [l for l in lines if HAN.search(l)]
    if not lines:
        raise ValueError('YUE2_NO_LYRICS')
    return lines, 'asr'


def place_lines(lines, units, score):
    """Unit times for the known lines: matched against ASR, the rest interpolated in order."""
    lyric = [(u.lower(), li) for li, line in enumerate(lines) for u in UNIT.findall(line)]
    matcher = difflib.SequenceMatcher(None, [u for u, _ in lyric], [u for u, _ in units], autojunk=False)
    times, reliable = [None] * len(lyric), [False] * len(lyric)
    for a, b, n in matcher.get_matching_blocks():
        for k in range(n):
            times[a + k], reliable[a + k] = units[b + k][1], True
    last = 0.0
    for i, t in enumerate(times):  # Whisper loops can match backwards in time
        if t is not None and t < last:
            times[i], reliable[i] = None, False
        elif t is not None:
            last = t
    notes = score.vocal.notes
    first, end = score.seconds(notes[0][0]), score.seconds(notes[-1][0] + notes[-1][2])
    anchors = [(-1, first - 0.3)] + [(i, t) for i, t in enumerate(times) if t is not None] + [(len(lyric), end)]
    for (i0, t0), (i1, t1) in zip(anchors, anchors[1:]):
        for i in range(i0 + 1, i1):
            times[i] = t0 + (t1 - t0) * (i - i0) / (i1 - i0)
    return lyric, times, reliable


def assign_sections(lines, lyric, times, sections):
    """Each line belongs to the section of its last syllable's onset (0.5 s slack for held tails / ASR lag)."""
    out = []
    for li in range(len(lines)):
        t = max(times[i] for i, (_, l) in enumerate(lyric) if l == li) - 0.5
        out.append(next((k for k, (_, t0, t1) in enumerate(sections) if t0 <= t < t1), len(sections) - 1))
    return out


def held_tails(lines, lyric, times, reliable, score):
    """Monotone syllable->note DP; then at most one line-end same-pitch tie per line with spare notes."""
    notes = score.vocal.notes
    on = [score.seconds(n[0]) for n in notes]
    gap = [True] + [notes[j][0] - (notes[j - 1][0] + notes[j - 1][2]) >= Fraction(1, 2) for j in range(1, len(notes))]
    first = []
    for line in lines:
        n = len(UNIT.findall(line))
        first += [True] + [False] * (n - 1)
    n, m, R, INF = len(lyric), len(notes), 6, float('inf')

    def solve(delta):
        D = [[INF] * (m + 1) for _ in range(n + 1)]
        B = [[None] * (m + 1) for _ in range(n + 1)]
        D[0][0] = 0.0
        for i in range(n):
            t, wt = times[i] + delta, (0.5 if reliable[i] else 0.08)
            for j in range(m + 1):
                if D[i][j] == INF:
                    continue
                if j and D[i][j] + 3.0 < D[i + 1][j]:
                    D[i + 1][j], B[i + 1][j] = D[i][j] + 3.0, (j, 0)
                if j == m:
                    continue
                cost = D[i][j] + wt * abs(t - on[j]) + ((0.0 if gap[j] else 0.8) if first[i] else (2.5 if gap[j] else 0.0))
                for k in range(j + 1, min(m, j + R) + 1):
                    if k > j + 1:
                        if gap[k - 1]:
                            break
                        cost += 0.4 if notes[k - 1][1] == notes[k - 2][1] else 1.0
                    if cost < D[i + 1][k]:
                        D[i + 1][k], B[i + 1][k] = cost, (j, k - j)
        return D[n][m], B

    cost, delta = min((solve(d / 20)[0], d / 20) for d in range(-16, 17))
    if cost == INF:
        return []
    _, B = solve(delta)
    i, j, runs = n, m, []
    while i > 0:
        pj, ln = B[i][j]
        runs.append((pj, ln))
        i, j = i - 1, pj
    runs.reverse()
    pos, ties = 0, []
    for line in lines:
        k = len(UNIT.findall(line))
        rr = runs[pos:pos + k]
        pos += k
        j0, j1 = rr[0][0], max(a + b for a, b in rr)
        span = notes[j0:j1]
        if len(span) <= k:
            continue
        for x in range(len(span) - 1, max(0, len(span) - 3), -1):
            a, b = span[x - 1], span[x]
            if a[1] == b[1] and a[0] + a[2] == b[0]:
                ties.append(b[0])
                break
    return ties


def tie_notes(text, onsets):
    """Insert '-' after the Vocal note token that ends where each given onset begins (same pitch)."""
    from abc_tools import TOKEN, meter_value
    score = parse_abc(text)
    notes = score.voices['Vocal'].notes
    by_start = {n[0]: n for n in notes}
    lines = text.splitlines()
    meter = meter_value(lines[2][2:])
    unit = Fraction(1, int(lines[3].split('/')[1]))
    tokens, voice, t = [], None, Fraction(0)
    for i, line in enumerate(lines):
        if i < 8 or line.startswith('% '):
            continue
        if line.startswith('V: '):
            voice = line[3:]
            continue
        if line.startswith('M:'):
            if voice == 'Vocal':
                meter = meter_value(line[2:])
            continue
        if line.startswith('K:') or voice != 'Vocal':
            continue
        length, pos = Fraction(4 * meter[0], meter[1]), 0
        for bar in line[:-1].split('|'):
            start, pos = pos, pos + len(bar) + 1
            rest = re.fullmatch(r'\s*Z([2-4])?\s*', bar)
            if rest:
                t += length * int(rest.group(1) or 1)
                continue
            off, c = Fraction(0), 0
            while c < len(bar):
                if bar[c].isspace():
                    c += 1
                    continue
                mt = TOKEN.match(bar, c)
                c = mt.end()
                if mt.group('note') is None:
                    continue
                if mt.group('note') != 'z':
                    tokens.append((i, start + mt.end('duration'), t + off, bool(mt.group('tie'))))
                off += int(mt.group('duration') or 1) * unit * 4
            t += length
    starts = [x[2] for x in tokens]
    # The note that ends where q begins; its last token may be the tail of a note SheetSage already tied (that token's
    # own start is no note start, and looking it up stopped the plan with a KeyError).
    by_end = {n[0] + n[2]: n for n in notes}
    edits = []
    for q in onsets:
        k = starts.index(q)
        line_i, end, first_q, has_tie = tokens[k - 1]
        before = by_end.get(q)
        if not (before is not None and before[0] <= first_q and before[1] == by_start[q][1] and not has_tie):
            raise ValueError('YUE2_TIE_MISMATCH')
        edits.append((line_i, end))
    for line_i, end in sorted(edits, reverse=True):
        lines[line_i] = lines[line_i][:end] + '-' + lines[line_i][end:]
    tied = '\n'.join(lines) + ('\n' if text.endswith('\n') else '')
    if len(parse_abc(tied).voices['Vocal'].notes) != len(notes) - len(onsets):
        raise ValueError('YUE2_TIE_COUNT')
    return tied


def build(work: Path, lyrics: str, arrangement: str, *, exclude=(), out: Path | None = None):
    """Write plan.json and request.json into out (default work); return the plan.

    exclude: reference clips not to use (YUE2_TEMPO_OUT_OF_RANGE when no other one is close enough).
    """
    score = Score(work / 'score')
    asr = json.loads((work / 'asr.json').read_text(encoding='utf-8'))
    refs, reading = tempo_view(score.score.bpm, load_references(), exclude)
    measures = song_features(work)  # not `song`: that name is the lyrics lookup's song below
    ref, pick = pick_reference(score.score.bpm, refs, measures, load_reference_features(), exclude)
    pick = {**pick, 'tempo_reading': reading}
    composite, ref_plan = reference_plan(ref, score.text)
    units = asr_units(asr)
    published, song = load_lookup(work)
    timed = load_timed(work)
    filled = 0
    given = lrc_lines(lyrics)
    # A Chinese song with timed lyrics that fit this recording is sung to them (lrc_fit above): the client's first, then
    # the lookup's. None fits: a Chinese song the lookup found keeps the heard lines, corrected; any other song with
    # timed lyrics is sung to them at one shift (Whisper hears a foreign song over a dense mix badly, a K-pop track).
    heard_chinese = bool(published) and text_language(published) == 'zh'
    match, extra, tried = None, 0, []
    candidates = [('client', given)] if given and text_language([t for _, t in given]) == 'zh' else []
    if timed and text_language([t for _, t in timed]) == 'zh' and (given or not (lyrics or '').strip()):
        candidates.append(('lookup', timed))
    for name, lrc in candidates:
        fit = lrc_fit(lrc, units, score, asr)
        tried.append({'source': name, **{k: v for k, v in (fit or {'share': 0.0, 'lines': len(lrc)}).items()
                                         if k != 'anchors'}})
        if fit and fit['on_time'] >= max(SYNC_LINES, MATCH_HEARD * fit['heard'], MATCH_LINES * fit['due'])\
                and (fit['cover'] or 0) >= MATCH_COVER:
            match = (name, lrc, fit)
            break
    if match:
        name, lrc, fit = match
        lines, lyric, times, reliable = place_timed(lrc, mapped_by(fit['anchors']), score, disjoint=True,
                                                    sung={t for t, _ in fit['anchors']})
        if len(lines) < 4:
            raise ValueError('YUE2_LRC_UNPLACED')
        times, reliable = heard_times(lines, lyric, times, reliable, units)
        lines, lyric, times, reliable, extra = heard_extra(lines, lyric, times, reliable, asr, units, score,
                                                            [t for _, t in lrc])
        lyric_source = ('lrc' if name == 'client' else 'lookup-lrc') + f":fit:{fit['shift']:+.2f}" + (
            f"~{fit['spread']:.1f}" if fit['spread'] > SYNC_SECONDS else '')
        published = None
        song = song if name == 'lookup' else None  # the receipt names the lookup's song only when it was sung
    elif (given or timed) and not heard_chinese and (given or not (lyrics or '').strip()):
        lines, lyric, times, reliable, sync = lrc_place(given or timed, units, score)
        if len(lines) < 4:
            raise ValueError('YUE2_LRC_UNPLACED')
        lyric_source = ('lrc:' if given else 'lookup-lrc:') + sync
        published = None
        song = song if not given else None  # the receipt names the lookup's song only when it was sung
    else:
        lines, lyric_source = lyric_lines('' if given else lyrics, asr)
        lyric, times, reliable = place_lines(lines, units, score)
    if published and lyric_source == 'asr':
        # The heard lines are placed first (their text is what ASR heard, so they line up exactly); then they take the
        # published words inside their own time spans (lyrics_step.py), and lines never heard come in at their
        # published times when those fit this recording.
        groups, changed = corrected(lines, published)
        lines, lyric, times, reliable = retimed(groups, lyric, times, reliable)
        notes = score.vocal.notes
        onsets = [float(score.seconds(n[0])) for n in notes]
        bounds = (onsets[0] - 0.3, float(score.seconds(notes[-1][0] + notes[-1][2])))
        lines, lyric, times, reliable, filled = timed_fill(lines, lyric, times, reliable, timed, onsets, bounds)
        lyric_source = f'asr+lookup:{changed}'
    language = text_language(lines, asr.get('language'))
    sections = score.sections()
    assigned = assign_sections(lines, lyric, times, sections)
    # Ties only for Chinese: one character is one syllable, so spare notes mean held tails.
    ties = held_tails(lines, lyric, times, reliable, score) if language == 'zh' else []
    boundary = Fraction(ref_plan['boundary_quarters'])
    abc_text = tie_notes(composite, [q + boundary for q in ties]) if ties else composite
    labels = [l[2:] for l in abc_text.splitlines() if l.startswith('% ')]
    target_labels = [label for label, _, _ in sections]
    ref_n = len(labels) - len(target_labels)
    if labels[ref_n:] != target_labels or labels[:ref_n] not in (['verse'], ['verse', 'interlude']):
        raise ValueError('YUE2_SECTION_LABELS')
    blocks = [[] for _ in labels]
    blocks[0] = list(ref['lyrics'])
    for line, k in zip(lines, assigned):
        blocks[ref_n + k].append(line)
    text = '\n\n'.join('[' + label.title() + ']' + ''.join('\n' + l for l in block)
                       for label, block in zip(labels, blocks)) + '\n'
    request = {'style': style_for(language, arrangement), 'lyrics': text, 'cot': 'full', 'seed': 42,
               'cfg_scale': 1.0, 'id': 'cover-song', 'abc': abc_text}
    max_tokens = math.ceil(ref_plan['composite_seconds'] * 25 * 1.10)
    if max_tokens >= 24576 - 4096:
        raise ValueError('YUE2_SONG_TOO_LONG')
    plan = {'reference': ref['clip'], 'reference_rate': ref_plan['rate'], 'lead_seconds': ref_plan['lead_seconds'],
            'reference_samples': ref_plan['samples'], 'reference_frames': ref_plan['frames'],
            'reference_seconds': ref_plan['samples'] / 48000, 'max_tokens': max_tokens,
            'target_bpm': score.score.bpm, 'reference_pick': pick,
            'song_features': {k: v for k, v in (measures or {}).items() if k != 'measure_seconds'} or None,
            'language': language, 'lyric_source': lyric_source,
            'lines': len(lines), 'timed_lines': filled, 'ties': len(ties), 'sections': target_labels,
            **({'lyric_match': {'tried': tried, 'chosen': match[0] if match else 'none', 'heard_extra': extra}}
               if tried else {}),
            **({'lyrics_song': song} if song and lyric_source.startswith(('asr+lookup', 'lookup-lrc')) else {})}
    out = Path(out) if out is not None else work
    out.mkdir(parents=True, exist_ok=True)
    (out / 'request.json').write_text(json.dumps(request, ensure_ascii=False, indent=1), encoding='utf-8')
    (out / 'plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=1), encoding='utf-8')
    return plan
