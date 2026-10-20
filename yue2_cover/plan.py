"""Plan one YuE2 cover from the song's own score: reference choice, composite score, lyric sections, held tails.

The recipe (each rule fixed a failure we heard; see the README):
- lyrics carry one native `[Label]` block per `% label` section of the score, empty for instrumental ones;
- the singer's reference clip is the nearest tempo in the library, stretched as little as possible and
  shifted so its first sung note lands where its score puts it;
- a line with more notes than syllables gets its line-end same-pitch pair tied (SheetSage splits held
  notes, and YuE re-sings each attack); every other note stays as transcribed.
"""
from __future__ import annotations

import difflib
import json
import math
import re
from fractions import Fraction
from pathlib import Path

import paths
from abc_tools import parse_abc
from compose_score import compose

HAN = re.compile(r'[一-鿿]')
UNIT = re.compile(r"[一-鿿]|[A-Za-z][A-Za-z']*")
NOISE = ('作词', '作曲', '编曲', '混音', '母带', '字幕', '独播', '剧场', '词曲', '订阅', '点赞', '转发', '打赏', 'Amara')
DEFAULT_ARRANGEMENT = 'emotional C-pop ballad'
MAX_STRETCH = 1.18


def style_for(language, arrangement):
    """Fixed singer description; the caller's arrangement words replace only the genre slot."""
    arrangement = re.sub(r'[\[\]\r\n\t]+', ' ', arrangement or '').strip()
    arrangement = re.sub(r'\s+', ' ', arrangement)[:200] or DEFAULT_ARRANGEMENT
    singer = {'zh': 'Mandarin female vocal', 'en': 'English female vocal'}.get(language, 'female vocal')
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


def choose_reference(target_bpm, refs):
    """Nearest tempo on a log scale; the stretch is the one source of timbre risk we control."""
    best = min(refs, key=lambda r: abs(math.log(target_bpm / r['true_bpm'])))
    rate = target_bpm / best['true_bpm']
    if not 1 / MAX_STRETCH <= rate <= MAX_STRETCH:
        raise ValueError('YUE2_TEMPO_OUT_OF_RANGE')
    return best, rate


def reference_plan(ref, target_text):
    """Composite score plus the exact audio edit that aligns the clip to its own score at the target tempo."""
    ref_text = (paths.REFERENCES / ref['clip'] / 'score.abc').read_text(encoding='utf-8')
    target_bpm = parse_abc(target_text).bpm
    rate = target_bpm / ref['true_bpm']
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
    # In a Chinese song Whisper hears stray English over instrumental parts ("Zither Harp" in an interlude,
    # 2026-10-21) and the line would be sung there; a mostly Chinese song keeps only lines with Chinese.
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
    edits = []
    for q in onsets:
        k = starts.index(q)
        line_i, end, first_q, has_tie = tokens[k - 1]
        if not (by_start[first_q][1] == by_start[q][1] and first_q + by_start[first_q][2] == q and not has_tie):
            raise ValueError('YUE2_TIE_MISMATCH')
        edits.append((line_i, end))
    for line_i, end in sorted(edits, reverse=True):
        lines[line_i] = lines[line_i][:end] + '-' + lines[line_i][end:]
    tied = '\n'.join(lines) + ('\n' if text.endswith('\n') else '')
    if len(parse_abc(tied).voices['Vocal'].notes) != len(notes) - len(onsets):
        raise ValueError('YUE2_TIE_COUNT')
    return tied


def build(work: Path, lyrics: str, arrangement: str):
    """Write work/plan.json and work/request.json; return the plan."""
    score = Score(work / 'score')
    asr = json.loads((work / 'asr.json').read_text(encoding='utf-8'))
    refs = load_references()
    ref, _ = choose_reference(score.score.bpm, refs)
    composite, ref_plan = reference_plan(ref, score.text)
    units = asr_units(asr)
    lines, lyric_source = lyric_lines(lyrics, asr)
    han = sum(len(HAN.findall(l)) for l in lines)
    total = sum(len(UNIT.findall(l)) for l in lines)
    language = 'zh' if han >= 0.5 * total else 'en' if (asr.get('language') == 'en') else 'other'
    lyric, times, reliable = place_lines(lines, units, score)
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
            'target_bpm': score.score.bpm, 'language': language, 'lyric_source': lyric_source,
            'lines': len(lines), 'ties': len(ties), 'sections': target_labels}
    (work / 'request.json').write_text(json.dumps(request, ensure_ascii=False, indent=1), encoding='utf-8')
    (work / 'plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=1), encoding='utf-8')
    return plan
