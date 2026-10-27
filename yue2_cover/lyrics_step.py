"""Find the song's published lyrics from what Whisper heard (optional; the only step that uses the network).

    python lyrics_step.py <asr.json> <lyrics-lookup.json>

Whisper's lyrics carry wrong and missing characters. A misheard word can leave a syllable out, so a note stays empty
and words go missing in the cover. A few of the longest heard lines are searched as lyrics on NetEase Cloud Music's
public web endpoints (cloudsearch type 1006 finds a song from its words, even misheard ones); each candidate's timed
lyrics are compared with everything heard, and the closest one is kept when enough of the heard lines have a close
published line. plan.py then puts its words into the lines that were actually sung. Nothing found, the service slow,
gone or changed: the heard lyrics stay as they are.

It runs only when no lyrics are given (job.py), at most 3 searches and 8 lyric downloads per cover, and can be turned
off with `--no-lyrics-lookup` or YUE2_COVER_LYRICS_LOOKUP=0. You are responsible for complying with that service's
terms of use and with the copyright of the lyrics; the published lines are kept in the job's work folder only.
"""
from __future__ import annotations

import difflib
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

from plan import NOISE, UNIT as WORD, plain

SEARCH = 'https://music.163.com/api/cloudsearch/pc'
LYRIC = 'https://music.163.com/api/song/lyric?id={}&lv=1'
HEADERS = {'Referer': 'https://music.163.com/', 'User-Agent': 'Mozilla/5.0',
           'Content-Type': 'application/x-www-form-urlencoded'}
TIMEOUT = 6
BUDGET = 40            # seconds for the whole lookup
QUERIES = 3
CANDIDATES = 8
ACCEPT = 0.5           # share of the heard lines that must have a close published line
CLOSE = 0.5            # a heard line this similar to a published one is the same line
CREDIT = re.compile(r'^\s*[^：:]{1,12}[：:]|【.*】|^\s*\(.*\)\s*$|^\s*（.*）\s*$')
STAMP = re.compile(r'\[(\d{1,3}):(\d{1,2}(?:\.\d{1,3})?)\]')


def heard_lines(asr):
    """The sung lines Whisper heard that are long enough to search for (credits and subtitle lines left out)."""
    lines = []
    for segment in asr.get('segments', []):
        text = segment.get('text', '')
        if any(x in text for x in NOISE):
            continue
        lines.extend(p.strip() for p in re.split(r'[，。,.!?！？\s]+', text) if len(WORD.findall(p)) >= 4)
    return lines


def parse_lrc(text):
    """[(seconds, line)] in time order; credit lines, blank lines and lines without words left out."""
    out = []
    for raw in text.splitlines():
        stamps = STAMP.findall(raw)
        words = STAMP.sub('', raw).strip()
        if not stamps or not WORD.search(words) or CREDIT.search(words):
            continue
        out.extend((int(m) * 60 + float(s), words) for m, s in stamps)
    return sorted(out)


def post(url, data, deadline):
    body = urllib.parse.urlencode(data).encode()
    timeout = max(1.0, min(TIMEOUT, deadline - time.monotonic()))
    with urllib.request.urlopen(urllib.request.Request(url, data=body, headers=HEADERS), timeout=timeout) as response:
        return json.load(response)


def get(url, deadline):
    timeout = max(1.0, min(TIMEOUT, deadline - time.monotonic()))
    with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=timeout) as response:
        return json.load(response)


def overlap(heard, published):
    """Share of the heard lines with a close published line. Line by line: one diff over the whole text lines up
    repeated choruses badly (0.41 for the right song where 40 of 45 lines matched)."""
    texts = [plain(line) for line in published if plain(line)]
    if not heard or not texts:
        return 0.0
    close = 0
    for line in heard:
        a = plain(line)
        if a and max(difflib.SequenceMatcher(None, a, b, autojunk=False).ratio() for b in texts) >= CLOSE:
            close += 1
    return close / len(heard)


def lookup(asr):
    deadline = time.monotonic() + BUDGET
    lines = heard_lines(asr)
    if not lines:
        return {'song': None, 'why': 'nothing heard'}
    queries = sorted(dict.fromkeys(lines), key=lambda line: -len(WORD.findall(line)))[:QUERIES]
    candidates = {}
    for query in queries:
        if time.monotonic() >= deadline:
            break
        found = post(SEARCH, {'s': query, 'type': 1006, 'limit': 5, 'offset': 0}, deadline)
        for song in (found.get('result') or {}).get('songs') or []:
            candidates.setdefault(song['id'], {'id': song['id'], 'name': song.get('name', ''),
                                               'artist': ','.join(a.get('name', '') for a in song.get('ar') or [])})
        if len(candidates) >= CANDIDATES:
            break
    best = None
    for song in list(candidates.values())[:CANDIDATES]:
        if time.monotonic() >= deadline:
            break
        lrc = parse_lrc(((get(LYRIC.format(song['id']), deadline).get('lrc') or {}).get('lyric')) or '')
        if not lrc:
            continue
        score = overlap(lines, [line for _, line in lrc])
        if best is None or score > best[0]:
            best = (score, song, lrc)
    if best is None or best[0] < ACCEPT:
        return {'song': None, 'why': 'no close lyrics', 'best_score': round(best[0], 3) if best else None,
                'candidates': len(candidates)}
    score, song, lrc = best
    return {'song': song, 'score': round(score, 3), 'lines': [{'t': round(t, 2), 'text': text} for t, text in lrc]}


def main():
    asr, out = Path(sys.argv[1]), Path(sys.argv[2])
    try:
        found = lookup(json.loads(asr.read_text(encoding='utf-8')))
    except Exception as exc:  # noqa: BLE001 - the heard lyrics stay; the cover goes on
        found = {'song': None, 'why': type(exc).__name__}
    out.write_text(json.dumps(found, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: v for k, v in found.items() if k != 'lines'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
