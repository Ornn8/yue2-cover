"""Join a transcribed reference and an unchanged full-song event timeline.

One native ABC tempo is used. Reference audio must be time-stretched without
pitch shifting and aligned separately before its newly encoded tokens are used.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

from abc_tools import AbcError, DURATIONS, TOKEN, VOICES, parse_abc


def require(condition, message):
    if not condition:
        raise AbcError(message)


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def scale_music_token(match, factor):
    if match.group("note") is None or factor == 1:
        return match.group(0)
    units = int(match.group("duration") or "1") * factor
    chunks = []
    while units:
        chunk = max(n for n in DURATIONS if n <= units)
        chunks.append(chunk)
        units -= chunk
    pitch = (match.group("acc") or "") + match.group("note") + match.group("oct")
    rest = match.group("note") == "z"
    return "".join(pitch + (str(n) if n != 1 else "") +
                   ("-" if not rest and (i < len(chunks) - 1 or match.group("tie")) else "")
                   for i, n in enumerate(chunks))


def normalize(text, unit, bpm):
    original = parse_abc(text)
    factor = original.unit / unit
    require(factor.denominator == 1, "Common unit must represent all original durations")
    lines = text.splitlines()
    lines[3] = f"L:1/{unit.denominator}"
    lines[4] = f"Q:1/4={bpm}"
    for index in original.music_lines:
        lines[index] = TOKEN.sub(lambda match: scale_music_token(match, factor.numerator), lines[index])
    output = "\n".join(lines) + "\n"
    normalized = parse_abc(output)
    for name in VOICES:
        before, after = original.voices[name], normalized.voices[name]
        require(before.notes == after.notes and before.bars == after.bars and
                before.chords == after.chords and before.keys == after.keys,
                f"Unit normalization changed {name} events")
    return output


def sections(score):
    """Section labels and exact quarter-note positions; comments are not lyrics."""
    positions = dict.fromkeys(VOICES, Fraction(0))
    bar_indices = dict.fromkeys(VOICES, 0)
    events = []
    for index, line in enumerate(score.text.splitlines()):
        if line.startswith("% "):
            require(positions["Vocal"] == positions["Ins"], "Section occurs between unsynchronized voices")
            events.append((positions["Vocal"], line))
        if index in score.music_lines:
            voice = score.music_lines[index]
            count = sum(int(bar[1:] or "1") if bar.startswith("Z") else 1
                        for bar in (part.strip() for part in line[:-1].split("|")))
            bar_indices[voice] += count
            last = score.voices[voice].bars[bar_indices[voice] - 1]
            positions[voice] = last[0] + last[1]
    return events


def effective_keys(events):
    """Collapse redundant declarations, retaining every effective key change."""
    by_time = {}
    for onset, key in events:
        by_time[onset] = key
    result = []
    for onset, key in sorted(by_time.items()):
        if not result or result[-1][1] != key:
            result.append((onset, key))
    return result


def target_body_with_boundary(target_text):
    lines = target_text.splitlines()
    boundary = {"M": lines[2], "K": lines[7]}
    seen, output = set(), []
    for index in range(8, len(lines)):
        line = lines[index]
        output.append(line)
        if line in ("V: Vocal", "V: Ins") and line not in seen:
            seen.add(line)
            next_index, fields = index + 1, set()
            while next_index < len(lines) and lines[next_index].startswith(("M:", "K:")):
                fields.add(lines[next_index].split(":", 1)[0])
                next_index += 1
            for name in ("M", "K"):
                if name not in fields:
                    output.append(boundary[name])
    require(len(seen) == 2, "Target does not begin with both native voices")
    return "\n".join(output) + "\n"


def verify_target(target, composite, offset):
    checks = {}
    for name in VOICES:
        expected, actual = target.voices[name], composite.voices[name]
        require(not any(t < offset < t + d for t, _, d in actual.notes),
                f"A {name} note crosses the reference/target boundary")
        notes = [[t - offset, p, d] for t, p, d in actual.notes if t >= offset]
        chords = [(t - offset, c) for t, c in actual.chords if t >= offset]
        bars = [(t - offset, d, meter) for t, d, meter in actual.bars if t >= offset]
        keys = [(t - offset, k) for t, k in actual.keys if t >= offset]
        checks[name] = {"notes_exact": notes == expected.notes,
                        "chords_exact": chords == expected.chords,
                        "bars_exact": bars == expected.bars,
                        "effective_keys_exact": effective_keys(keys) == effective_keys(expected.keys),
                        "note_count": len(notes), "bar_count": len(bars), "chord_count": len(chords)}
        require(all(checks[name][k] for k in ("notes_exact", "chords_exact", "bars_exact", "effective_keys_exact")),
                f"Target {name} relative events changed")
    expected_sections = sections(target)
    actual_sections = [(t - offset, label) for t, label in sections(composite) if t >= offset]
    require(actual_sections == expected_sections, "Target section positions/labels changed")
    require(composite.bpm == target.bpm, "Target tempo changed")
    require(composite.voices["Vocal"].time - offset == target.voices["Vocal"].time,
            "Target duration changed")
    return {"pass": True, "scope": "symbolic target events only; not generated-audio adherence",
            "tempo_exact": True, "sections_exact": True, "duration_exact": True, "voices": checks}


def compose(reference_text, target_text, reference_audio_seconds=None):
    reference, target = parse_abc(reference_text), parse_abc(target_text)
    unit = min(reference.unit, target.unit)
    normalized_ref = normalize(reference_text, unit, target.bpm)
    normalized_target = normalize(target_text, unit, target.bpm)
    original_quarters = reference.voices["Vocal"].time
    # The smallest native length grid that also lands exactly on a 25 Hz frame.
    quarter_unit = unit * 4
    frame_unit = quarter_unit * 1500 / target.bpm
    boundary_step = quarter_unit * frame_unit.denominator
    boundary_quarters = boundary_step * math.ceil(original_quarters / boundary_step)
    while Fraction(0) < (boundary_quarters * 60 / target.bpm) % 30 < 1:
        boundary_quarters += boundary_step
    padding_quarters = boundary_quarters - original_quarters
    if padding_quarters:
        meter = padding_quarters / 4
        normalized_ref += ("% interlude\n"
                           f"V: Vocal\nM:{meter.numerator}/{meter.denominator}\nZ|\n"
                           f"V: Ins\nM:{meter.numerator}/{meter.denominator}\nZ|\n")
    parse_abc(normalized_ref)
    joined = normalized_ref + target_body_with_boundary(normalized_target)
    composite = parse_abc(joined)
    invariant = verify_target(target, composite, boundary_quarters)
    duration = boundary_quarters * 60 / target.bpm
    frames = duration * 25
    require(frames.denominator == 1, "Reference boundary does not align to a codec frame")
    target_seconds = target.voices["Vocal"].time * 60 / target.bpm
    original_score_seconds = original_quarters * 60 / reference.bpm
    rate = Fraction(target.bpm, reference.bpm)
    timing = {
        "reference_original_bpm": reference.bpm, "target_and_composite_bpm": target.bpm,
        "common_unit": str(unit), "reference_original_quarters": str(original_quarters),
        "composite_abc_utf8_bytes": len(joined.encode("utf-8")),
        "reference_original_score_seconds": float(original_score_seconds),
        "reference_atempo_rate": float(rate), "reference_atempo_rate_fraction": str(rate),
        "reference_padding_quarters": str(padding_quarters),
        "reference_padding_seconds": float(padding_quarters * 60 / target.bpm),
        "reference_boundary_step_quarters": str(boundary_step),
        "reference_boundary_quarters": str(boundary_quarters),
        "reference_normalized_seconds": float(duration), "reference_codec_frames": frames.numerator,
        "reference_samples_48khz": int(duration * 48000),
        "target_seconds": float(target_seconds), "composite_seconds": float(duration + target_seconds),
        "composite_nominal_codec_frames": int((duration + target_seconds) * 25),
        "target_first_vocal_note_seconds": float(target.voices["Vocal"].notes[0][0] * 60 / target.bpm),
        "audio_alignment_verified": False,
        "alignment_instruction": "Align original audio to reference score t=0 using transcription beat/padding evidence; stretch at the reported atempo rate, preserve pitch, then add only documented silence to reach the exact sample count. Retokenize and require the exact codec frame count. Do not silently trim sung notes.",
    }
    if reference_audio_seconds is not None:
        require(math.isfinite(reference_audio_seconds) and reference_audio_seconds > 0, "Invalid reference audio duration")
        stretched = Fraction(str(reference_audio_seconds)) / rate
        timing.update({"input_reference_audio_seconds": reference_audio_seconds,
                       "audio_seconds_after_tempo_only": float(stretched),
                       "score_minus_audio_seconds_before_stretch": float(original_score_seconds - Fraction(str(reference_audio_seconds))),
                       "normalized_score_minus_tempo_only_audio_seconds": float(duration - stretched)})
    return joined, normalized_ref, normalized_target, invariant, timing


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--reference", required=True, type=Path)
    cli.add_argument("--target", required=True, type=Path)
    cli.add_argument("--output", required=True, type=Path)
    cli.add_argument("--reference-audio-seconds", type=float)
    args = cli.parse_args()
    require(not args.output.exists(), "Use a new output directory")
    reference_bytes, target_bytes = args.reference.read_bytes(), args.target.read_bytes()
    joined, ref, target, invariant, timing = compose(reference_bytes.decode("utf-8"), target_bytes.decode("utf-8"),
                                                    args.reference_audio_seconds)
    provenance = {"reference_path": str(args.reference.resolve()), "reference_sha256": sha_bytes(reference_bytes),
                  "target_path": str(args.target.resolve()), "target_sha256": sha_bytes(target_bytes),
                  "script_sha256": sha_bytes(Path(__file__).read_bytes()),
                  "parser_sha256": sha_bytes(Path(__file__).with_name("abc_tools.py").read_bytes()),
                  "composite_sha256": sha_bytes(joined.encode("utf-8"))}
    args.output.mkdir(parents=True)
    for name, text in (("composite.abc", joined), ("reference-normalized.abc", ref), ("target-normalized.abc", target)):
        (args.output / name).write_bytes(text.encode("utf-8"))
    for name, value in (("invariants.json", invariant), ("timing-plan.json", timing), ("provenance.json", provenance)):
        (args.output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "invariants_pass": invariant["pass"], **timing}))


if __name__ == "__main__":
    main()
