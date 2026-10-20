"""Encode one WAV/FLAC into raw YuE2 semantic tokens with the community joint-v9 real-audio tokenizer head.

MERT-v2-FullSong layer-20 features -> instance norm -> the v9 head, using the tokenizer repo's own definitions
(mert_l20, Tok, instnorm, predict), read from its scripts at a pinned revision without running those scripts.
The caller sets CUDA_VISIBLE_DEVICES. No downloads; only the output path is written.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from math import gcd
import os
from pathlib import Path
import tempfile
import time

import paths

# Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4 at revision e2e63d859f3af879baf1b4d4e9f22d1eeda6fde5.
SOURCE_HASHES = {
    'prep_real.py': '961b3d826a06f05836a0b8a435f8a4af811ac561356440545800e7597e9431cf',
    'ar_prep.py': '952c85bff9c49939564120cd0064573706f68eec0abc5219dab1ee26092db775',
    'ckpt_io.py': '44d48fc08aa7c4a2e0f3120adbe4a05c5f6e1ec54442153bd8a7bc00a24a0e9a',
}
HEAD_SHA = '06440f25605c6c12c4e517ef033d7b70b8d9e961b3b06aa3848b1f8028fd7eb8'
NAR_SHA = '585f303da1d5252d228d1e8ac6d4c4d11d970df9297406935cc8bdafa49cfa7e'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def checked_source(path):
    if sha(path) != SOURCE_HASHES[path.name]:
        raise ValueError('UPSTREAM_SOURCE_HASH_MISMATCH: ' + str(path))
    return ast.parse(path.read_text(encoding='utf-8'), filename=str(path))


def selected_definitions(path, names, constants=()):
    """Reuse source AST verbatim; never execute Runtime() or the training loops."""
    tree, selected, found = checked_source(path), [], set()
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names:
            selected.append(node)
            found.add(node.name)
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in constants:
                ast.literal_eval(node.value)
                selected.append(node)
                found.add(target.id)
    if found != set(names) | set(constants):
        raise ValueError('UPSTREAM_SYMBOLS_CHANGED: ' + str(path))
    return compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec')


def validate_codes(codes, expected_frames):
    import numpy as np
    if (codes.ndim != 1 or codes.dtype != np.int32 or len(codes) != expected_frames
            or not len(codes) or codes.min() < 0 or codes.max() >= 32768):
        raise ValueError('INVALID_RAW_INT32_CODEC_OUTPUT')


def write_json(path, value):
    partial = path.with_name(path.name + '.partial')
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(partial, path)


def encode(args):
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    started = time.perf_counter()
    if args.input.resolve() == args.output.resolve() or args.output.suffix != '.npy':
        raise ValueError('OUTPUT_MUST_BE_A_NEW_NPY_PATH')
    if sha(args.input) != args.input_sha256:
        raise ValueError('INPUT_AUDIO_SHA_MISMATCH')
    for name in SOURCE_HASHES:
        checked_source(args.scripts_dir / name)
    if sha(args.head) != HEAD_SHA or sha(args.nar) != NAR_SHA:
        raise ValueError('V9_HEAD_NAR_PAIR_HASH_MISMATCH')
    audio, rate = sf.read(args.input, dtype='float32', always_2d=True)
    if rate <= 0 or audio.shape[1] not in (1, 2) or len(audio) < rate or not np.isfinite(audio).all():
        raise ValueError('REQUIRES_FINITE_MONO_OR_STEREO_AUDIO_AT_LEAST_ONE_SECOND')
    factor = gcd(rate, 24000)
    mono24 = resample_poly(audio.mean(1), 24000 // factor, rate // factor).astype(np.float32)
    frames = int(round(len(mono24) / 24000 * 25))
    remainder = len(mono24) % (24000 * 30)
    if 0 < remainder < 24000 and len(mono24) > 24000 * 30:
        raise ValueError('UPSTREAM_MERT_WOULD_DROP_SUBSECOND_TAIL: choose a reference duration outside (30n,30n+1) seconds; no silent truncation performed')
    contract = {'input_sha256': args.input_sha256, 'input_frames': len(audio), 'input_rate': rate,
                'input_channels': audio.shape[1], 'head_sha256': HEAD_SHA, 'nar_sha256': NAR_SHA,
                'scripts_sha256': SOURCE_HASHES, 'encoder_sha256': sha(__file__),
                'mert_dir': str(args.mert), 'expected_codec_frames': frames,
                'frame_policy': 'round(resampled_mono_duration * 25), MERT-only; no VAE latent clamp',
                'vocabulary_offset_applied': False}
    receipt_path = args.output.with_suffix('.json')
    if args.output.exists() and receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        if receipt.get('contract') != contract or sha(args.output) != receipt.get('codec_sha256'):
            raise ValueError('EXISTING_OUTPUT_CONTRACT_OR_HASH_MISMATCH')
        validate_codes(np.load(args.output, allow_pickle=False), frames)
        print(json.dumps({'status': 'reused', 'codec_frames': frames, 'codec_seconds': frames / 25}), flush=True)
        return
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ.setdefault('HF_HOME', str(paths.HF_HOME))
    import torch
    import torch.nn as nn
    from transformers import AutoFeatureExtractor, AutoModel
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA_DEVICE_REQUIRED')
    torch.cuda.reset_peak_memory_stats()
    processor = AutoFeatureExtractor.from_pretrained(str(args.mert), trust_remote_code=True, local_files_only=True)
    mert = AutoModel.from_pretrained(str(args.mert), trust_remote_code=True, local_files_only=True).to('cuda').eval()
    namespace = {'torch': torch, 'np': np, 'nn': nn, 'dev': 'cuda', 'proc': processor, 'mert': mert}
    exec(selected_definitions(args.scripts_dir / 'prep_real.py', {'mert_l20'}), namespace)
    features = namespace['mert_l20'](mono24)
    if features.shape != (frames, 1024) or not np.isfinite(features).all():
        raise ValueError('INVALID_MERT_L20_FEATURES')
    del namespace['mert'], mert
    torch.cuda.empty_cache()
    exec(selected_definitions(args.scripts_dir / 'ckpt_io.py', {'_mods', 'load_ckpt'}), namespace)
    exec(selected_definitions(args.scripts_dir / 'ar_prep.py', {'Tok', 'instnorm', 'predict'},
                              {'VOCAB', 'WIN', 'D', 'L', 'H'}), namespace)
    head = namespace['Tok'](1024).to('cuda').eval()
    head.load_state_dict(namespace['load_ckpt'](str(args.head), 'cuda')['model'])
    namespace['head'] = head
    codes = namespace['predict'](namespace['instnorm'](features)).astype(np.int32)
    validate_codes(codes, frames)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_name(args.output.name + '.partial')
    with partial.open('wb') as stream:
        np.save(stream, codes, allow_pickle=False)
    os.replace(partial, args.output)
    receipt = {'status': 'complete', 'contract': contract, 'output': str(args.output),
               'codec_sha256': sha(args.output), 'codec_frames': frames, 'codec_rate_hz': 25,
               'codec_seconds': frames / 25, 'dtype': 'int32', 'minimum': int(codes.min()),
               'maximum': int(codes.max()), 'input_seconds': len(audio) / rate,
               'audio_token_duration_difference_seconds': frames / 25 - len(audio) / rate,
               'nar_companion_sha256': NAR_SHA, 'elapsed_seconds': time.perf_counter() - started,
               'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
               'human_accepted': False, 'reconstruction_tested': False}
    write_json(receipt_path, receipt)
    print(json.dumps(receipt), flush=True)


def self_test(scripts_dir):
    import numpy as np
    selected_definitions(scripts_dir / 'prep_real.py', {'mert_l20'})
    selected_definitions(scripts_dir / 'ckpt_io.py', {'_mods', 'load_ckpt'})
    selected_definitions(scripts_dir / 'ar_prep.py', {'Tok', 'instnorm', 'predict'}, {'VOCAB', 'WIN', 'D', 'L', 'H'})
    valid = np.array([0, 59, 32767], dtype=np.int32)
    validate_codes(valid, 3)
    for bad, count in [(valid.astype(np.int64), 3), (valid, 4), (np.array([-1], dtype=np.int32), 1),
                       (np.array([32768], dtype=np.int32), 1), (valid.reshape(1, 3), 3)]:
        try:
            validate_codes(bad, count)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid codec fixture accepted')
    with tempfile.TemporaryDirectory(prefix='fixture-', dir=Path(__file__).parent) as directory:
        path = Path(directory) / 'raw.npy'
        np.save(path, valid, allow_pickle=False)
        restored = np.load(path, allow_pickle=False)
        assert np.array_equal(restored, valid) and restored.dtype == np.int32
    print('SELF_TEST_PASS: frozen-source AST extraction, raw int32 codec bounds/shape/count, safe NPY roundtrip; no torch/GPU/model loading')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path)
    parser.add_argument('--input-sha256')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--scripts-dir', type=Path, default=paths.TOKENIZER_SCRIPTS)
    parser.add_argument('--mert', type=Path, default=paths.MERT)
    parser.add_argument('--head', type=Path, default=paths.HEAD)
    parser.add_argument('--nar', type=Path, default=paths.NAR)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test(args.scripts_dir)
    elif args.input is None or args.output is None or not args.input_sha256:
        parser.error('--input, --input-sha256, and --output are required')
    else:
        encode(args)
