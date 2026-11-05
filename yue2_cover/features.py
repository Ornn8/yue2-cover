"""Mix measures of a song from its audio alone (CPU, no separation), for the reference pick.

    python features.py <audio> <out.json>        (needs numpy, scipy, soundfile, librosa)

The reference library's clips are measured with this same function (tools/add_reference.py stores the result in
<references>/features.json), so a song and a clip compare on one scale. Mono at 22.05 kHz, 46 ms frames.
drums: percussive share of the spectrogram energy (median-filter HPSS); a cover's backing drums follow its reference
  clip's (in our measurements r = 0.66-0.75 between the two), so the pick matches the clip's value to the song's.
hf4k_db / centroid_hz: brightness (share of energy above 4 kHz; spectral centroid), on frames within 50 dB of the loudest.
onsets_per_s: attack density. lra: loudness range (dB, 3 s windows). low150_db: share below 150 Hz.
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

RATE, N_FFT, HOP = 22050, 1024, 256


def load(path):
    import soundfile as sf
    from scipy.signal import resample_poly
    audio, rate = sf.read(str(path), dtype='float32', always_2d=True)
    mono = audio.mean(1)
    if rate != RATE:
        g = math.gcd(int(rate), RATE)
        mono = resample_poly(mono, RATE // g, int(rate) // g).astype(np.float32)
    return mono


def kweighted_levels(y, win, hop):
    """BS.1770 block loudness (K-weighting at RATE), LUFS per window."""
    from scipy.signal import lfilter
    G, Q, fc = 3.99984385397, 0.7071752369554193, 1681.9744509555319
    A, w0 = 10 ** (G / 40), 2 * math.pi * fc / RATE
    al, c = math.sin(w0) / (2 * Q), math.cos(w0)
    b = [A * ((A + 1) + (A - 1) * c + 2 * math.sqrt(A) * al), -2 * A * ((A - 1) + (A + 1) * c),
         A * ((A + 1) + (A - 1) * c - 2 * math.sqrt(A) * al)]
    a = [(A + 1) - (A - 1) * c + 2 * math.sqrt(A) * al, 2 * ((A - 1) - (A + 1) * c),
         (A + 1) - (A - 1) * c - 2 * math.sqrt(A) * al]
    z = lfilter(b, a, y)
    Q, fc = 0.5003270373253953, 38.13547087613982
    w0 = 2 * math.pi * fc / RATE
    al, c = math.sin(w0) / (2 * Q), math.cos(w0)
    z = lfilter([(1 + c) / 2, -(1 + c), (1 + c) / 2], [1 + al, -2 * c, 1 - al], z) ** 2
    n, h = int(win * RATE), int(hop * RATE)
    if len(z) < n:
        return np.array([-70.0])
    cs = np.concatenate([[0.0], np.cumsum(z, dtype=np.float64)])
    s = np.arange(0, len(z) - n + 1, h)
    return -0.691 + 10 * np.log10(np.maximum((cs[s + n] - cs[s]) / n, 1e-12))


def loudness_range(y):
    s = kweighted_levels(y, 3.0, 0.1)
    s = s[s > -70]
    if len(s) < 3:
        return 0.0
    s = s[s > 10 * np.log10(np.mean(10 ** (s / 10))) - 20]
    return float(np.percentile(s, 95) - np.percentile(s, 10))


def measure(path):
    import librosa
    y = load(path)
    M = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))
    S = M ** 2
    f = librosa.fft_frequencies(sr=RATE, n_fft=N_FFT)
    e = S.sum(0) + 1e-12
    act = 10 * np.log10(e) > 10 * np.log10(e.max()) - 50
    Sa, Ma = S[:, act], M[:, act]
    H, P = librosa.decompose.hpss(M)
    cent = (f[:, None] * Ma).sum(0) / (Ma.sum(0) + 1e-12)
    onsets = librosa.onset.onset_detect(y=y, sr=RATE, hop_length=HOP, units='time')
    seconds = len(y) / RATE
    return {'seconds': round(seconds, 2),
            'drums': round(float((P ** 2).sum() / ((H ** 2).sum() + (P ** 2).sum())), 4),
            'hf4k_db': round(float(10 * np.log10(Sa[f >= 4000].sum() / Sa.sum())), 2),
            'centroid_hz': round(float(np.median(cent))),
            'low150_db': round(float(10 * np.log10(Sa[f < 150].sum() / Sa.sum())), 2),
            'onsets_per_s': round(len(onsets) / max(seconds, 1e-6), 2),
            'lra': round(loudness_range(y), 2)}


def main():
    began = time.monotonic()
    out = measure(sys.argv[1])
    out['measure_seconds'] = round(time.monotonic() - began, 1)
    Path(sys.argv[2]).write_text(json.dumps(out), encoding='utf-8')
    print(json.dumps(out))


if __name__ == '__main__':
    main()
