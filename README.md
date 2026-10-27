# yue2-cover: AI song covers in your singer's voice with YuE2

**yue2-cover** is an open-source pipeline for **AI cover songs**: it re-sings any song in **your singer's voice** over
a **newly generated arrangement**, while keeping the song's original melody, lyrics and structure. It runs locally on
one GPU (tested on an RTX 3090) and is built on [YuE2](https://github.com/multimodal-art-projection/YuE) by M-A-P,
with SheetSage2 lead-sheet transcription and Whisper lyric timing. No fine-tuning of the singer is needed: a few
30-second clips of their singing are enough (a zero-shot **voice prefix**). Tuned on Mandarin (C-pop) songs; English
songs work too.

- Keeps the melody and the lyric timing of the original song, section by section.
- Regenerates the accompaniment from a style text ("acoustic guitar ballad", ...), unlike voice-conversion (SVC/RVC)
  covers that keep the original backing track.
- One command: `python yue2_cover/job.py --source song.mp3 --output cover.wav --work work/song1`.

[中文说明](#中文说明) · [FAQ](#faq)

## How it works

```
song.mp3 ─► SheetSage2 ─► lead sheet (melody, chords, sections, beats)
        └─► Whisper ─────► word timings ─► lyric lines placed in the score's sections
                                   │
singer's clip library ─► nearest tempo ─► composite ABC score: [reference bars][song bars]
                                   │        section-tagged lyrics, held line ends tied
reference clip ─► stretch + align ─► real-audio tokenizer ─► YuE2 semantic tokens (forced prefix)
                                   │
                       YuE2 AR (CFG 1) continues the prefix over the song's score
                       YuE2 NAR (joint-v9 LoRA, 64 ODE steps) ─► audio, prefix cut off ─► cover.wav
```

YuE2 normally writes its own score and voice. Here the score is the song's own transcription, and the first bars are
a real recording of your singer, encoded into YuE2's semantic tokens and forced into the model's history. The model
then continues singing *as that voice* through the song's melody, with an arrangement it generates itself from the
style text.

## What we learned (the rules in `plan.py`)

Each rule fixed a failure we heard:

1. **One tempo for the whole composite score.** When the reference section kept its own tempo, the model followed
   the audio's tempo and slid into speech. Pick the library clip nearest the song's tempo and stretch it as little
   as possible (at most 1.18x; otherwise no reference fits).
2. **Lyrics in YuE2's native section blocks.** One `[Verse]` / `[Chorus]` block per `% label` section of the score,
   in order, with empty blocks for instrumental sections. All lyrics in one block drift after the first section.
3. **Tie held line ends only where needed.** SheetSage splits a held line-final note into two same-pitch attacks and
   YuE re-sings the last syllable. For Chinese (one character = one syllable), a line with more notes than syllables
   gets its line-end same-pitch pair tied; nothing else is changed. Whisper word times run ~0.4 s early, so notes
   are counted against syllables, not decided from timing alone.
4. **Clean Whisper text.** No previous-text conditioning (long songs loop on one line); a simplified-Chinese prompt
   on every window; credit and subtitle hallucinations ("作词", "字幕…") are dropped, and so is stray English heard
   over the instrumental parts of a Chinese song. Whole-song Whisper also skips sung lines now and then (one pop
   song lost a whole verse, so the model got 84 notes for 29 syllables there and stretched and rushed the words):
   sung stretches that no word starts within a second of are found from the score's vocal melody and heard again
   on their own (`clip_timestamps`); they are listed under `refilled` in `work/asr.json`. Passing the real lyrics
   (`--lyrics-file`) is always better.
5. **Exact reference alignment.** The clip is shifted so its first sung note lands where its score puts it, then
   padded to a 25 Hz codec-frame boundary. Reference lengths just over a multiple of 30 s are avoided (the MERT
   feature extractor drops a sub-second tail).
6. **Settings:** AR CFG 1.0, NAR 64 ODE steps, seed 42.

Typical results: every line in order and on time; the model sometimes adds an echo in a long instrumental outro or
ends it early. Lyric errors in Whisper's text are sung as they are.

## Other models

The front end (transcription, lyric sections, tempo and key, reference choice) is not tied to YuE2. We also drove
ACE-Step 1.5 XL with it (cover task, the song's sections, tempo and key, the singer's clip as reference audio, with and
without a singer LoRA). It runs end to end and is about three times faster, but in our listening test the voice and the
arrangement were clearly worse than YuE2's, so this repository ships the YuE2 back end only.

## Requirements

- Linux, an NVIDIA GPU with 12 GB or more (peak 8.5 GB in our runs), ~20 GB of disk for the models.
- **YuE2 environment** (Python 3.12): `yue2-infer` 0.1.6 from the [YuE repository](https://github.com/multimodal-art-projection/YuE)
  (release `yue2-v0.1.6`), torch 2.10 (CUDA 12.8), transformers 4.57, `openai-whisper` ≥ 20250625, numpy, scipy,
  soundfile, safetensors.
- **SheetSage2 environment** (we use Python 3.10, torch 2.8, transformers 4.45; it can be the same environment if
  SheetSage2's remote code runs there).
- ffmpeg 5 or newer (tested with 6.1).

## Setup

```bash
export YUE2_COVER_HOME=~/yue2-cover-data           # every path below can also be set on its own, see yue2_cover/paths.py
for m in YuE2-3B YuE2-Vae SheetSage2 MERT-v2-FullSong; do
  huggingface-cli download m-a-p/$m --local-dir $YUE2_COVER_HOME/models/$m
done
huggingface-cli download Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4 \
  --revision e2e63d859f3af879baf1b4d4e9f22d1eeda6fde5 --local-dir $YUE2_COVER_HOME/realaudio-tokenizer \
  scripts/prep_real.py scripts/ar_prep.py scripts/ckpt_io.py \
  tokenizer_head_joint_v9.safetensors nar_lora_joint_v9.safetensors
export YUE2_COVER_PYTHON=/path/to/yue2-env/bin/python
export YUE2_COVER_SAGE_PYTHON=/path/to/sheetsage-env/bin/python
export YUE2_COVER_WHISPER=large-v3-turbo           # or a checkpoint path
```

The model steps run offline (`HF_HUB_OFFLINE=1`), so download first; the only step that uses the network is the optional
[lyrics lookup](#run). The tokenizer scripts are checked against pinned hashes and only their model definitions are
used; they are never run.

### Your singer's reference library

```bash
python tools/add_reference.py --clip clip-01.flac --lyrics-file clip-01.txt --id singer-ballad-01
```

Use 25-40 s clips of the singer's own recordings (vocals over accompaniment, like a released song), starting just
before a phrase. Add clips at several tempos (for example 60, 64, 85 and 121 BPM): a song is covered only if some
clip is within 1.18x of its tempo. The tool transcribes the clip, keeps one section, measures its tempo from the
beats and its first sung note from the vocal melody, and prints the numbers; `--true-bpm` and
`--first-note-seconds` override them. Use only voices you have the right to use.

## Run

```bash
python yue2_cover/job.py --source song.mp3 --output cover.wav --work work/song1 \
  --lyrics-file song1-lyrics.txt --arrangement "acoustic guitar ballad"
```

- `--lyrics-file`: one sung line per line. Without it, Whisper's text is sung, corrected from the song's published
  lyrics when the lookup below finds them.
- `--arrangement`: genre and arrangement words; the default is an emotional C-pop ballad.
- `--asr-prompt ''` for songs that are not in Chinese.
- `--no-lyrics-lookup`: do not look the published lyrics up (same as `export YUE2_COVER_LYRICS_LOOKUP=0`).

**Published-lyrics lookup (optional, uses the network).** Whisper's text can carry wrong or missing characters, and a
missing syllable leaves a note empty. With no `--lyrics-file`, the job searches the three longest heard lines as lyrics
on NetEase Cloud Music's public web endpoints (at most 3 searches and 8 lyric downloads) and keeps a candidate only if
at least half of the heard lines have a close published line. The heard lines are placed in time first; each one (or
each pair that is one published line) then takes the published words inside its own time span. If the published lyrics
are timed to this very recording (at least four matched lines, 60% of them starting within a second of the same
offset), a published line no heard line matched is also sung at its own time, on the melody's notes there, in place of
whatever was heard over it; the count is `timed_lines` in `receipt.json`. Another recording of the song (live, a
cover) is not timed like the lyrics and is left as heard. It runs only when no lyrics are given, and any failure (no
match, offline, a timeout, a changed endpoint) keeps the heard lyrics. The song found is recorded as `lyrics_song` in
`receipt.json`; its lines are saved as `work/lyrics-lookup.json`. These endpoints are not an official API. You are
responsible for complying with that service's terms of use and with the copyright of the lyrics. Turn it off with `--no-lyrics-lookup` or `YUE2_COVER_LYRICS_LOOKUP=0`.

A 228 s song took 230 s on one RTX 3090: score 21 s, ASR 20 s, plan 4 s, encode 10 s, generate 174 s.
`work/receipt.json` records the reference clip, tempo, sections, ties and timings.

**If a cover stops right after the voice prompt** (the job then fails with `YUE2_TOO_SHORT`): the model sometimes
ends the song at the boundary. It is sensitive to the reference alignment: in our tests the same song and clip
failed with the measured clip tempo 60.609 BPM and sang the whole song with 60.683 BPM. Re-measure the clip tempo
(`--true-bpm`), try another clip, or give the lyrics. Use a new `--work` folder for each attempt.

Tests (CPU only, no models needed; `pip install pytest`): `python -m pytest -q`;
`python yue2_cover/encode_impl.py --self-test` checks the tokenizer scripts against the pinned hashes.

## FAQ

**How do I make an AI cover of a song in a specific singer's voice with YuE2?**
Build a reference library of the singer with `tools/add_reference.py` (a few 25-40 s clips at different tempos),
then run `yue2_cover/job.py` on the song. The pipeline transcribes the song, picks the clip nearest in tempo, encodes
it into YuE2 semantic tokens and lets YuE2 continue singing in that voice over the song's own score.

**How is this different from RVC / so-vits-svc / SoulX-SVC covers?**
Voice conversion keeps the original recording and swaps the timbre of the separated vocal. yue2-cover generates a
new performance and a new arrangement, so there is no source-separation residue and the arrangement can change
style. The trade-off: it is slower (about real time on a 3090) and the lyrics must be right.

**Does it need a LoRA or fine-tuning?**
No. The voice comes from a forced prefix of the singer's own recording. YuE2's weights stay unchanged.

**Does it keep the original melody?**
Yes. The score YuE2 sings from is SheetSage2's transcription of the song itself, with one tempo for the whole score,
lyrics placed in the native section blocks and held line ends tied (see "What we learned").

**What hardware does it need?**
An NVIDIA GPU with 12 GB or more; a 228 s song took 230 s on an RTX 3090 with an 8.5 GB peak.

**Which languages?**
The rules are tuned for Mandarin Chinese; English songs work with `--asr-prompt ''` (no note ties are added).

**Can I use it commercially?**
The code is Apache 2.0, but the model weights are not; see [Licenses](#licenses).

## Licenses

- This repository's code: Apache License 2.0.
- **Model weights are not Apache.** YuE2-3B and YuE2-Vae: CC BY-NC 4.0 with M-A-P's additional permission for
  individual creators; companies need a separate license from M-A-P (see the
  [YuE license section](https://github.com/multimodal-art-projection/YuE#license)).
  SheetSage2 and MERT-v2 (M-A-P) and the real-audio tokenizer head and NAR LoRA (Mothersuperior): CC BY-NC 4.0.
- `yue2-infer`: Apache License 2.0. openai-whisper: MIT.
- Covering a song does not license the song. Keep covers for personal listening unless you hold the rights.

## Credits

- [YuE2](https://huggingface.co/m-a-p/YuE2-3B), [SheetSage2](https://huggingface.co/m-a-p/SheetSage2) and
  [MERT](https://huggingface.co/m-a-p/MERT-v2-FullSong) by M-A-P (Multimodal Art Projection) and collaborators.
- [Real-audio tokenizer head and NAR LoRA](https://huggingface.co/Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4)
  by Mothersuperior.
- Forced-history sampling follows the approach of the audio.cpp YuE2 implementation (commit f1c299b).
- [Whisper](https://github.com/openai/whisper) by OpenAI.

---

## 中文说明

**yue2-cover：基于 YuE2 的开源 AI 翻唱（AI 歌声翻唱、AI cover）流程。** 用 **你的歌手的声音** 重新演唱一首歌，
伴奏由模型 **重新编曲**，旋律、歌词和结构保持原曲。基于 M-A-P 的 YuE2，配合 SheetSage2 扒谱和 Whisper 歌词对时，
单张显卡本地运行。不需要 LoRA 或微调，只需要几段 30 秒左右的演唱片段（零样本声音前缀）。针对中文流行歌调过，英文歌也能用。

前半段（扒谱、歌词分段、速度和调性、选参考片段）不绑定 YuE2：我们也用它驱动过 ACE-Step 1.5 XL，能完整跑通、速度快三倍左右，
但试听下来声音和编曲都明显不如 YuE2，所以仓库里只放 YuE2 的后端。

和 RVC、so-vits-svc、SoulX-SVC 这类换声翻唱不同：换声保留原伴奏、只换人声音色；yue2-cover 重新生成演唱和伴奏，
没有分轨残留，编曲风格也可以改。

**原理**：SheetSage2 把原曲扒成谱（旋律、和弦、段落、节拍），Whisper 给出逐字时间，把歌词分配到各段。
从歌手片段库里选速度最接近的一段，拉伸对齐后编码成 YuE2 的语义 token，作为强制前缀放在生成序列最前面；
后面接原曲的乐谱，YuE2 就用这个声音把整首歌唱下去，伴奏按风格描述自己生成。最后切掉前缀部分。

**我们踩过的坑**（都写进了 `plan.py`）：

1. 整个合成乐谱只能有一个速度。参考段保留自己的速度时，模型会跟着音频速度走，最后变成念白。所以选速度最接近的片段，尽量少拉伸（最多 1.18 倍）。
2. 歌词按乐谱的段落分成 YuE2 原生的 `[Verse]` / `[Chorus]` 块，纯器乐段留空块。全部歌词放在一块里，第一段之后就会跑偏。
3. SheetSage 会把句尾的长音拆成两个同音高的音，YuE 会把最后一个字唱两遍。中文一字一音，只在音符比字多的句子里把句尾同音高的两个音连起来，其他不动。
4. Whisper 不用前文条件（长歌会循环同一句），每个窗口都带简体中文提示；去掉“作词”“字幕”之类的幻觉，以及中文歌间奏里误识别出的英文。整首识别时 Whisper 偶尔还会漏掉唱出来的句子（有首歌漏了整段主歌，模型在那里拿到 84 个音符却只有 29 个字，字被拉长、抢拍），所以会用乐谱里的人声旋律找出一秒内没有任何字开始的唱段，单独再听一遍（`clip_timestamps`），重听的区间记在 `asr.json` 的 `refilled` 里。能给真实歌词就给。
5. 参考片段按第一个唱出的音精确对齐，再补到 25 Hz 编码帧边界；避开略长于 30 秒整数倍的长度。
6. 参数：AR CFG 1.0，NAR 64 步，seed 42。

**歌词查询（可选，会联网）**：没有给 `--lyrics-file` 时，Whisper 听出的歌词可能有错字、漏字，漏一个字就会空出一个音符。
此时会把听到的最长三句当作歌词，在网易云音乐的公开网页接口上搜索（每首歌最多搜 3 次、取 8 份歌词），只有至少一半的听写句在候选歌词里
有相近的一句时才采用。先把听到的句子按时间放好，再让每一句（或合起来是同一句的两句）在自己的时间段内换成已发布的歌词。
若歌词的时间与这版录音对得上（至少 4 句匹配，其中 60% 的起点与歌词时间相差同一个偏移量不超过 1 秒），
没被听到的整句会按歌词给的时间、落在旋律该处的音符上补唱，并替换那段时间里听到的句子，句数记在 `receipt.json` 的 `timed_lines`；
对不上的版本（现场版、翻唱）保持听到的样子。只在没给歌词时才查，
任何失败（没搜到、断网、超时、接口变了）都保留听到的歌词；查到的歌曲记在 `receipt.json` 的 `lyrics_song`，歌词行存在 `work/lyrics-lookup.json`。
这些接口不是官方 API，使用者需自行遵守该服务的使用条款和歌词版权。关闭：`--no-lyrics-lookup` 或 `YUE2_COVER_LYRICS_LOOKUP=0`。

**用法**见上面的 Setup / Run。228 秒的歌在一张 RTX 3090 上用时 230 秒，显存峰值 8.5 GB。

**许可**：本仓库代码 Apache 2.0。YuE2 权重是 CC BY-NC 4.0 加 M-A-P 给个人创作者的额外许可，
公司使用需向 M-A-P 另行取得授权；SheetSage2、MERT2 和社区的音频编码器权重是 CC BY-NC 4.0。翻唱不等于获得歌曲本身的授权，请仅供个人欣赏。请只使用你有权使用的声音。
