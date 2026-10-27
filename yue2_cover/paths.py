"""Where the models, the tokenizer adapters, the singer's reference library and the tools live.

Every location can be set with an environment variable; the defaults follow the layout in the README.
The one step that uses the network, the published-lyrics lookup, is on by default and has a switch (LYRICS_LOOKUP).
"""
import os
import sys
from pathlib import Path


def flag(name, default=True):
    """An on/off switch from the environment: unset or empty keeps `default`; "0", "false", "no" and "off" turn it off."""
    value = os.environ.get(name, '').strip().lower()
    return default if not value else value not in ('0', 'false', 'no', 'off')


HOME = Path(os.environ.get('YUE2_COVER_HOME', Path.home() / 'yue2-cover-data'))
# m-a-p/YuE2-3B, m-a-p/YuE2-Vae, m-a-p/SheetSage2 and m-a-p/MERT-v2-FullSong, one folder each.
MODELS = Path(os.environ.get('YUE2_COVER_MODELS', HOME / 'models'))
# A snapshot of Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4 (scripts/ plus the joint v9 head and NAR).
TOKENIZER = Path(os.environ.get('YUE2_COVER_TOKENIZER', HOME / 'realaudio-tokenizer'))
# The singer: refs.json, one folder per clip with its score.abc, and the clips themselves.
REFERENCES = Path(os.environ.get('YUE2_COVER_REFERENCES', HOME / 'references'))
# An openai-whisper model name or checkpoint path.
WHISPER = os.environ.get('YUE2_COVER_WHISPER', 'large-v3-turbo')
# The environment with the yue2 package, torch and openai-whisper.
PYTHON = os.environ.get('YUE2_COVER_PYTHON', sys.executable)
# SheetSage2 may need its own transformers version; by default the same environment.
SAGE_PYTHON = os.environ.get('YUE2_COVER_SAGE_PYTHON', PYTHON)
FFMPEG = os.environ.get('YUE2_COVER_FFMPEG', 'ffmpeg')
HF_HOME = Path(os.environ.get('HF_HOME', HOME / 'hf-cache'))
# With no lyrics given, look the song's published lyrics up on NetEase Cloud Music (needs the network; any failure
# leaves the heard lyrics). YUE2_COVER_LYRICS_LOOKUP=0 turns it off; job.py --no-lyrics-lookup does so for one run.
LYRICS_LOOKUP = flag('YUE2_COVER_LYRICS_LOOKUP')

YUE2 = MODELS / 'YuE2-3B'
VAE = MODELS / 'YuE2-Vae'
SHEETSAGE = MODELS / 'SheetSage2'
MERT = MODELS / 'MERT-v2-FullSong'
TOKENIZER_SCRIPTS = TOKENIZER / 'scripts'
HEAD = TOKENIZER / 'tokenizer_head_joint_v9.safetensors'
NAR = TOKENIZER / 'nar_lora_joint_v9.safetensors'
