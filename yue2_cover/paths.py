"""Where the models, the tokenizer adapters, the singer's reference library and the tools live.

Every location can be set with an environment variable; the defaults follow the layout in the README.
"""
import os
import sys
from pathlib import Path

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

YUE2 = MODELS / 'YuE2-3B'
VAE = MODELS / 'YuE2-Vae'
SHEETSAGE = MODELS / 'SheetSage2'
MERT = MODELS / 'MERT-v2-FullSong'
TOKENIZER_SCRIPTS = TOKENIZER / 'scripts'
HEAD = TOKENIZER / 'tokenizer_head_joint_v9.safetensors'
NAR = TOKENIZER / 'nar_lora_joint_v9.safetensors'
