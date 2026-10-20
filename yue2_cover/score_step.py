"""SheetSage2 full-song transcription: lead-sheet ABC, beats, downbeats, vocal melody."""
import json
import sys
from pathlib import Path

import paths


def main():
    source, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    from transformers import AutoModel
    model = AutoModel.from_pretrained(paths.SHEETSAGE, trust_remote_code=True, base_model_path=str(paths.MERT),
                                      local_files_only=True).eval().to('cuda')
    result = model.transcribe(str(source), output_dir=str(out), melody_only=False)
    if not result.get('abc') or result.get('abc_error'):
        raise SystemExit('SCORE_INVALID')
    (out / 'target-score.abc').write_text(result['abc'], encoding='utf-8')
    print(json.dumps({'duration_seconds': result.get('duration_seconds'), 'warnings': result.get('warnings', [])}))


if __name__ == '__main__':
    main()
