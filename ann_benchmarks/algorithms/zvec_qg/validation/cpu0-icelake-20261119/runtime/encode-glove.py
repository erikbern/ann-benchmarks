"""Prepare a standalone block4/rank8 M64 snapshot and router for a new GloVe base."""
import json
import os
import sys
from pathlib import Path

source = Path(os.environ['BASE']).resolve(strict=True)
snapshot = Path(os.environ['SNAP']).resolve()
if snapshot.exists() or Path(str(snapshot) + '.router').exists():
    raise FileExistsError(snapshot)
sys.path.insert(0, os.environ['MODULE_DIR'])
import _direct_core
profile = Path(__file__).resolve().parents[1] / 'glove/profile.json'
storage = json.loads(profile.read_text())['storage']
storage.update(local_qg_file=str(snapshot), build_snapshot=True)
index = _direct_core.Index(str(source), json.dumps(storage), 100)
index.close()
