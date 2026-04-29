import sys
from pathlib import Path


TASK_ROOT = Path(__file__).resolve().parents[2]
if str(TASK_ROOT) not in sys.path:
    sys.path.insert(0, str(TASK_ROOT))
