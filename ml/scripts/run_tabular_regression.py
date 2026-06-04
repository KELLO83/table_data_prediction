import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.src.experiments.tabular_runner import main
from ml.src.models.candidates import EXPERIMENT_MODEL_NAMES

MODEL_CHOICES = list(EXPERIMENT_MODEL_NAMES)


if __name__ == "__main__":
    main()
