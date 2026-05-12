from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "raw"
OUTPUT_DIR = ROOT_DIR / "outputs"

TARGET_COLUMNS = ["IC50, mM", "CC50, mM", "SI"]
SUBMISSION_TARGET_COLUMNS = ["IC50", "CC50", "SI"]
INDEX_COLUMN = "index"
RANDOM_STATE = 42

