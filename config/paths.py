import os
from pathlib import Path

# ==========================================================
# Base project location
# ==========================================================

BASE_DIR = Path(r"Q:\anat2vessels")

# ==========================================================
# External data
# ==========================================================

T1_SOURCE = Path(r"Q:\Notes+Files\job\S26 WorkLearn\Data\CALSNIC 2\Nifti")
T2_SOURCE = Path(r"Q:\Notes+Files\job\S26 WorkLearn\Data\T2w10\Nifti")

# ==========================================================
# Internal structure
# ==========================================================

DATA_DIR = BASE_DIR / "data"

T1_DIR = DATA_DIR / "t1w"
T2_DIR = DATA_DIR / "t2w"

TEMP_DIR = BASE_DIR / "temp"
TEMP_T1_DIR = TEMP_DIR / "t1w"
TEMP_T2_DIR = TEMP_DIR / "t2w"

PREPROCESSED_DIR = BASE_DIR / "preprocessed"
PREDICTIONS_DIR = BASE_DIR / "predictions"
FEATURES_DIR = BASE_DIR / "features"
MODEL_DIR = BASE_DIR / "model"
CACHE_DIR = BASE_DIR / "cache"

CORE_DIR = BASE_DIR / "core" / "inference"

DONE_FILE = PREPROCESSED_DIR / "subjects_done.json"

# ==========================================================
# Executables
# ==========================================================

NNUNET_PREDICT_EXE = Path(
    r"Q:\conda_envs\anat_inference\Scripts\nnUNetv2_predict.exe"
)

PREPROCESS_ENV = Path(
    r"Q:\conda_envs\anat_preprocess\python.exe"
)

INFERENCE_ENV = Path(
    r"Q:\conda_envs\anat_inference\python.exe"
)

# ==========================================================
# Environment setup
# ==========================================================

def setup_env():

    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

    os.environ["nnUNet_results"] = str(MODEL_DIR)
    os.environ["nnUNet_raw"]          = str(TEMP_DIR / "nnunet_raw")
    os.environ["nnUNet_preprocessed"] = str(TEMP_DIR / "nnunet_preprocessed")

    os.environ["TMP"] = str(TEMP_DIR)
    os.environ["TEMP"] = str(TEMP_DIR)
    os.environ["TMPDIR"] = str(TEMP_DIR)

    os.environ["XDG_CACHE_HOME"] = str(CACHE_DIR)
    os.environ["KERAS_HOME"] = str(CACHE_DIR / "keras")