import os
from pathlib import Path

BASE_DIR = r"Q:\anat2vessels"

# External data
T1_SOURCE = r"Q:\Notes+Files\job\S26 WorkLearn\Data\CALSNIC 2\Nifti"
T2_SOURCE = r"Q:\Notes+Files\job\S26 WorkLearn\Data\T2w10\Nifti"

# Internal structure
DATA_DIR = rf"{BASE_DIR}\data"
T1_DIR = rf"{DATA_DIR}\t1w"
T2_DIR = rf"{DATA_DIR}\t2w"

TEMP_DIR = rf"{BASE_DIR}\temp"
TEMP_T1_DIR = rf"{TEMP_DIR}\t1w"
TEMP_T2_DIR = rf"{TEMP_DIR}\t2w"

PREPROCESSED_DIR = rf"{BASE_DIR}\preprocessed"
PREDICTIONS_DIR = rf"{BASE_DIR}\predictions"
FEATURES_DIR = rf"{BASE_DIR}\features"
MODEL_DIR = rf"{BASE_DIR}\model"

REPO_DIR = rf"{BASE_DIR}\repo\anat2vessels\inference"

DONE_FILE = rf"{PREPROCESSED_DIR}\subjects_done.json"

# executables
NNUNET_PREDICT_EXE = r"Q:\conda_envs\anat_inference\Scripts\nnUNetv2_predict.exe"
PREPROCESS_ENV = r"Q:\conda_envs\anat_preprocess\python.exe"
INFERENCE_ENV = r"Q:\conda_envs\anat_inference\python.exe"


def setup_env():
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

    os.environ["nnUNet_results"] = r"Q:\nnUNet_results"
    os.environ["nnUNet_raw"] = r"Q:\nnUNet_raw"
    os.environ["nnUNet_preprocessed"] = r"Q:\nnUNet_preprocessed"
    os.environ["nnUNet_temp"] = r"Q:\nnUNet_temp"

    os.environ["TMP"] = TEMP_DIR
    os.environ["TEMP"] = TEMP_DIR
    os.environ["TMPDIR"] = TEMP_DIR

    os.environ["XDG_CACHE_HOME"] = r"Q:\cache"
    os.environ["KERAS_HOME"] = r"Q:\cache\keras"