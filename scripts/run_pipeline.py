import os
import subprocess
from pathlib import Path
import re
import json
import torch
from utils import is_valid_nifti
import time
import shutil

# =========================
# PATH CONFIGURATION
# =========================

BASE_DIR = r"Q:\anat2vessels"

T1_DIR = rf"{BASE_DIR}\data\t1w"
T2_DIR = rf"{BASE_DIR}\data\t2w"

TEMP_T1_DIR = rf"{BASE_DIR}\temp\t1w"
TEMP_T2_DIR = rf"{BASE_DIR}\temp\t2w"

PREPROCESSED_DIR = rf"{BASE_DIR}\preprocessed"
PREDICTIONS_DIR = rf"{BASE_DIR}\predictions"
FEATURES_DIR = rf"{BASE_DIR}\features"

MODEL_DIR = rf"{BASE_DIR}\model"

REPO_DIR = rf"{BASE_DIR}\repo\anat2vessels\inference"

PREPROCESS_ENV = r"Q:\conda_envs\anat_preprocess\python.exe"
INFERENCE_ENV = r"Q:\conda_envs\anat_inference\python.exe"

DONE_FILE = os.path.join(PREPROCESSED_DIR, "subjects_done.json")

os.environ["nnUNet_results"] = r"Q:\nnUNet_results"
os.environ["nnUNet_raw"] = r"Q:\nnUNet_raw"
os.environ["nnUNet_preprocessed"] = r"Q:\nnUNet_preprocessed"
os.environ["nnUNet_temp"] = r"Q:\nnUNet_temp"

os.environ["TMP"] = rf"{BASE_DIR}\temp"
os.environ["TEMP"] = rf"{BASE_DIR}\temp"
os.environ["TMPDIR"] = rf"{BASE_DIR}\temp"

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["XDG_CACHE_HOME"] = r"Q:\cache"
os.environ["KERAS_HOME"] = r"Q:\cache\keras"

# to execute shell commands
def run_command(command_list):
    print("\n===================================")
    print("RUNNING COMMAND:")
    print(" ".join(command_list))
    print("===================================\n")

    result = subprocess.run(
        command_list,
        text=True
    )

    print(result.stdout)
    print(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    
# must rename CALSNIC files otherwise delimeter will be unable to differentiate
def normalize_filenames(folder, modality):

    for f in os.listdir(folder):

        if not is_valid_nifti(f):
            continue

        # -----------------------------------
        # ALREADY NORMALIZED
        # -----------------------------------

        normalized_pattern = (
            rf".+_V\d+"
            rf"(?:_run\d+)?"
            rf"_{modality}\.nii\.gz"
        )

        if re.fullmatch(normalized_pattern, f):
            print(f"Already normalized: {f}")
            continue

        # -----------------------------------
        # REMOVE EXTENSION
        # -----------------------------------

        base = f.replace(".nii.gz", "")

        parts = base.split("_")

        subject_parts = []
        visit = None
        run = None

        for p in parts:

            # skip modality markers
            if "T1w" in p or "T2w" in p:
                continue

            # visit
            elif re.fullmatch(r"V\d+", p):
                visit = p

            # run variants
            elif "run" in p.lower():

                # normalize:
                # run1 -> run1
                # run_1 -> run1

                digits = re.findall(r"\d+", p)

                if len(digits) > 0:
                    run = f"run{digits[0]}"
                else:
                    run = "run"

            else:
                subject_parts.append(p)

        # -----------------------------------
        # VALIDATION
        # -----------------------------------

        if visit is None:
            print(f"Could not identify visit: {f}")
            continue

        subject = "_".join(subject_parts)

        # -----------------------------------
        # BUILD NEW NAME
        # -----------------------------------

        new_name = f"{subject}_{visit}"

        if run is not None:
            new_name += f"_{run}"

        new_name += f"_{modality}.nii.gz"

        old_path = os.path.join(folder, f)
        new_path = os.path.join(folder, new_name)

        # avoid overwriting
        if os.path.exists(new_path):
            print(f"Target already exists, skipping:")
            print(new_name)
            continue

        print(f"\nRenaming:")
        print(f"{f}")
        print("→")
        print(f"{new_name}\n")

        os.rename(old_path, new_path)

# keep track of already preprocessed files
def load_done():
    if os.path.exists(DONE_FILE):
        with open(DONE_FILE, "r") as f:
            return json.load(f)
    return {}


def save_done(done):
    with open(DONE_FILE, "w") as f:
        json.dump(done, f, indent=2)

def preprocessing_output_exists(case_id):

    files = os.listdir(PREPROCESSED_DIR)
    case_tokens = case_id.split("_")

    for f in files:
        if "_0000" not in f:
            continue

        file_tokens = f.replace(".nii.gz", "").split("_")

        # require ordered prefix match (stronger than subset)
        if file_tokens[:len(case_tokens)] == case_tokens:
            return True

    return False

# perform preprocessing including skull stripping
def preprocess_subject(case_id, t1_file, t2_file):
    done = load_done()

    if preprocessing_output_exists(case_id):
        print(f"Skipping (already exists): {case_id}")
        done[case_id] = True
        save_done(done)
        return

    print(f"\nProcessing: {case_id}")

    # Create temp folders
    os.makedirs(TEMP_T1_DIR, exist_ok=True)
    os.makedirs(TEMP_T2_DIR, exist_ok=True)

    # Remove old temp files
    for f in os.listdir(TEMP_T1_DIR):
        os.remove(os.path.join(TEMP_T1_DIR, f))

    for f in os.listdir(TEMP_T2_DIR):
        os.remove(os.path.join(TEMP_T2_DIR, f))

    # Copy current T1
    shutil.copy2(
        os.path.join(T1_DIR, t1_file),
        os.path.join(TEMP_T1_DIR, t1_file)
    )

    # Copy current T2
    shutil.copy2(
        t2_file,
        os.path.join(TEMP_T2_DIR, os.path.basename(t2_file))
    )

    # ----------------------------
    # BUILD COMMAND
    # ----------------------------
    cmd = [
        PREPROCESS_ENV,
        "preprocess_imgs.py",
        "--t1_dir", TEMP_T1_DIR,
        "--t2_dir", TEMP_T2_DIR,
        "--output_dir", PREPROCESSED_DIR,
        "--id_delim", "_T1",
        "--skull_strip", "True"
    ]

    os.chdir(REPO_DIR)

    print("\nRUNNING FROM:", os.getcwd())
    print("SCRIPT EXISTS:", os.path.exists(os.path.join(REPO_DIR, "preprocess_imgs.py")))
    print("PYTHON:", PREPROCESS_ENV)

    print("Launching preprocess_imgs.py")
    result = subprocess.run(
        cmd,
        cwd=REPO_DIR
    )

    print("\n===== PREPROCESS DEBUG =====")
    print("RETURN CODE:", result.returncode)

    print("\n--- STDOUT ---")
    print(result.stdout if result.stdout else "[EMPTY]")

    print("\n--- STDERR ---")
    print(result.stderr if result.stderr else "[EMPTY]")
    # print(result.stdout)
    # print(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(f"Preprocessing failed for {case_id}")

    # ----------------------------
    # MARK DONE ONLY IF SUCCESSFUL
    # ----------------------------

    print(f"Preprocessing finished for: {case_id}")

    time.sleep(1)

    if preprocessing_output_exists(case_id):
        done[case_id] = True
        save_done(done)
        print(f"Confirmed output exists: {case_id}")
    else:
        print(f"WARNING: subprocess finished but output not found: {case_id}")

    # create json for metadata of preprocessed cases
    # meta = {
    #     "T1": f"{case_id}_0000.nii.gz",
    #     "T2": f"{case_id}_0001.nii.gz"
    # }

    # with open(os.path.join(PREPROCESSED_DIR, f"{case_id}.json"), "w") as f:
    #     json.dump(meta, f, indent=2)

    

def get_unprocessed_cases():

    pending = []

    for f in os.listdir(PREPROCESSED_DIR):

        if not f.endswith("_0000.nii.gz"):
            continue

        case_id = f.replace("_0000.nii.gz", "")

        prediction_file = os.path.join(
            PREDICTIONS_DIR,
            f"{case_id}.nii.gz"
        )

        if os.path.exists(prediction_file):
            print(f"Skipping inference (already exists): {case_id}")
            continue

        pending.append(case_id)

    return pending

def inference():

    pending_cases = get_unprocessed_cases()

    if len(pending_cases) == 0:
        print("No new cases for inference.")
        return

    print(f"\nRunning inference on all cases, including {len(pending_cases)} new cases")

    temp_dir = os.path.join(BASE_DIR, "temp_predict")

    # Clear old temp directory
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

    os.makedirs(temp_dir)

    # Copy only pending cases
    for case_id in pending_cases:

        case_files = [
            f for f in os.listdir(PREPROCESSED_DIR)
            if f.startswith(f"{case_id}_") and f.endswith(".nii.gz")
        ]

        for f in case_files:
            shutil.copy(
                os.path.join(PREPROCESSED_DIR, f),
                os.path.join(temp_dir, f)
            )

    print(f"\nRunning inference on {len(pending_cases)} new cases")

    cmd = [
        r"Q:\conda_envs\anat_inference\Scripts\nnUNetv2_predict.exe",
        "-d", "096",
        # "-i", PREPROCESSED_DIR,
        "-i", temp_dir,
        "-o", PREDICTIONS_DIR,
        "-f", "0", "1", "2", "3", "4",
        "-c", "3d_fullres",
        "-tr", "nnUNetTrainer",
        "-p", "nnUNetResEncUNetLPlans",
        # "--disable_tta",
        "-device", "cuda",
        "-npp", "4",
        "-nps", "2"
    ]

    try:
        run_command(cmd)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

def extract_features():

    cmd = [
        INFERENCE_ENV,
        "csv_from_predictions.py",
        "--input_dir", PREDICTIONS_DIR,
        "--output_path", os.path.join(FEATURES_DIR, "features.csv")
    ]

    os.chdir(REPO_DIR)

    run_command(cmd)

if __name__ == "__main__":
    # # GPU checks
    # # print(torch.cuda.is_available())
    # # print(torch.cuda.device_count())
    # # print(torch.cuda.get_device_name(0))

    # print("\n=== NORMALIZING FILENAMES ===")
    # normalize_filenames(T1_DIR, "T1")
    # normalize_filenames(T2_DIR, "T2")

    print("\n=== PREPROCESSING (RESUMABLE) ===")

    done = load_done()

    i = 0
    for t1_file in os.listdir(T1_DIR):

        if not t1_file.endswith(".nii.gz"):
            continue

        # ----------------------------
        # PARSE SUBJECT + VISIT
        # ----------------------------
        base = t1_file.replace(".nii.gz", "")

        # remove modality token explicitly 
        base = base.replace("_T1", "").replace("_T2", "").replace("_T1w", "").replace("_T2w", "")

        parts = base.split("_")

        subject_parts = []
        visit = None

        for p in parts:
            if re.fullmatch(r"V\d+", p):
                visit = p
            else:
                subject_parts.append(p)

        subject = "_".join(subject_parts)

        if visit is None:
            print(f"Skipping (no visit found): {t1_file}")
            continue

        case_id = f"{subject}_{visit}"

        if preprocessing_output_exists(case_id):
            print(f"Skipping (already exists): {case_id}")
            continue
        
        # ----------------------------
        # FIND MATCHING T2
        # ----------------------------
        t2_path = None

        expected_prefix = f"{subject}_{visit}"

        for f in os.listdir(T2_DIR):

            if not f.endswith(".nii.gz"):
                continue

            # enforce same case_id structure as T1-derived logic
            if not f.startswith(expected_prefix):
                continue

            # ensure modality match (robust against naming variation)
            if "_T2" not in f and "T2w" not in f:
                continue

            t2_path = os.path.join(T2_DIR, f)
            break

        if t2_path is None:
            print(f"Missing T2 for: {case_id}")
            continue

        # ----------------------------
        # RUN PREPROCESSING (RESUMABLE)
        # ----------------------------
        # if i == 2:
        #     continue
        # i += 1
        # preprocess_subject(case_id, t1_file, t2_path)

    print("\n=== RUNNING INFERENCE ===")
    inference()

    # print("\n=== EXTRACTING FEATURES ===")
    # extract_features()

    print("\n=== PIPELINE COMPLETE ===")