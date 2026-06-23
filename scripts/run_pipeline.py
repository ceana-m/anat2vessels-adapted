import subprocess
from pathlib import Path
import re
import json
import torch
from utils import is_valid_nifti
import time
import shutil

from config.paths import *
from config.paths import setup_env

setup_env()

# to execute shell commands
def run_command(command_list):
    command_list = [str(x) for x in command_list]

    print("\n===================================")
    print("RUNNING COMMAND:")
    print(" ".join(command_list))
    print("===================================\n")

    result = subprocess.run(command_list, text=True)

    print(result.stdout)
    print(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    
# must rename CALSNIC files otherwise delimeter will be unable to differentiate
def normalize_filenames(folder, modality):

    for f in folder.iterdir():

        if not is_valid_nifti(f):
            continue

        name = f.name
        # -----------------------------------
        # ALREADY NORMALIZED
        # -----------------------------------

        normalized_pattern = (
            rf".+_V\d+"
            rf"(?:_run\d+)?"
            rf"_{modality}\.nii\.gz"
        )

        if re.fullmatch(normalized_pattern, name):
            print(f"Already normalized: {name}")
            continue

        # -----------------------------------
        # REMOVE EXTENSION
        # -----------------------------------

        base = name.replace(".nii.gz", "")

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

        old_path = f
        new_path = folder / new_name

        # avoid overwriting
        if new_path.exists():
            print(f"Target already exists, skipping:")
            print(new_name)
            continue

        print(f"\nRenaming:")
        print(f"{f}")
        print("→")
        print(f"{new_name}\n")

        old_path.rename(new_path)

# keep track of already preprocessed files
def load_done():
    if DONE_FILE.exists():
        with open(DONE_FILE, "r") as f:
            return json.load(f)
    return {}

def save_done(done):
    with open(DONE_FILE, "w") as f:
        json.dump(done, f, indent=2)

def preprocessing_output_exists(case_id):

    case_tokens = case_id.split("_")

    for f in PREPROCESSED_DIR.iterdir():

        if "_0000" not in f.name:
            continue

        file_tokens = f.stem.replace(".nii.gz", "").split("_")

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

    # ----------------------------
    # TEMP DIRS (Path only internally)
    # ----------------------------
    TEMP_T1_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_T2_DIR.mkdir(parents=True, exist_ok=True)

    # Clear old temp files
    for f in TEMP_T1_DIR.iterdir():
        if f.is_file():
            f.unlink()

    for f in TEMP_T2_DIR.iterdir():
        if f.is_file():
            f.unlink()

    # ----------------------------
    # COPY FILES (Path-safe)
    # ----------------------------
    shutil.copy2(
        str(T1_DIR / t1_file),
        str(TEMP_T1_DIR / t1_file)
    )

    shutil.copy2(
        str(Path(t2_file)),
        str(TEMP_T2_DIR / Path(t2_file).name)
    )

    # ----------------------------
    # SUBPROCESS COMMAND (boundary = strings)
    # ----------------------------
    cmd = [
        str(PREPROCESS_ENV),
        "preprocess_imgs.py",
        "--t1_dir", str(TEMP_T1_DIR),
        "--t2_dir", str(TEMP_T2_DIR),
        "--output_dir", str(PREPROCESSED_DIR),
        "--id_delim", "_T1",
        "--skull_strip", "True"
    ]

    print("\nRUNNING FROM:", REPO_DIR)
    print("SCRIPT EXISTS:", (REPO_DIR / "preprocess_imgs.py").exists())
    print("PYTHON:", PREPROCESS_ENV)

    result = subprocess.run(cmd, cwd=REPO_DIR, text=True)

    print("\n===== PREPROCESS DEBUG =====")
    print("RETURN CODE:", result.returncode)
    print("\n--- STDOUT ---")
    print(result.stdout or "[EMPTY]")
    print("\n--- STDERR ---")
    print(result.stderr or "[EMPTY]")

    if result.returncode != 0:
        raise RuntimeError(f"Preprocessing failed for {case_id}")

    print(f"Preprocessing finished for: {case_id}")

    time.sleep(1)

    if preprocessing_output_exists(case_id):
        done[case_id] = True
        save_done(done)
        print(f"Confirmed output exists: {case_id}")
    else:
        print(f"WARNING: subprocess finished but output not found: {case_id}")

def get_unprocessed_cases():

    pending = []

    for f in PREPROCESSED_DIR.iterdir():

        if not f.name.endswith("_0000.nii.gz"):
            continue

        case_id = f.name.replace("_0000.nii.gz", "")

        prediction_file = PREDICTIONS_DIR / f"{case_id}.nii.gz"

        if prediction_file.exists():
            print(f"Skipping inference (already exists): {case_id}")
            continue

        pending.append(case_id)

    return pending

def inference():

    pending_cases = get_unprocessed_cases()

    if len(pending_cases) == 0:
        print("No new cases for inference.")
        return

    print(f"\nRunning inference on {len(pending_cases)} cases")

    # ----------------------------
    # TEMP DIR (Path only)
    # ----------------------------
    temp_dir = TEMP_DIR / "predict"

    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    temp_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # COPY FILES
    # ----------------------------
    for case_id in pending_cases:

        case_files = [
            f for f in PREPROCESSED_DIR.iterdir()
            if f.name.startswith(f"{case_id}_") and f.name.endswith(".nii.gz")
        ]

        for f in case_files:
            shutil.copy2(str(f), str(temp_dir / f.name))

    # print(f"\nRunning inference on {len(pending_cases)} cases")

    # ----------------------------
    # SUBPROCESS COMMAND
    # ----------------------------
    cmd = [
        str(NNUNET_PREDICT_EXE),
        "-d", "096",
        "-i", str(temp_dir),
        "-o", str(PREDICTIONS_DIR),
        "-f", "0", "1", "2", "3", "4",
        "-c", "3d_fullres",
        "-tr", "nnUNetTrainer",
        "-p", "nnUNetResEncUNetLPlans",
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
        str(INFERENCE_ENV),
        "csv_from_predictions.py",
        "--input_dir", PREDICTIONS_DIR,
        "--output_path", str(FEATURES_DIR / "features.csv")
    ]

    run_command(cmd)

if __name__ == "__main__":
    # # GPU checks
    # # print(torch.cuda.is_available())
    # # print(torch.cuda.device_count())
    # # print(torch.cuda.get_device_name(0))

    # print("\n=== NORMALIZING FILENAMES ===")
    normalize_filenames(T1_DIR, "T1")
    normalize_filenames(T2_DIR, "T2")

    print("\n=== PREPROCESSING (RESUMABLE) ===")

    done = load_done()

    i = 0
    for t1_file in T1_DIR.iterdir():

        if not t1_file.name.endswith(".nii.gz"):
            continue

        # ----------------------------
        # PARSE SUBJECT + VISIT
        # ----------------------------
        base = t1_file.name.replace(".nii.gz", "")

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

        for f in T2_DIR.iterdir():

            if not f.name.endswith(".nii.gz"):
                continue

            # enforce same case_id structure as T1-derived logic
            if not f.name.startswith(expected_prefix):
                continue

            # ensure modality match (robust against naming variation)
            if "_T2" not in f.name and "T2w" not in f.name:
                continue

            t2_path = f
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