from utils import is_valid_nifti
from pathlib import Path
from config.paths import T1_SOURCE, T2_SOURCE, T1_DIR, T2_DIR

# ============================================
# GET ALL NIFTI FILENAMES
# ============================================

def get_source_files(root_dir):

    files_set = set()

    for f in root_dir.rglob("*"):

        if f.is_file() and is_valid_nifti(f.name):
            files_set.add(f.name)

    return files_set


def get_dest_files(root_dir):

    files_set = set()

    for f in root_dir.iterdir():

        if f.is_file() and is_valid_nifti(f.name):
            files_set.add(f.name)

    return files_set

# ============================================
# VERIFY
# ============================================

def verify(source_dir, dest_dir, label):

    source_files = get_source_files(source_dir)
    dest_files = get_dest_files(dest_dir)

    missing = source_files - dest_files
    extra = dest_files - source_files

    print(f"\n========== {label} ==========")
    print(f"Source count: {len(source_files)}")
    print(f"Destination count: {len(dest_files)}")

    if not missing and not extra:
        print("\nSUCCESS: All files copied correctly.")
        return

    if missing:
        print(f"\nMissing files ({len(missing)}):")
        for f in sorted(missing):
            print(f)

    if extra:
        print(f"\nExtra files ({len(extra)}):")
        for f in sorted(extra):
            print(f)

# ============================================
# MAIN
# ============================================

if __name__ == "__main__":

    verify(T1_SOURCE, T1_DIR, "T1")
    verify(T2_SOURCE, T2_DIR, "T2")
