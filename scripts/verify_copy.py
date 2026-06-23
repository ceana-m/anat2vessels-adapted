import os
from utils import is_valid_nifti
from config.paths import T1_SOURCE, T2_SOURCE, T1_DIR, T2_DIR

# ============================================
# GET ALL NIFTI FILENAMES
# ============================================

def get_source_files(root_dir):

    files_set = set()

    for root, dirs, files in os.walk(root_dir):

        for f in files:

            if is_valid_nifti(f):
                files_set.add(f)

    return files_set


def get_dest_files(root_dir):

    files_set = set()

    for f in os.listdir(root_dir):

        if is_valid_nifti(f):
            files_set.add(f)

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

    # ----------------------------------------
    # PERFECT MATCH
    # ----------------------------------------

    if len(missing) == 0 and len(extra) == 0:
        print("\nSUCCESS: All files copied correctly.")
        return

    # ----------------------------------------
    # REPORT ISSUES
    # ----------------------------------------

    if len(missing) > 0:

        print(f"\nMissing files ({len(missing)}):")

        for f in sorted(missing):
            print(f)

    if len(extra) > 0:

        print(f"\nExtra files ({len(extra)}):")

        for f in sorted(extra):
            print(f)

# ============================================
# MAIN
# ============================================

if __name__ == "__main__":

    verify(
        T1_SOURCE,
        T1_DIR,
        "T1"
    )

    verify(
        T2_SOURCE,
        T2_DIR,
        "T2"
    )
