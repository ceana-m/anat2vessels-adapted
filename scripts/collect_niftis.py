import os
import shutil

# ============================================
# SOURCE DIRECTORIES
# ============================================

T1_SOURCE = r"Q:\Notes+Files\job\S26 WorkLearn\Data\CALSNIC 2\Nifti"
T2_SOURCE = r"Q:\Notes+Files\job\S26 WorkLearn\Data\T2w10\Nifti"

# ============================================
# DESTINATION DIRECTORIES
# ============================================

T1_DEST = r"Q:\anat2vessels\data\t1w"
T2_DEST = r"Q:\anat2vessels\data\t2w"

# ============================================
# COPY FUNCTION
# ============================================

def collect_niftis(source_root, destination_root, label):

    os.makedirs(destination_root, exist_ok=True)

    copied = 0
    skipped = 0

    print(f"\n=== COLLECTING {label} FILES ===\n")

    # walk through ALL nested folders
    for root, dirs, files in os.walk(source_root):

        for file in files:

            if not file.endswith(".nii.gz"):
                continue

            source_path = os.path.join(root, file)
            dest_path = os.path.join(destination_root, file)

            # avoid overwriting duplicates
            if os.path.exists(dest_path):
                print(f"Skipping existing file:")
                print(file)
                skipped += 1
                continue

            print(f"Copying:")
            print(file)

            shutil.copy2(source_path, dest_path)

            copied += 1

    print(f"\nFinished {label}")
    print(f"Copied: {copied}")
    print(f"Skipped existing: {skipped}")

# ============================================
# MAIN
# ============================================

if __name__ == "__main__":

    collect_niftis(
        T1_SOURCE,
        T1_DEST,
        "T1"
    )

    collect_niftis(
        T2_SOURCE,
        T2_DEST,
        "T2"
    )

    print("\n=== DONE ===")
