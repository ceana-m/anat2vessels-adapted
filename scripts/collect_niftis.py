import os
import shutil
from config.paths import T1_SOURCE, T2_SOURCE, T1_DIR, T2_DIR

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
        T1_DIR,
        "T1"
    )

    collect_niftis(
        T2_SOURCE,
        T2_DIR,
        "T2"
    )

    print("\n=== DONE ===")
