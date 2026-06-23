import shutil
from pathlib import Path
from config.paths import T1_SOURCE, T2_SOURCE, T1_DIR, T2_DIR

# ============================================
# COPY FUNCTION
# ============================================

def collect_niftis(source_root, destination_root, label):

    destination_root.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped = 0

    print(f"\n=== COLLECTING {label} FILES ===\n")

    for file in source_root.rglob("*.nii.gz"):

        dest_path = destination_root / file.name

        if dest_path.exists():
            print(f"Skipping existing: {file.name}")
            skipped += 1
            continue

        print(f"Copying: {file.name}")

        shutil.copy2(file, dest_path)
        copied += 1

    print(f"\nFinished {label}")
    print(f"Copied: {copied}")
    print(f"Skipped existing: {skipped}")

def delete_metadata_files(*folders):

    for folder in folders:
        folder = Path(folder)

        deleted = 0

        for f in folder.iterdir():

            if f.is_file() and f.name.startswith("._"):
                print(f"Deleting metadata file: {f.name}")
                f.unlink()
                deleted += 1

        print(f"Deleted {deleted} metadata files in {folder.name}")
# ============================================
# MAIN
# ============================================

if __name__ == "__main__":

    collect_niftis(T1_SOURCE, T1_DIR, "T1")
    collect_niftis(T2_SOURCE, T2_DIR, "T2")

    delete_metadata_files(T1_DIR, T2_DIR)

    print("\n=== DONE ===")
