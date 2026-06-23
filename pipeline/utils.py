from pathlib import Path

def is_valid_nifti(filename):
    name = Path(filename).name
    return name.endswith(".nii.gz") and not name.startswith("._")