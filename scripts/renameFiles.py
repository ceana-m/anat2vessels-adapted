import os

t1_dir = r"Q:\anat2vessels\data\t1w"
t2_dir = r"Q:\anat2vessels\data\t2w"

def rename(folder, modality):
    for f in os.listdir(folder):
        if not f.endswith(".nii.gz"):
            continue

        parts = f.replace(".nii.gz","").split("_")

        # assumes structure: CALSNIC2_CAL_C003_T1w10_V1
        subject = "_".join(parts[:3])
        visit = parts[-1]

        new_name = f"{subject}_{visit}_{modality}.nii.gz"

        os.rename(
            os.path.join(folder, f),
            os.path.join(folder, new_name)
        )

rename(t1_dir, "T1")
rename(t2_dir, "T2")