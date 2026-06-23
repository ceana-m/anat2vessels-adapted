def is_valid_nifti(filename):

    return (
        filename.endswith(".nii.gz")
        and not filename.startswith("._")
    )