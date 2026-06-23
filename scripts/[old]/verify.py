import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

this_case = "CALSNIC2_CAL_C003"

# Load one predictions
img = nib.load(fr"Q:\anat2vessels\predictions\{this_case}.nii.gz")
data = img.get_fdata()

print(data.shape)
print(np.unique(data))

# Visual inspection of the middle slice
slice_idx = data.shape[2] // 2

plt.imshow(data[:, :, slice_idx], cmap="gray")
plt.title("Predicted vessels")
plt.axis("off")
plt.show()

# # Overlay prediciton on t1
# t1 = nib.load(fr"Q:\anat2vessels\data\t1w\{this_case}_T1w10_V1.nii.gz").get_fdata()
# plt.imshow(t1[:, :, slice_idx], cmap="gray")
# plt.imshow(data[:, :, slice_idx], alpha=0.4)
# plt.title("Overlay check")
# plt.show()

print("Done")