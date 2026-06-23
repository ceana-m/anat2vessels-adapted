import numpy as np

r = np.load(r"Q:\anat2vessels\features\radii\CALSNIC2_CAL_C003_V1_radii.npy")

print(type(r))
print(r.shape)
print(r[:20])   # first 20 values

print("min:", np.min(r))
print("max:", np.max(r))
print("mean:", np.mean(r))
print("num values:", len(r))
