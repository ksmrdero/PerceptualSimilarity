import numpy as np

for i in range(1, 801):
    x = np.array([1])
    np.save('/data/div2k/lpips_train/judge/%04d.npy'%(i), x)
