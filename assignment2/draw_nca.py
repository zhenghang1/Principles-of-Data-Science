import numpy as np

log = np.load('log/nca/log.npy')
target = np.unique(log[0])

idx = []
for i in list(target):
    idx.append(list(np.where(log[0]==i))[0])

assert len(idx) == target.shape[0]

# accuracy = 