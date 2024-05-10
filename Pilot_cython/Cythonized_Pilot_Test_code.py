### Cythonized Pilot Test code ###
import numpy as np
# import Cython
from matplotlib import pyplot as plt
# import Pilot
from ...PILOT.pilot import Pilot, Tree
import time


# generate some random dataset
n_samples = 100
X = 2 * np.random.rand(n_samples, 400)
cats = np.random.choice(3,n_samples)
X = np.c_[X, cats]
y = (3 * X[:,0]**2 + X[:,8] - X[:,5]**3 + X[:,10] + np.random.normal(size = n_samples)).reshape(-1, 1)

# initialization
# model = Pilot.PILOT()
model = Pilot.PILOT()
start = time.time()
model.fit(X,y, categorical = np.array([10], dtype = np.int64))
end = time.time()
print(end - start)
pred_y = model.predict(X = X)
print(pred_y.shape[0])
print(pred_y)
mse = np.mean((pred_y.reshape(-1, 1) - y)**2)
plt.scatter(pred_y,y)
plt.savefig("C:/Workdir/Research/Code/yrc17/Pilot_cython/debug.png")
print(mse)
