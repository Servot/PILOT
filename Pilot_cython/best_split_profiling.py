### Cythonized Pilot Profiling: Big O time###
import numpy as np
import split_function
from tqdm import tqdm
import time
import pandas as pd

# generate data
def generate_data(n_sample, n_feature):
    X = []
    y = np.random.randn(n_sample)
    cat_feat = []
    for feature in range(n_feature):
        if np.random.choice(['cat', 'num'], p=[0, 1]) == 'cat':
            x = np.random.choice([0, 1, 2], size=n_sample)
            if np.random.choice(['p_conc', 'nothing']) == 'p_conc':
                y = y + np.where(x == 0, 1, np.where(x == 1, -1, 0))
            cat_feat.append(feature)
        else:
            kind = np.random.choice(['pcon', 'plin', 'blin', 'lin', 'nothing'])
            x = np.random.randn(n_sample) * np.random.randint(1, 10) + np.random.randint(1, 10)
            split = np.random.choice(x)
            if kind == 'pcon':
                y = y + np.where(x < split, 1, -1)
            elif kind == 'lin':
                y = y + 0.1 + x * 0.1
            elif kind == 'plin':
                y = y + np.where(x < split, 0.1 - 0.1 * x, 0.1 + 0.1 * x)
            elif kind == 'blin':
                crossing_y = 0.1 - 0.1 * split
                second_intercept = crossing_y - 0.1 * split
                y = y + np.where(x < split, 0.1 - 0.1 * x, second_intercept + 0.1 * x)
        X.append(x)        
    return np.array(X).T, y, np.array(cat_feat, dtype=np.int64)

n_samples = np.logspace(2, 4, num=2, dtype=np.int64)
n_features = np.logspace(0, 1, num=2, dtype=np.int64) + 1

results = []
for n_sample in tqdm(n_samples):
    for n_feature in tqdm(n_features):
        for random_seed in range(5):
            np.random.seed(random_seed)
            X, y, cat_feat = generate_data(n_sample, n_feature)

            print(X)
            print(X.shape)
            # preprocessing for using the split function
            X = np.c_[np.arange(0, n_sample), X]

            time1 = time.time()
            # index and sorted_X_indices
            index = 1 + np.arange(n_sample, dtype = np.int64)
            sorted_indices = np.array(
                    [
                        np.argsort(X[:, feature_id], axis=0).flatten()
                        for feature_id in range(1, n_feature + 1)
                    ]
                )
            sorted_X_indices = X[:, 0][sorted_indices].astype(np.int64)

            time2 = time.time()

            # run split function
            result = split_function.best_split(index, # NEW: regression_nodes deleted
            n_feature,
            sorted_X_indices,
            X,
            y.reshape(-1,1),
            min_sample_leaf = 5,
            k_con = np.array([1], dtype = np.int64),
            k_lin = np.array([2], dtype = np.int64),
            k_split_nodes = np.array([5, 5, 7], dtype = np.int64),
            k_pconc = np.array([5], dtype = np.int64),
            categorical = np.array([n_feature + 1], dtype=np.int64),
            max_features_considered = n_feature,
            min_unique_values_regression = 5)
            end = time.time()
            results.append(dict(n_samples=n_sample, n_features=n_feature,
                                time_elapsed_sort=time2 - time1,
                                time_elapsed_split=end - time2))
pd.DataFrame(results).to_csv('C:/Workdir/Research/Code/yrc17/Pilot_cython/split_function_computation_time_simulation.csv', index=False)
