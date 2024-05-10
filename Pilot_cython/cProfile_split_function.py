import cProfile
import pstats
import split_function
import numpy as np

n_samples = 100
n_features = 5000

X = 2 * np.random.rand(100, 5000)
y = (3 * X[:,0]**2 + X[:,8] - X[:,100]**3 ).reshape(-1, 1)
X = np.c_[np.arange(0, n_samples), X]

# index and sorted_X_indices
index = 1 + np.arange(n_samples, dtype = np.int64)
sorted_indices = np.array(
        [
            np.argsort(X[:, feature_id], axis=0).flatten()
            for feature_id in range(1, n_features + 1)
        ]
    )
sorted_X_indices = X[:, 0][sorted_indices].astype(np.int64)

min_sample_leaf = 5
k_con = np.array([1], dtype = np.int64)
k_lin = np.array([2], dtype = np.int64)
k_split_nodes = np.array([5, 5, 7], dtype = np.int64)
k_pconc = np.array([5], dtype = np.int64)
categorical = np.array([n_features + 1], dtype=np.int64)
max_features_considered = n_features
min_unique_values_regression = 5

cProfile.run('split_function.best_split(index, n_features,sorted_X_indices,X,y,min_sample_leaf,k_con,k_lin,k_split_nodes,k_pconc,categorical,max_features_considered,min_unique_values_regression)', 'profiling_result')
p = pstats.Stats('profiling_result')
p.sort_stats('cumulative').print_stats(10)