import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Import our visualization functions
from capstone_project.utility.acquisition_functions import upper_confidence_bound, expected_improvement


X_train = np.load("../initial_data/function_7/initial_inputs.npy")
y_train = np.load("../initial_data/function_7/initial_outputs.npy")

new_data_point_X_1 = [0.021549, 0.388382, 0.305172, 0.153496, 0.372780, 0.741816]
new_data_point_y_1 = [1.7026965064815247]

new_data_point_X_2 = 		[0.016760, 0.897058, 0.954288, 0.297366, 0.726390, 0.872772]
new_data_point_y_2 = [0.014529060505633472]

new_data_point_X_3 = 		[0.016760, 0.897058, 0.954288, 0.297366, 0.726390, 0.872772]
new_data_point_y_3 = [0.014529060505633472]

new_data_point_X_4 = 		[0.035527, 0.721632, 0.116963, 0.574678, 0.035012, 0.094444]
new_data_point_y_4 = [0.2454624804253238]

new_data_point_X_5 = 		[0.351216, 0.223271, 0.232050, 0.006079, 0.065846, 0.952998]
new_data_point_y_5 = [0.2570889203883812]

new_data_point_X_6 = 		[0.507467, 0.917329, 0.961358, 0.026947, 0.844939, 0.372116]
new_data_point_y_6 = [0.028511376059372002]

new_data_point_X_7 = 		[0.377507, 0.792470, 0.070390, 0.169817, 0.329624, 0.157411]
new_data_point_y_7 = [0.5178188686002313]

new_data_point_X_8 = 		[0.941902, 0.323744, 0.769689, 0.743990, 0.638263, 0.912585]
new_data_point_y_8 = [0.09415579954501725]

new_data_point_X_9 = 		[0.181744, 0.181375, 0.700992, 0.001009, 0.630436, 0.888688]
new_data_point_y_9 = [0.474109042356806]

new_data_point_X_10 = [0.870369, 0.802211, 0.956458, 0.893957, 0.051341, 0.578796]
new_data_point_y_10 = [0.01727985510247288]

new_data_point_X_11 = [0.032224, 0.100510, 0.928704, 0.065394, 0.510110, 0.068991]
new_data_point_y_11 = [0.03428832780528657]

X_train = np.vstack([X_train, new_data_point_X_1, new_data_point_X_2, new_data_point_X_3, new_data_point_X_4, new_data_point_X_5,
                     new_data_point_X_6, new_data_point_X_7, new_data_point_X_8, new_data_point_X_9, new_data_point_X_10, new_data_point_X_11])
y_train = np.append(y_train, [new_data_point_y_1, new_data_point_y_2, new_data_point_y_3, new_data_point_y_4, new_data_point_y_5,
                              new_data_point_y_6, new_data_point_y_7, new_data_point_y_8, new_data_point_y_9, new_data_point_y_10, new_data_point_y_11])

print(X_train)
print(y_train)
print()


def bayesian_optimization_nd(X_train, y_train, n_dims, n_iterations=50,
                             n_initial=10, acquisition='ei'):
    """
    Bayesian Optimization for n-dimensional functions.
    Assumes bounds [0, 1] for all dimensions.
    """
    bounds = [(0, 1) for _ in range(n_dims)]

    # Initialize
    X_samples = X_train
    y_samples = y_train

    best_values = [y_samples.max()]

    print(f"Starting {n_dims}D optimization with {n_initial} initial samples...")
    print(f"Initial best: {y_samples.max():.4f}\n")

    for iteration in range(n_iterations):
        # Fit GP
        kernel = ConstantKernel(1.0) * RBF(length_scale=0.3)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6,
                                      n_restarts_optimizer=5)
        gp.fit(X_samples, y_samples)

        # Optimize acquisition function
        def acq_objective(X):
            X = X.reshape(1, -1)
            mu, sigma = gp.predict(X, return_std=True)

            if acquisition == 'ei':
                acq_val = expected_improvement(mu, sigma, y_samples.max())
            else:
                acq_val = upper_confidence_bound(mu, sigma, kappa=2.0)

            return -acq_val

        # Multi-start optimization
        best_acq = np.inf
        X_next = None

        for _ in range(20):  # More starts for higher dimensions
            x0 = np.random.uniform(0, 1, n_dims)
            result = minimize(acq_objective, x0, bounds=bounds, method='L-BFGS-B')
            if result.fun < best_acq:
                best_acq = result.fun
                X_next = result.x

        # Evaluate
        # y_next = objective_func(X_next.reshape(1, -1)).ravel()

        # Update
        X_samples = np.vstack([X_samples, X_next])
        #y_samples = np.append(y_samples, y_next)
        #best_values.append(y_samples.max())

        if (iteration + 1) % 10 == 0:
            print(f"Iteration {iteration + 1}: Best f(x)={y_samples.max():.4f}")

    # return X_samples, y_samples, best_values
    return X_samples, best_values


# Run 6d optimization
X_samples_6d, best_values_6d = bayesian_optimization_nd(X_train, y_train,
    n_dims=6,
    n_iterations=1,
    acquisition='ei'
)

print(X_samples_6d, best_values_6d )