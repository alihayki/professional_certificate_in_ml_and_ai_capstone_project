import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Import our visualization functions
from capstone_project.utility.acquisition_functions import upper_confidence_bound, expected_improvement


X_train = np.load("../initial_data/function_2/initial_inputs.npy")
y_train = np.load("../initial_data/function_2/initial_outputs.npy")

new_data_point_X_1 = [0.784851, 0.962255]
new_data_point_y_1 = [0.05412601145278319]

new_data_point_X_2 = [0.714299, 0.853921]
new_data_point_y_2 = [0.6023518670131792]

new_data_point_X_3 = [0.867715, 0.126193]
new_data_point_y_3 = [0.23265727595901992]

new_data_point_X_4 = [0.482429, 0.955208]
new_data_point_y_4 = [0.13571335326110615]

new_data_point_X_5 = [0.047652, 0.016612]
new_data_point_y_5 = [-0.0037459982246542097]

new_data_point_X_6 = [0.687982, 0.441207]
new_data_point_y_6 = [0.6140661961806388]

new_data_point_X_7 = [0.731828, 0.875343]
new_data_point_y_7 = [0.5616664153246701]

new_data_point_X_8 = [0.243043, 0.752085]
new_data_point_y_8 = [-0.02631183058571447]

new_data_point_X_9 = [0.136922, 0.612922]
new_data_point_y_9 = [-0.1666534406284661]

new_data_point_X_10 = [0.265451, 0.954748]
new_data_point_y_10 = [-0.02021069405508203]

new_data_point_X_11 = [0.305598, 0.490409]
new_data_point_y_11 = [0.010773909071369311]

X_train = np.vstack([X_train, new_data_point_X_1, new_data_point_X_2, new_data_point_X_3, new_data_point_X_4, new_data_point_X_5,
                     new_data_point_X_6, new_data_point_X_7, new_data_point_X_8, new_data_point_X_9, new_data_point_X_10, new_data_point_X_11])
y_train = np.append(y_train, [new_data_point_y_1, new_data_point_y_2, new_data_point_y_3, new_data_point_y_4, new_data_point_y_5,
                              new_data_point_y_6, new_data_point_y_7, new_data_point_y_8, new_data_point_y_9, new_data_point_y_10, new_data_point_y_11])

print(X_train)
print(y_train)

def bayesian_optimization_2d(X_train, y_train, bounds, n_iterations=20,
                             n_initial=5, acquisition='ei'):
    """
    Bayesian Optimization for 2D functions.

    Parameters:
    -----------
    objective_func : callable, function to maximize
    bounds : list of tuples, [(x1_min, x1_max), (x2_min, x2_max)]
    n_iterations : int, number of BO iterations
    n_initial : int, number of random initial samples
    acquisition : str, 'ei' or 'ucb'
    """
    n_dims = len(bounds)

    # Initialize with random samples
    X_samples = X_train
    print(X_samples)
    y_samples = y_train
    print(y_samples)
    best_values = [y_samples.max()]

    for iteration in range(n_iterations):
        # Fit GP
        kernel = ConstantKernel(1.0) * RBF(length_scale=0.2)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6,
                                      n_restarts_optimizer=10)
        gp.fit(X_samples, y_samples)

        # Define acquisition function for optimization
        def acq_objective(X):
            X = X.reshape(1, -1)
            mu, sigma = gp.predict(X, return_std=True)

            if acquisition == 'ei':
                acq_val = expected_improvement(mu, sigma, y_samples.max())
            else:
                acq_val = upper_confidence_bound(mu, sigma, kappa=2.0)

            return -acq_val  # Negative because we use the 'minimize' function later

        # Optimize acquisition function (multi-start)
        best_acq = np.inf
        X_next = None

        for _ in range(10):  # Multiple random starts
            x0 = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds])
            result = minimize(
                acq_objective, x0,
                bounds=bounds,
                method='L-BFGS-B'
            )
            if result.fun < best_acq:
                best_acq = result.fun
                X_next = result.x

        # Evaluate objective at next point
        # y_next = objective_func(X_next.reshape(1, -1)).ravel()

        # Add to dataset
        X_samples = np.vstack([X_samples, X_next])
        # y_samples = np.append(y_samples, y_next)
        # best_values.append(y_samples.max())


    # return X_samples, y_samples, best_values
    return X_samples, best_values


# Run 2D optimization
print("Running 2D Bayesian Optimization...\n")
X_samples_2d, best_values_2d = bayesian_optimization_2d(X_train, y_train,
    bounds=[(0, 1), (0, 1)],
    n_iterations=1,
    acquisition='ei'
)
print(X_samples_2d, best_values_2d )