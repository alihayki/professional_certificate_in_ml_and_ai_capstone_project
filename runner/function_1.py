import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Import our visualization functions
from capstone_project.utility.acquisition_functions import upper_confidence_bound, expected_improvement


X_train = np.load("../initial_data/function_1/initial_inputs.npy")
y_train = np.load("../initial_data/function_1/initial_outputs.npy")

new_data_point_X_1 = [0.042174, 0.966190]
new_data_point_y_1 = [5.92955205163488e-307]

new_data_point_X_2 = [0.837142, 0.525106]
new_data_point_y_2 = [1.0912815543194745e-38]

new_data_point_X_3 = [0.617858, 0.661682]
new_data_point_y_3 = [0.063472485153747]

new_data_point_X_4 = [0.842757, 0.217224]
new_data_point_y_4 = [3.454191981626374e-150]

new_data_point_X_5 = [0.333036, 0.915354]
new_data_point_y_5 = [9.753317946930657e-119]

new_data_point_X_6 = [0.083472, 0.472931]
new_data_point_y_6 = [-1.6354621807969917e-82]

new_data_point_X_7 = [0.663829, 0.311178]
new_data_point_y_7 = [1.536045417036192e-50]

new_data_point_X_8 = [0.976474, 0.157099]
new_data_point_y_8 = [1.202526273466913e-239]

new_data_point_X_9 = [0.700893, 0.330066]
new_data_point_y_9 = [-1.3656268909042339e-61]

new_data_point_X_10 = [0.991453, 0.481121]
new_data_point_y_10 = [-1.6462874540538084e-107]

new_data_point_X_11 = [0.965492, 0.368293]
new_data_point_y_11 = [-7.493173745458915e-127]

X_train = np.vstack([X_train, new_data_point_X_1, new_data_point_X_2, new_data_point_X_3, new_data_point_X_4, new_data_point_X_5,
                     new_data_point_X_6, new_data_point_X_7, new_data_point_X_8, new_data_point_X_9, new_data_point_X_10, new_data_point_X_11])
y_train = np.append(y_train, [new_data_point_y_1, new_data_point_y_2, new_data_point_y_3, new_data_point_y_4, new_data_point_y_5,
                              new_data_point_y_6, new_data_point_y_7, new_data_point_y_8, new_data_point_y_9, new_data_point_y_10, new_data_point_y_11])

print(X_train)
print(y_train)


def bayesian_optimization_2d(X_train, y_train, bounds, n_iterations=20,
                             n_initial=5, acquisition='ucb'):
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