import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Import our visualization functions
from capstone_project.utility.acquisition_functions import upper_confidence_bound, expected_improvement


X_train = np.load("../initial_data/function_5/initial_inputs.npy")
y_train = np.load("../initial_data/function_5/initial_outputs.npy")

new_data_point_X_1 = [0.142863, 0.061082, 0.156043, 0.094943]
new_data_point_y_1 = [161.5219897777812]

new_data_point_X_2 = 		[0.392033, 0.052940, 0.617408, 0.418812]
new_data_point_y_2 = [50.35465430868304]

new_data_point_X_3 = 		[0.103602, 0.771897, 0.390398, 0.912755]
new_data_point_y_3 = [257.37996844682885]

new_data_point_X_4 = 		[0.295262, 0.235436, 0.206612, 0.315191]
new_data_point_y_4 = [144.21950606154775]

new_data_point_X_5 = 		[0.326948, 0.830017, 0.977678, 0.946911]
new_data_point_y_5 = [2107.434680025152]

new_data_point_X_6 = 		[0.704452, 0.395251, 0.187101, 0.896163]
new_data_point_y_6 = [143.00391103851013]

new_data_point_X_7 = 		[0.507763, 0.483538, 0.225736, 0.882967]
new_data_point_y_7 = [49.70352653540951]

new_data_point_X_8 = 		[0.319021, 0.837563, 0.977659, 0.935901]
new_data_point_y_8 = [2058.916581090673]

new_data_point_X_9 = 		[0.102215, 0.593676, 0.810880, 0.627370]
new_data_point_y_9 = [44.56542276861604]

new_data_point_X_10 = [0.542004, 0.985988, 0.394438, 0.873415]
new_data_point_y_10 = [960.0485450479752]

new_data_point_X_11 = [0.675164, 0.785179, 0.994116, 0.030528]
new_data_point_y_11 = [843.9878249188754]

X_train = np.vstack([X_train, new_data_point_X_1, new_data_point_X_2, new_data_point_X_3, new_data_point_X_4, new_data_point_X_5,
                     new_data_point_X_6, new_data_point_X_7, new_data_point_X_8, new_data_point_X_9, new_data_point_X_10])
y_train = np.append(y_train, [new_data_point_y_1, new_data_point_y_2, new_data_point_y_3, new_data_point_y_4, new_data_point_y_5,
                              new_data_point_y_6, new_data_point_y_7, new_data_point_y_8, new_data_point_y_9, new_data_point_y_10])

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


# Run 4d optimization
X_samples_4d, best_values_4d = bayesian_optimization_nd(X_train, y_train,
    n_dims=4,
    n_iterations=1,
    acquisition='ei'
)

print(X_samples_4d, best_values_4d )