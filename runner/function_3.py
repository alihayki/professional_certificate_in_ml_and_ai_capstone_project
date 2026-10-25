import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Import our visualization functions
from capstone_project.utility.acquisition_functions import upper_confidence_bound, expected_improvement


X_train = np.load("../initial_data/function_3/initial_inputs.npy")
y_train = np.load("../initial_data/function_3/initial_outputs.npy")

new_data_point_X_1 = [0.467618, 0.103729, 0.530680]
new_data_point_y_1 = [-0.05809710299313475]

new_data_point_X_2 = 	[0.213096, 0.945174, 0.761470]
new_data_point_y_2 = [-0.12881751750441095]

new_data_point_X_3 = 	[0.755960, 0.864526, 0.772123]
new_data_point_y_3 = [-0.12123941083522227]

new_data_point_X_4 = 	[0.693172, 0.794609, 0.719455]
new_data_point_y_4 = [-0.12987283482410464]

new_data_point_X_5 = 	[0.977634, 0.910693, 0.018434]
new_data_point_y_5 = [-0.12913955169889024]

new_data_point_X_6 = 	[0.890066, 0.071176, 0.574784]
new_data_point_y_6 = [-0.10116654577436605]

new_data_point_X_7 = 	[0.584071, 0.087333, 0.990518]
new_data_point_y_7 = [-0.45036812933834897]

new_data_point_X_8 = 	[0.911868, 0.516988, 0.259088]
new_data_point_y_8 = [-0.09405996319041127]

new_data_point_X_9 = 	[0.885451, 0.507967, 0.784365]
new_data_point_y_9 = [-0.07527399288483543]

new_data_point_X_10 = [0.719747, 0.987022, 0.186040]
new_data_point_y_10 = [-0.14781385193044666]

new_data_point_X_11 = [0.965740, 0.159856, 0.983122]
new_data_point_y_11 = [-0.3843397511420752]

X_train = np.vstack([X_train, new_data_point_X_1, new_data_point_X_2, new_data_point_X_3, new_data_point_X_4, new_data_point_X_5,
                     new_data_point_X_6,new_data_point_X_7, new_data_point_X_8, new_data_point_X_9, new_data_point_X_10, new_data_point_X_11])
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


# Run 3D optimization
X_samples_3d, best_values_3d = bayesian_optimization_nd(X_train, y_train,
    n_dims=3,
    n_iterations=1,
    acquisition='ei'
)

print(X_samples_3d, best_values_3d )