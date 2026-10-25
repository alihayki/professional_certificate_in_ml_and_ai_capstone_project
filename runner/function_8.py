import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Import our visualization functions
from capstone_project.utility.acquisition_functions import upper_confidence_bound, expected_improvement


X_train = np.load("../initial_data/function_8/initial_inputs.npy")
y_train = np.load("../initial_data/function_8/initial_outputs.npy")

new_data_point_X_1 = [0.078418, 0.210342, 0.116367, 0.163052, 0.690208, 0.531367, 0.184668, 0.565198]
new_data_point_y_1 = [9.9870970581276]

new_data_point_X_2 = 		[0.129734, 0.187639, 0.301854, 0.754706, 0.433158, 0.875532, 0.245985, 0.105637]
new_data_point_y_2 = [9.305564818250101]

new_data_point_X_3 = 		[0.280863, 0.909956, 0.057332, 0.416616, 0.840158, 0.604233, 0.816914, 0.832933]
new_data_point_y_3 = [8.4918556265861]

new_data_point_X_4 = 		[0.833657, 0.573491, 0.137679, 0.548037, 0.016727, 0.260868, 0.039040, 0.086080]
new_data_point_y_4 = [8.1433698016005]

new_data_point_X_5 = 		[0.743058, 0.015683, 0.885445, 0.043528, 0.884828, 0.619112, 0.086954, 0.903223]
new_data_point_y_5 = [7.378945251583099]

new_data_point_X_6 = 		[0.380350, 0.052276, 0.332138, 0.038533, 0.989665, 0.350403, 0.144315, 0.131561]
new_data_point_y_6 = [9.6297427529594]

new_data_point_X_7 = 		[0.020236, 0.260497, 0.570879, 0.030770, 0.052551, 0.524390, 0.019519, 0.219651]
new_data_point_y_7 = [9.0181789559734]

new_data_point_X_8 = 		[0.496856, 0.033826, 0.091486, 0.715399, 0.931182, 0.968806, 0.234273, 0.951399]
new_data_point_y_8 = [9.1043073914869]

new_data_point_X_9 = 		[0.841680, 0.066398, 0.406932, 0.769267, 0.608244, 0.045965, 0.896713, 0.026843]
new_data_point_y_9 = [7.0510647792391]

new_data_point_X_10 = [0.600755, 0.036443, 0.055877, 0.045209, 0.494287, 0.128331, 0.159346, 0.128205]
new_data_point_y_10 = [9.247697244253]

new_data_point_X_11 = [0.366909, 0.890016, 0.310192, 0.650678, 0.910390, 0.199424, 0.001339, 0.265367]
new_data_point_y_11 = [8.7752403427691]

X_train = np.vstack([X_train, new_data_point_X_1, new_data_point_X_2, new_data_point_X_3,new_data_point_X_4, new_data_point_X_5,
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


# Run 8d optimization
X_samples_8d, best_values_8d = bayesian_optimization_nd(X_train, y_train,
    n_dims=8,
    n_iterations=1,
    acquisition='ei'
)

print(X_samples_8d, best_values_8d )