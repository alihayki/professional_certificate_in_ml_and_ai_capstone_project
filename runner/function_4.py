import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Import our visualization functions
from capstone_project.utility.acquisition_functions import upper_confidence_bound, expected_improvement


X_train = np.load("../initial_data/function_4/initial_inputs.npy")
y_train = np.load("../initial_data/function_4/initial_outputs.npy")

new_data_point_X_1 = [0.427098, 0.390844, 0.362688, 0.412116]
new_data_point_y_1 = [0.5798393591182287]


new_data_point_X_2 = 	[0.413245, 0.894560, 0.117415, 0.385740]
new_data_point_y_2 = [-16.352746413359387]

new_data_point_X_3 = 	[0.661355, 0.455413, 0.001327, 0.314611]
new_data_point_y_3 = [-12.780893182248878]

new_data_point_X_4 = 	[0.377718, 0.214109, 0.990146, 0.840797]
new_data_point_y_4 = [-26.008041795690733]

new_data_point_X_5 = 	[0.136836, 0.699431, 0.643861, 0.977821]
new_data_point_y_5 = [-24.22450809104549]

new_data_point_X_6 = 	[0.135301, 0.278127, 0.861470, 0.289278]
new_data_point_y_6 = [-15.738034945104683]

new_data_point_X_7 = 	[0.921366, 0.112661, 0.740332, 0.824322]
new_data_point_y_7 = [-28.998055535358528]

new_data_point_X_8 = 		[0.905800, 0.008377, 0.276337, 0.491480]
new_data_point_y_8 = [-20.014771636296256]

new_data_point_X_9 = 		[0.075168, 0.548446, 0.951631, 0.124515]
new_data_point_y_9 = [-22.42200284031674]

new_data_point_X_10 = [0.265234, 0.502124, 0.014981, 0.449422]
new_data_point_y_10 = [-9.786970571535857]

new_data_point_X_11 = [0.987609, 0.065670, 0.969615, 0.221392]
new_data_point_y_11 = [-32.683385103085335]

X_train = np.vstack([X_train, new_data_point_X_1, new_data_point_X_2, new_data_point_X_3, new_data_point_X_4, new_data_point_X_5,
                     new_data_point_X_6, new_data_point_X_7, new_data_point_X_8, new_data_point_X_9, new_data_point_X_10])
y_train = np.append(y_train, [new_data_point_y_1, new_data_point_y_2, new_data_point_y_3,new_data_point_y_4, new_data_point_y_5,
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