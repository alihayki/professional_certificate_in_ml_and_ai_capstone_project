import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Import our visualization functions
from capstone_project.utility.acquisition_functions import upper_confidence_bound, expected_improvement


X_train = np.load("../initial_data/function_6/initial_inputs.npy")
y_train = np.load("../initial_data/function_6/initial_outputs.npy")

new_data_point_X_1 = [0.000000, 0.147069, 0.128883, 1.000000, 0.000000]
new_data_point_y_1 = [-1.5401585272834453]

new_data_point_X_2 = 		[0.340010, 0.093508, 0.286299, 0.204504, 0.096102]
new_data_point_y_2 = [-1.3105249273950572]

new_data_point_X_3 = 		[0.408010, 0.074684, 0.842808, 0.210899, 0.098990]
new_data_point_y_3 = [-1.0047369990346784]

new_data_point_X_4 = 		[0.485071, 0.917631, 0.956317, 0.926125, 0.977978]
new_data_point_y_4 = [-1.9423209082355182]

new_data_point_X_5 = 		[0.409928, 0.056932, 0.580295, 0.946684, 0.004202]
new_data_point_y_5 = [-0.6526695954914449]

new_data_point_X_6 = 		[0.461674, 0.025892, 0.896063, 0.889086, 0.964671]
new_data_point_y_6 = [-1.6728977447942832]

new_data_point_X_7 = 		[0.129357, 0.129305, 0.984005, 0.222963, 0.874081]
new_data_point_y_7 = [-2.2501800732780666]

new_data_point_X_8 = 		[0.014157, 0.961554, 0.267941, 0.613037, 0.073017]
new_data_point_y_8 = [-1.5856469147314127]

new_data_point_X_9 = 		[0.948380, 0.482002, 0.735793, 0.041179, 0.552061]
new_data_point_y_9 = [-1.9540839191548027]

new_data_point_X_10 = [0.929295, 0.081420, 0.474925, 0.241035, 0.905305]
new_data_point_y_10 = [-2.3234044653035975]

new_data_point_X_11 = [0.010904, 0.172893, 0.962173, 0.878068, 0.730497]
new_data_point_y_11 = [-1.666734980242383]


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


# Run 5d optimization
X_samples_5d, best_values_5d = bayesian_optimization_nd(X_train, y_train,
    n_dims=5,
    n_iterations=1,
    acquisition='ei'
)

print(X_samples_5d, best_values_5d )