"""Runner for benchmark function 2 (2 dimensions)."""

import numpy as np

from runner.benchmark_runner import BenchmarkRunner


class Function2Runner(BenchmarkRunner):
    FUNCTION_ID = 2
    DIMENSIONS = 2
    LENGTH_SCALE = 0.2
    N_RESTARTS_OPTIMIZER = 10
    N_STARTS = 10
    ACQUISITION = 'ei'

    ADDITIONAL_INPUTS = np.array(
        [
            [0.784851, 0.962255],
            [0.714299, 0.853921],
            [0.867715, 0.126193],
            [0.482429, 0.955208],
            [0.047652, 0.016612],
            [0.687982, 0.441207],
            [0.731828, 0.875343],
            [0.243043, 0.752085],
            [0.136922, 0.612922],
            [0.265451, 0.954748],
            [0.305598, 0.490409],
            [0.060996, 0.541564],
        ],
        dtype=float,
    )
    ADDITIONAL_OUTPUTS = np.array(
        [
            0.05412601145278319,
            0.6023518670131792,
            0.23265727595901992,
            0.13571335326110615,
            -0.0037459982246542097,
            0.6140661961806388,
            0.5616664153246701,
            -0.02631183058571447,
            -0.1666534406284661,
            -0.02021069405508203,
            0.010773909071369311,
            -0.006385232790677246,
        ],
        dtype=float,
    )


if __name__ == "__main__":
    Function2Runner().run()
