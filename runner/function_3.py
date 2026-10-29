"""Runner for benchmark function 3 (3 dimensions)."""

import numpy as np

from runner.benchmark_runner import BenchmarkRunner


class Function3Runner(BenchmarkRunner):
    FUNCTION_ID = 3
    DIMENSIONS = 3
    LENGTH_SCALE = 0.3
    N_RESTARTS_OPTIMIZER = 5
    N_STARTS = 20
    ACQUISITION = 'ei'

    ADDITIONAL_INPUTS = np.array(
        [
            [0.467618, 0.103729, 0.53068],
            [0.213096, 0.945174, 0.76147],
            [0.75596, 0.864526, 0.772123],
            [0.693172, 0.794609, 0.719455],
            [0.977634, 0.910693, 0.018434],
            [0.890066, 0.071176, 0.574784],
            [0.584071, 0.087333, 0.990518],
            [0.911868, 0.516988, 0.259088],
            [0.885451, 0.507967, 0.784365],
            [0.719747, 0.987022, 0.18604],
            [0.96574, 0.159856, 0.983122],
            [0.334892, 0.742871, 0.668583],
        ],
        dtype=float,
    )
    ADDITIONAL_OUTPUTS = np.array(
        [
            -0.05809710299313475,
            -0.12881751750441095,
            -0.12123941083522227,
            -0.12987283482410464,
            -0.12913955169889024,
            -0.10116654577436605,
            -0.45036812933834897,
            -0.09405996319041127,
            -0.07527399288483543,
            -0.14781385193044666,
            -0.3843397511420752,
            -0.11078238798534756,
        ],
        dtype=float,
    )


if __name__ == "__main__":
    Function3Runner().run()
