"""Runner for benchmark function 1 (2 dimensions)."""

import numpy as np

from runner.benchmark_runner import BenchmarkRunner


class Function1Runner(BenchmarkRunner):
    FUNCTION_ID = 1
    DIMENSIONS = 2
    LENGTH_SCALE = 0.2
    N_RESTARTS_OPTIMIZER = 10
    N_STARTS = 10
    ACQUISITION = 'ei'

    ADDITIONAL_INPUTS = np.array(
        [
            [0.042174, 0.96619],
            [0.837142, 0.525106],
            [0.617858, 0.661682],
            [0.842757, 0.217224],
            [0.333036, 0.915354],
            [0.083472, 0.472931],
            [0.663829, 0.311178],
            [0.976474, 0.157099],
            [0.700893, 0.330066],
            [0.991453, 0.481121],
            [0.965492, 0.368293],
            [0.920261, 0.230799],
        ],
        dtype=float,
    )
    ADDITIONAL_OUTPUTS = np.array(
        [
            5.92955205163488e-307,
            1.0912815543194745e-38,
            0.063472485153747,
            3.454191981626374e-150,
            9.753317946930657e-119,
            -1.6354621807969917e-82,
            1.536045417036192e-50,
            1.202526273466913e-239,
            -1.3656268909042339e-61,
            -1.6462874540538084e-107,
            -7.493173745458915e-127,
            -1.0605783947692943e-169,
        ],
        dtype=float,
    )


if __name__ == "__main__":
    Function1Runner().run()
