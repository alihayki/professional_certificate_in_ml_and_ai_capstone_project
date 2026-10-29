"""Save a PDF chart of the additional outputs for all benchmark functions."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from runner.function_1 import Function1Runner
from runner.function_2 import Function2Runner
from runner.function_3 import Function3Runner
from runner.function_4 import Function4Runner
from runner.function_5 import Function5Runner
from runner.function_6 import Function6Runner
from runner.function_7 import Function7Runner
from runner.function_8 import Function8Runner
from utility.plotting_utils import plot_additional_outputs


class AdditionalOutputsPlotRunner:
    """Create and save a PDF plot of each function's recorded outputs."""

    RUNNERS = (
        Function1Runner,
        Function2Runner,
        Function3Runner,
        Function4Runner,
        Function5Runner,
        Function6Runner,
        Function7Runner,
        Function8Runner,
    )

    def __init__(self, output_path=None):
        project_root = Path(__file__).resolve().parents[1]
        self.output_path = Path(output_path) if output_path else project_root / "additional_outputs.pdf"
        if self.output_path.suffix.lower() != ".pdf":
            raise ValueError("The output path must use the .pdf extension.")

    def run(self):
        """Generate the plot and save it as a PDF at the configured path."""
        outputs_by_function = {
            runner.FUNCTION_ID: runner.ADDITIONAL_OUTPUTS
            for runner in self.RUNNERS
        }
        figure, _ = plot_additional_outputs(outputs_by_function)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(self.output_path, format="pdf", bbox_inches="tight")
        plt.close(figure)
        print(f"Saved output plots to {self.output_path}")
        return self.output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="PDF output path (default: additional_outputs.pdf in the project root).",
    )
    arguments = parser.parse_args()
    AdditionalOutputsPlotRunner(arguments.output).run()
