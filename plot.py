#!/usr/bin/env python3

# MIT License

# Copyright (c) 2024 Hoel Kervadec

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import pickle

def load_metrics(file_path: Path) -> np.ndarray:
    """Load metrics from a .npy or .pkl file."""
    if file_path.suffix == '.npy':
        return np.load(file_path)
    elif file_path.suffix == '.pkl':
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            # Transform the data into the desired shape: metrics[epoch, patient, class]
            E = len(data) # Count outer keys
            first_epoch_key = next(iter(data))
            N = len(data[first_epoch_key])  # Get number of samples
            K = len(next(iter(data[first_epoch_key].values())))  # Get number of classes from the first sample

            metrics = np.zeros((E, N, K))

            for epoch, patients in data.items():
                for patient_index, (patient_id, class_values) in enumerate(patients.items()):
                    metrics[epoch, patient_index, :] = class_values

            return metrics
    else:
        raise ValueError(f"Unsupported file type: {file_path.suffix}")

def run(args: argparse.Namespace) -> None:
    metrics: np.ndarray = load_metrics(args.metric_file)
    match metrics.ndim:
        case 2:
            E, N = metrics.shape
            K = 1
        case 3:
            E, N, K = metrics.shape

    # Compute the mean for each k and epoch
    mean_scores = metrics.mean(axis=1).mean(axis=1)  # Average across axis 1 (over samples) and axis 2 (over k)
    best_epoch = np.argmax(mean_scores)  # Find the epoch with the highest mean score

    print(f"Best epoch: {best_epoch}")

    fig = plt.figure()
    ax = fig.gca()
    #ax.set_title(str(args.metric_file))
    ax.set_title(args.plot_title)
    if args.x_label is None:
        ax.set_xlabel('Number of Epochs')
    else:
        ax.set_xlabel(args.x_label)
    ax.set_ylabel(args.y_label)

    epcs = np.arange(E)

    labels = ["Background","Esophagus","Heart","Trachea","Aorta"]
    colors = ['darkgray', '#7FAC7F', '#EBD08D', '#AC7662', '#6BB1CA']

    for k in range(1, K):
        y = metrics[:, :, k].mean(axis=1)
        #std = metrics[:, :, k].std(axis=1)
        if K > 2:
            ax.plot(epcs, y, label=labels[k], color=colors[k], linewidth=1.5)
            #ax.fill_between(epcs, y - std, y + std, color=colors[k], alpha=0.3)
            print(f"Best score {labels[k]}: {y[best_epoch]}")
        else:
            ax.plot(epcs, y, label=f"{k=}", linewidth=1.5)

    if K > 2:
        ax.plot(epcs, metrics.mean(axis=1).mean(axis=1), label="All classes", linewidth=3, color=colors[0])
        #std = metrics.mean(axis=1).std(axis=1)
        #ax.fill_between(epcs, metrics.mean(axis=1).mean(axis=1) - std, metrics.mean(axis=1).mean(axis=1) + std, color=colors[0], alpha=0.3)
        ax.legend()
        print(f"Best score overall: {metrics.mean(axis=1).mean(axis=1)[best_epoch]}")
    else:
        ax.plot(epcs, metrics.mean(axis=1), linewidth=3)

    # Set y-axis limits between given values
    if args.set_ylim:
        ax.set_ylim(args.ylim_lower, args.ylim_upper)

    fig.tight_layout()
    if args.dest:
        fig.savefig(args.dest)

    if not args.headless:
        plt.show()


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Plot data over time')
    parser.add_argument('--metric_file', type=Path, required=True, metavar="METRIC_MODE.npy",
                        help="The metric file to plot.")
    parser.add_argument('--dest', type=Path, metavar="METRIC_MODE.png",
                        help="Optional: save the plot to a .png file")
    parser.add_argument("--headless", action="store_true",
                        help="Does not display the plot and save it directly (implies --dest to be provided.")
    parser.add_argument("--plot_title", type=str, required=True, help="Graph title")
    parser.add_argument("--y_label", type=str, required=True, help="Label for the y-axis of the plot")
    parser.add_argument("--x_label", type=str, help="Label for x-axis of the plot")
    parser.add_argument("--set_ylim", type=bool, default=False, help="Set to True if you want to specify a range for the y-axis")
    parser.add_argument("--ylim_lower", type=float, help="Lower limit of y-axis")
    parser.add_argument("--ylim_upper", type=float, help="Upper limit of y-axis")

    args = parser.parse_args()

    print(args)

    return args


if __name__ == "__main__":
    run(get_args())
