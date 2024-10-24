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
            E = len(data)  # Count outer keys
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

    mean_scores = metrics.mean(axis=1).mean(axis=1)
    best_epoch = np.argmax(mean_scores)

    print(f"Best epoch: {best_epoch}")

    fig = plt.figure()
    ax = fig.gca()

    ax.set_title(args.plot_title)
    ax.set_xlabel(f'Data at Best Epoch (Epoch {best_epoch})')
    ax.set_ylabel(args.y_label)

    best_epoch_data = metrics[best_epoch, :, :] if K > 1 else metrics[best_epoch, :]

    if K > 1:
        # Define labels for the relevant classes only (excluding "#All Classes")
        labels = ["Esophagus", "Heart", "Trachea", "Aorta"]
        colors = ['#7FAC7F', '#EBD08D', '#AC7662', '#6BB1CA']
        
        # Select data for the classes 
        best_epoch_data = best_epoch_data[:, 1:5]  #Removed Background class

        parts = ax.violinplot([best_epoch_data[:, k] for k in range(best_epoch_data.shape[1])], showmeans=True, showmedians=True, showextrema=False)

        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels)  

        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)

    else:
        ax.violinplot(best_epoch_data, showmeans=True, showmedians=False)

    if args.set_ylim:
        ax.set_ylim(args.ylim_lower, args.ylim_upper)

    fig.tight_layout()
    if args.dest:
        fig.savefig(args.dest)

    if not args.headless:
        plt.show()

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Plot violin plot for the best epoch data')
    parser.add_argument('--metric_file', type=Path, required=True, metavar="METRIC_MODE.npy",
                        help="The metric file to plot.")
    parser.add_argument('--dest', type=Path, metavar="METRIC_MODE.png",
                        help="Optional: save the plot to a .png file")
    parser.add_argument("--headless", action="store_true",
                        help="Does not display the plot and save it directly (implies --dest to be provided).")
    parser.add_argument("--plot_title", type=str, required=True, help="Violin plot title")
    parser.add_argument("--y_label", type=str, required=True, help="Label for the y-axis of the plot")
    parser.add_argument("--set_ylim", type=bool, default=False, help="Set to True if you want to specify a range for the y-axis")
    parser.add_argument("--ylim_lower", type=float, help="Lower limit of y-axis")
    parser.add_argument("--ylim_upper", type=float, help="Upper limit of y-axis")

    args = parser.parse_args()

    print(args)

    return args

if __name__ == "__main__":
    run(get_args())
