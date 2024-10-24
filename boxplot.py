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
    # Load the metrics from the provided .npy file
    metrics: np.ndarray = load_metrics(args.metric_file)
    #print(metrics)
    
    # Handle cases where the metric array has 2 or 3 dimensions
    match metrics.ndim:
        case 2:
            E, N = metrics.shape  # E: epochs, N: samples
            K = 1  # No classes dimension
        case 3:
            E, N, K = metrics.shape  # E: epochs, N: samples, K: classes

    # Compute the mean for each k and epoch
    if args.epoch is None:
        mean_scores = metrics.mean(axis=1).mean(axis=1)  # Average across axis 1 (over samples) and axis 2 (over k)
        best_epoch = np.argmax(mean_scores)  # Find the epoch with the highest mean score

        print(f"Best epoch: {best_epoch}")
    else:
        best_epoch = args.epoch

    # Create a figure and axis for the boxplot
    fig = plt.figure()
    ax = fig.gca()

    # Set title and labels
    ax.set_title(args.plot_title)
    if args.x_label is None:
        ax.set_xlabel(f'Data at Epoch {best_epoch}')
    else:
        ax.set_xlabel(args.x_label)
    ax.set_ylabel(args.y_label)

    # Extract data for the best epoch
    best_epoch_data = metrics[best_epoch, :, :] if K > 1 else metrics[best_epoch, :]

    # Generate boxplot with different colors for each class
    if K > 1:
        labels = ["Background", "Esophagus", "Heart", "Trachea", "Aorta"]
        colors = ['darkgray', '#7FAC7F', '#EBD08D', '#AC7662', '#6BB1CA']
        
        for k in range(K):
            print(f"Median {labels[k]}: {np.median(best_epoch_data[:,k])}")
            print(f"Mean {labels[k]}: {np.mean(best_epoch_data[:,k])}")
            print(f"Standard Deviation {labels[k]}: {np.std(best_epoch_data[:,k])}")

        # Create boxplots for each class with a unique color
        boxplots = ax.boxplot([best_epoch_data[:, k] for k in range(1,K)], patch_artist=True, tick_labels=labels[1:])
        
        # Apply unique colors to each boxplot
        for patch, color in zip(boxplots['boxes'], colors[1:]):
            patch.set_facecolor(color)

        if args.include_avg:
            # Calculate and add a boxplot for the average values of each class
            average_values = best_epoch_data.mean(axis=0)
            ax.boxplot(average_values[1:], positions=[len(labels[1:]) + 1], patch_artist=True, boxprops=dict(facecolor=colors[0]), widths=0.5, tick_labels=["Average"])

    else:
        ax.boxplot(best_epoch_data, patch_artist=True)

    # Set y-axis limits if specified
    if args.set_ylim:
        ax.set_ylim(args.ylim_lower, args.ylim_upper)

    # Save or display the plot
    fig.tight_layout()
    if args.dest:
        fig.savefig(args.dest)

    if not args.headless:
        plt.show()


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Plot boxplot for the best epoch data')
    parser.add_argument('--metric_file', type=Path, required=True, metavar="METRIC_MODE.npy",
                        help="The metric file to plot.")
    parser.add_argument('--dest', type=Path, metavar="METRIC_MODE.png",
                        help="Optional: save the plot to a .png file")
    parser.add_argument("--headless", action="store_true",
                        help="Does not display the plot and save it directly (implies --dest to be provided).")
    parser.add_argument('--epoch', type=int, help="Epoch to use for the data in the boxplot")                    
    parser.add_argument("--include_avg", type=bool, default=False, help="Set to True if you want to include a box for the average")
    parser.add_argument("--plot_title", type=str, required=True, help="Boxplot title")
    parser.add_argument("--x_label", type=str, help="Label for x-axis of the plot")
    parser.add_argument("--y_label", type=str, required=True, help="Label for the y-axis of the plot")
    parser.add_argument("--set_ylim", type=bool, default=False, help="Set to True if you want to specify a range for the y-axis")
    parser.add_argument("--ylim_lower", type=float, help="Lower limit of y-axis")
    parser.add_argument("--ylim_upper", type=float, help="Upper limit of y-axis")

    args = parser.parse_args()

    print(args)

    return args


if __name__ == "__main__":
    run(get_args())
