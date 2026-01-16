import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def add_bar_labels(ax, bars, is_int=False):
    # attaching a text label above each bar in *bars* to display its height
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            label = f"{int(height)}" if is_int else f"{height:.2f}"
            ax.annotate(
                label,
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # Some offset above the bar
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=18,
            )


def main():
    # Generating visualizations from the aggregated results CSV file as backup,
    # if such files were produced by an earlier run of replication_experiment/run_experiment.py
    INPUT_CSV = "replication_experiment/replication_results.csv"
    OUTPUT_DIR = INPUT_CSV.split("/")[0]

    if not os.path.exists(INPUT_CSV):
        print(
            f"Error: Input file not found at '{INPUT_CSV}'\nRun 'replication_experiment/run_experiment.py' first to generate results."
        )
        return

    print(f"Loading data from {INPUT_CSV}...")
    agg_df = pd.read_csv(INPUT_CSV)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial",
                "DejaVu Sans",
                "Liberation Sans",
                "Bitstream Vera Sans",
            ],
            "font.size": 12,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "figure.titlesize": 18,
            "figure.dpi": 300,
            "axes.grid": True,
            "grid.alpha": 0.5,
            "grid.linestyle": ":",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    # Distinct strategies get distinct colors or something like that
    strategy_styles = {
        "All": {"color": "#2C3E50"},
        "Pareto": {"color": "#E74C3C"},
        "TopK_Outcome": {"color": "#3498DB"},
        "TopK_Frequency": {"color": "#2ECC71"},
        "TopK_CaseDistance": {"color": "#F1C40F"},
    }

    print("Generating plots for each dataset...")
    for dataset_name in agg_df["dataset"].unique():
        dataset_title_name = (
            dataset_name.replace("_trunc.csv", "").replace("_", " ").title()
        )
        dataset_df = agg_df[agg_df["dataset"] == dataset_name]
        # Filter strategies to only those present in the data
        strategies = [
            s for s in strategy_styles.keys() if s in dataset_df["strategy"].unique()
        ]
        extension_steps = sorted(dataset_df["extension_step"].unique())
        # Skip if no data for this dataset
        if not extension_steps:
            print(f"No data to plot for {dataset_name}. Skipping plot generation.")
            continue

        # X-axis positions of #bars
        x = np.arange(len(extension_steps))
        n_strategies = len(strategies)
        bar_width = 0.8 / n_strategies if n_strategies > 0 else 0.8

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
        fig.suptitle(
            f"{dataset_title_name} Dataset: Performance vs. Complexity",
            fontsize=18,
            fontweight="bold",
        )

        # Subfig1 : Predictive performance (F1-Score = 2 * (precision * recall) / (precision + recall))
        ax1.set_title("Predictive Performance (F1-Score)", pad=15)
        for i, strategy in enumerate(strategies):
            subset = (
                dataset_df[dataset_df["strategy"] == strategy]
                .set_index("extension_step")
                .reindex(extension_steps)
            )
            # Missing data filled with zeros
            means = subset["mean_f1"].fillna(0)
            stds = subset["std_f1"].fillna(0)
            position = (
                x - (bar_width * n_strategies / 2) + (i * bar_width) + bar_width / 2
            )
            ax1.bar(
                position,
                means,
                bar_width,
                yerr=stds,
                label=strategy,
                color=strategy_styles[strategy]["color"],
                capsize=4,
                alpha=0.9,
            )
        ax1.set_ylabel("Weighted F1-Score")
        ax1.set_ylim(bottom=0)
        if not dataset_df.empty:
            # Top limit slightly above max bar height (incl. error bar)
            max_y = (dataset_df["mean_f1"] + dataset_df["std_f1"]).max()
            ax1.set_ylim(top=max(1.0, max_y * 1.1))

        # Subfig2 : Model Complexity (Number of Patterns)
        ax2.set_title("Model Complexity (Number of Patterns)", pad=15)
        for i, strategy in enumerate(strategies):
            subset = (
                dataset_df[dataset_df["strategy"] == strategy]
                .set_index("extension_step")
                .reindex(extension_steps)
            )
            # Same here, fill missing data with zeros
            num_patterns = subset["num_patterns"].fillna(0)
            position = (
                x - (bar_width * n_strategies / 2) + (i * bar_width) + bar_width / 2
            )
            bars = ax2.bar(
                position,
                num_patterns,
                bar_width,
                label=strategy,
                color=strategy_styles[strategy]["color"],
                alpha=0.9,
            )
            add_bar_labels(ax2, bars, is_int=True)

        ax2.set_ylabel("Number of Patterns (Log Scale)")
        ax2.set_xlabel("Extension Step")
        ax2.set_yscale("log")
        ax2.set_xticks(x)  # integers now
        ax2.set_xticklabels(extension_steps)

        # Setting up one single, shared legend
        handles, labels = ax1.get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                loc="upper right",
                bbox_to_anchor=(0.95, 0.95),
                fancybox=True,
                shadow=True,
            )

        # Add annotation for BPIC11 early stopping
        if (
            dataset_name == "bpic11_trunc.csv"
            and extension_steps
            and max(extension_steps) == 0
        ):
            ax1.text(
                0.5,
                0.5,
                "Early stopping after Step 0\ndue to computational complexity",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax1.transAxes,
                fontsize=14,
                color="red",
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="red", alpha=0.8),
            )

        # Make layout account for suptitle and xlabel
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plot_filename = f"replication_plot_{dataset_title_name.replace(' ', '_')}.png"
        plot_path = os.path.join(OUTPUT_DIR, plot_filename)
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")
        plt.close()

    # Sophisticated final report
    print("Plots generated.")


if __name__ == "__main__":
    main()
