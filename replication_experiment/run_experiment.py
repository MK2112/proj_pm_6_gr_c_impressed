import os
import sys
import random
import warnings
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from paretoset import paretoset
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

# Assuming the script is run from the project root:
# Adding project's root to the python path to allow imports from the sub-directories:
sys.path.append("./InteractivePatternDetection-main")

from IMIPD import (
    Trace_graph_generator,
    Pattern_extension,
    create_pattern_attributes,
    calculate_pairwise_case_distance,
    Single_Pattern_Extender,
)

N_SPLITS = 5  # Number of folds for cross-validation
MAX_EXTENSION_STEPS = 2  # As per the paper's evaluation graph (steps 0, 1, 2)
DATA_DIR = "replication_experiment/data"
OUTPUT_DIR = "replication_experiment"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATASETS = ["production_trunc.csv", "bpic12_trunc.csv", "bpic11_trunc.csv"]

# Map the standard column names for each dataset
COLUMN_MAPPING = {
    "production_trunc.csv": {
        "case_id": "case:concept:name",
        "activity": "concept:name",
        "timestamp": "Complete Timestamp",
        "label": "label",
    },
    "bpic12_trunc.csv": {
        "case_id": "case",
        "activity": "event",
        "timestamp": "completeTime",
        "label": "label",  # This will be derived
    },
    "bpic11_trunc.csv": {
        "case_id": "case",
        "activity": "event",
        "timestamp": "completeTime",
        "label": "label",  # This will be derived
    },
}

# Defining numerical and categorical attributes for each dataset
DATASET_ATTRIBUTES = {
    "production_trunc.csv": {
        "numerical": [
            "Work_Order_Qty",
            "Qty_Completed",
            "Qty_for_MRB",
            "activity_duration",
            "timesincemidnight",
            "month",
            "weekday",
            "hour",
            "timesincelastevent",
            "timesincecasestart",
            "event_nr",
            "open_cases",
        ],
        "categorical": [
            "Resource",
            "Variant",
            "Part_Desc_",
            "Rework",
            "Report_Type",
            "Resource.1",
        ],
    },
    "bpic12_trunc.csv": {"numerical": ["AMOUNT_REQ"], "categorical": ["org:resource"]},
    "bpic11_trunc.csv": {
        "numerical": ["Age", "Number of executions"],
        "categorical": [
            "Specialism code",
            "Diagnosis",
            "Treatment code",
            "Activity code",
            "Section",
            "Producer code",
            "org:group",
        ],
    },
}


def run_single_fold(train_df, test_df, dataset_name, fold_num):
    # Run the pattern discovery and evaluation for a single fold of cross-validation
    print(f"\n--- Running Fold {fold_num + 1}/{N_SPLITS} for {dataset_name} ---")
    mapping = COLUMN_MAPPING[dataset_name]
    case_id_col, activity_col, timestamp_col, label_col = (
        mapping["case_id"],
        mapping["activity"],
        mapping["timestamp"],
        mapping["label"],
    )
    all_attrs = DATASET_ATTRIBUTES[dataset_name]
    numerical_attrs = all_attrs["numerical"]
    categorical_attrs = all_attrs["categorical"]

    # Create patient-level data for this fold (train and test, to encode pattern frequencies)
    train_patient_data = (
        train_df[[case_id_col, label_col] + numerical_attrs + categorical_attrs]
        .drop_duplicates(subset=[case_id_col])
        .reset_index(drop=True)
    )
    test_patient_data = (
        test_df[[case_id_col, label_col] + numerical_attrs + categorical_attrs]
        .drop_duplicates(subset=[case_id_col])
        .reset_index(drop=True)
    )

    print("Calculating pairwise case distances...")
    combined_patient_data = pd.concat(
        [train_patient_data, test_patient_data]
    ).reset_index(drop=True)
    X_features = combined_patient_data[numerical_attrs + categorical_attrs].copy()

    # The original implementation expects all columns to be present. We will add placeholder columns if they are missing.
    for col in numerical_attrs + categorical_attrs:
        if col not in X_features.columns:
            X_features[col] = 0

    pairwise_distances_array = calculate_pairwise_case_distance(
        X_features, numerical_attrs
    )

    # pair_cases must contain all pairs from combined_patient_data to match pairwise_distances_array structure (pdist output)
    pair_cases = [
        (i, j)
        for i in range(len(combined_patient_data))
        for j in range(i + 1, len(combined_patient_data))
    ]
    start_search_points = {
        i: int(i * (len(combined_patient_data) - (i + 1) / 2))
        for i in range(len(combined_patient_data))
    }
    print("Pairwise distances calculated.")

    ###
    # Experiment Setup
    ###
    pareto_features = [
        "Frequency_Interest",
        "Outcome_Interest",
        "Case_Distance_Interest",
    ]

    # maximize Freq & Outcome, Minimize Distance
    pareto_sense = ["max", "max", "min"]
    delta_time, max_gap = 1.0, 1
    # random colors for visualization of activities seems best for now, works ig
    color_act_dict = {
        act: f"#{random.randint(0, 0xFFFFFF):06x}"
        for act in train_df[activity_col].unique()
    }

    all_fold_results = []
    foundational_patterns = list(train_df[activity_col].unique())
    all_discovered_patterns_df = pd.DataFrame()
    all_extended_patterns_dict = {}

    for ext_step in range(MAX_EXTENSION_STEPS + 1):
        # Early Stop for bpic11_trunc.csv data, computationally *really* expensive to run (tested for 12h, cancelled run)
        #if dataset_name == "bpic11_trunc.csv" and ext_step > 0:
        #    print(
        #        f"Early stopping for {dataset_name} after Extension Step {ext_step - 1} due to computational complexity."
        #    )
        #    break

        print(f"\n>> Extension Step {ext_step}")

        if not foundational_patterns:
            print("No foundational patterns to extend. Stopping.")
            break

        current_patterns_list, newly_extended_patterns_dict = (
            (foundational_patterns, {})
            if ext_step == 0
            else extend_patterns(
                foundational_patterns,
                all_extended_patterns_dict,
                train_df,
                delta_time,
                max_gap,
                color_act_dict,
                case_id_col,
                activity_col,
                timestamp_col,
            )
        )

        if not current_patterns_list:
            print("No new patterns discovered. Stopping extension.")
            break

        all_extended_patterns_dict.update(newly_extended_patterns_dict)

        # Encoding patterns, calculating interest funcs
        encode_pattern_frequencies(
            train_patient_data,
            train_df,
            current_patterns_list,
            all_extended_patterns_dict,
            activity_col,
            case_id_col,
            ext_step == 0,
        )
        encode_pattern_frequencies(
            test_patient_data,
            test_df,
            current_patterns_list,
            all_extended_patterns_dict,
            activity_col,
            case_id_col,
            ext_step == 0,
        )

        print(
            f"Calculating interest functions for {len(current_patterns_list)} patterns..."
        )
        patterns_df = create_pattern_attributes(
            train_patient_data,
            label_col,
            current_patterns_list,
            pairwise_distances_array,
            pair_cases,
            start_search_points,
            "binary",
        ).fillna(0)

        all_discovered_patterns_df = (
            pd.concat([all_discovered_patterns_df, patterns_df])
            .drop_duplicates(subset=["patterns"])
            .reset_index(drop=True)
        )

        strategies, k = create_feature_sets(
            all_discovered_patterns_df, pareto_features, pareto_sense
        )

        ###
        # Evaluation
        ###
        for name, p_set in strategies.items():
            if not p_set:
                continue
            f1 = evaluate_strategy(
                train_patient_data, test_patient_data, p_set, label_col
            )
            all_fold_results.append(
                {
                    "dataset": dataset_name,
                    "fold": fold_num,
                    "extension_step": ext_step,
                    "strategy": name,
                    "f1_score": f1,
                    "num_patterns": len(p_set),
                }
            )
            print(
                f"  - Strategy: {name:<20} | Patterns: {len(p_set):<5} | F1-Score: {f1:.4f}"
            )

        foundational_patterns = strategies.get("Pareto", [])

    return all_fold_results


def extend_patterns(
    foundational, all_patterns_dict, train_df, delta, max_gap, colors, case_id, act, ts
):
    # Extends a set of foundational patterns
    # God this is awful, but I need it for my laptop
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, desc=None: x

    print(f"Generating trace graphs for {len(train_df[case_id].unique())} cases...")
    graphs = {}
    # Split dict comprehension to allow tqdm visualization
    unique_cases = train_df[case_id].unique()
    for case in tqdm(unique_cases, desc="Generating Graphs"):
        graphs[case] = Trace_graph_generator(
            train_df[train_df[case_id] == case], delta, case, colors, case_id, act, ts
        )

    extended_dict = {}
    new_pattern_ids = []

    print(f"Extending {len(foundational)} foundational patterns...")
    for pattern_id in tqdm(foundational, desc="Extending Patterns"):
        if (
            pattern_id in all_patterns_dict
            and "Instances" in all_patterns_dict[pattern_id]
        ):  # Complex patter (i.e. not single activity)
            if any(
                nx.get_edge_attributes(
                    all_patterns_dict[pattern_id]["pattern"], "eventually"
                ).values()
            ):
                continue
            _, extended, _ = Single_Pattern_Extender(
                all_patterns_dict,
                pattern_id,
                pd.DataFrame(),
                graphs,
                train_df,
                max_gap,
                act,
                case_id,
            )
            extended_dict.update(extended)
            new_pattern_ids.extend(extended.keys())
        else:  # Simple activity (single activity pattern)
            core_activity = pattern_id.split("_")[0]
            cases_with_core = train_df[train_df[act] == core_activity][case_id].unique()
            for case in cases_with_core:
                case_data = train_df[train_df[case_id] == case]
                extended_dict, new_ids = Pattern_extension(
                    case_data,
                    graphs[case],
                    pattern_id,
                    case_id,
                    extended_dict,
                    max_gap,
                    new_patterns_for_core=[],
                )
                if new_ids:
                    new_pattern_ids.extend(new_ids)
    return list(set(new_pattern_ids)), extended_dict


def encode_pattern_frequencies(
    patient_df, event_df, patterns, patterns_dict, activity_col, case_id_col, is_simple
):
    # Encodes the frequency of patterns for each case
    # Ensure patient_df is indexed by case_id for efficient mapping
    if not patient_df.index.name == case_id_col:
        patient_df.set_index(case_id_col, inplace=True)

    for pattern in patterns:
        if is_simple:
            # Foundational activities
            counts = (
                event_df[event_df[activity_col] == pattern].groupby(case_id_col).size()
            )
        else:
            # Extended patterns from dictionary
            if pattern in patterns_dict and "Instances" in patterns_dict[pattern]:
                counts = pd.Series(
                    patterns_dict[pattern]["Instances"]["case"]
                ).value_counts()
            else:
                counts = pd.Series()
        # Update the column in the correctly indexed dataframe
        patient_df[pattern] = counts.reindex(patient_df.index, fill_value=0)
    patient_df.reset_index(inplace=True)


def create_feature_sets(patterns_df, pareto_features, pareto_sense):
    # Creating different feature sets based on selection strategies
    if patterns_df.empty:
        return {}, 0
    patterns_all = patterns_df["patterns"].tolist()
    mask = paretoset(patterns_df[pareto_features], sense=pareto_sense)
    patterns_pareto = patterns_df[mask]["patterns"].tolist()
    k = (
        len(patterns_pareto)
        if len(patterns_pareto) > 0
        else int(0.1 * len(patterns_all))
    )
    if k == 0 and len(patterns_all) > 0:
        k = 1

    # Isn't this nicely formatted and not at all rushed at 2 AM?
    return {
        "All": patterns_all,
        "Pareto": patterns_pareto,
        "TopK_Outcome": patterns_df.nlargest(k, "Outcome_Interest")[
            "patterns"
        ].tolist(),
        "TopK_Frequency": patterns_df.nlargest(k, "Frequency_Interest")[
            "patterns"
        ].tolist(),
        "TopK_CaseDistance": patterns_df.nsmallest(k, "Case_Distance_Interest")[
            "patterns"
        ].tolist(),
    }, k


def evaluate_strategy(train_data, test_data, features, label_col):
    # Training a Decision Tree and evaluating it using F1-score

    # Ensure dataframes are aligned and only contain necessary columns
    X_train = train_data[features]  # Features only
    y_train = train_data[label_col]  # Labels only
    X_test = test_data[features]  # Features only
    y_test = test_data[label_col]  # Guess what, labels only

    # Trusty Decision Tree from sklearn
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return f1_score(y_test, y_pred, average="weighted", zero_division=0)


def add_bar_labels(ax, bars, is_int=False):
    # Attach a text label above each bar in *bars*, displaying its height
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            label = f"{int(height)}" if is_int else f"{height:.2f}"
            ax.annotate(
                label,
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=18,
            )


def main():
    all_results = []
    for dataset_name in DATASETS:
        print(
            f"\n==================== Starting Dataset: {dataset_name} ===================="
        )
        file_path = os.path.join(DATA_DIR, dataset_name)
        if not os.path.exists(file_path):
            print(f"Dataset {dataset_name} not found. Skipping.")
            continue

        df = pd.read_csv(file_path, low_memory=False)
        mapping = COLUMN_MAPPING[dataset_name]

        # Data-specific Preprocessing and Labeling
        if dataset_name == "production_trunc.csv":
            df[mapping["label"]] = (df[mapping["label"]] != "regular").astype(int)
        elif dataset_name == "bpic12_trunc.csv":
            declined_cases = df[df[mapping["activity"]] == "A_DECLINED"][
                mapping["case_id"]
            ].unique()
            df[mapping["label"]] = (
                df[mapping["case_id"]].isin(declined_cases).astype(int)
            )
        elif dataset_name == "bpic11_trunc.csv":
            # Paper uses e.g. 'diagnosis' column specifically, so we do so too
            # as the target for outcome-oriented pattern discovery
            # Taking the first non-empty diagnosis per case
            case_diagnosis = df.groupby(mapping['case_id'])['Diagnosis'].first()
            case_diagnosis = case_diagnosis.fillna('Unknown')
            df[mapping['label']] = df[mapping['case_id']].map(case_diagnosis)
            print(f"BPIC11: Mapped 'Diagnosis' to outcome label (Multi-class).")

        df[mapping["timestamp"]] = pd.to_datetime(df[mapping["timestamp"]])
        df = df.sort_values(mapping["timestamp"]).reset_index(drop=True)

        X = df[[mapping["case_id"]]].drop_duplicates()
        y = df.groupby(mapping["case_id"])[mapping["label"]].first()

        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            train_cases = X.iloc[train_idx][mapping["case_id"]]
            test_cases = X.iloc[test_idx][mapping["case_id"]]

            fold_results = run_single_fold(
                df[df[mapping["case_id"]].isin(train_cases)].copy(),
                df[df[mapping["case_id"]].isin(test_cases)].copy(),
                dataset_name,
                fold,
            )
            all_results.extend(fold_results)

    if not all_results:
        print("No results generated. Exiting.")
        return

    results_df = pd.DataFrame(all_results)
    agg_df = (
        results_df.groupby(["dataset", "extension_step", "strategy"])
        .agg(
            mean_f1=("f1_score", "mean"),
            std_f1=("f1_score", "std"),
            num_patterns=("num_patterns", "mean"),
        )
        .reset_index()
        .fillna(0)
    )

    raw_results_path = os.path.join(OUTPUT_DIR, "replication_results_raw.csv")
    agg_results_path = os.path.join(OUTPUT_DIR, "replication_results.csv")
    results_df.to_csv(raw_results_path, index=False)
    agg_df.to_csv(agg_results_path, index=False)
    print(f"\nRaw results saved to {raw_results_path}")
    print(f"Aggregated results saved to {agg_results_path}")

    # Onwards to the plotting!
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

    # Colors from from_csv.py
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

        strategies = [
            s for s in strategy_styles.keys() if s in dataset_df["strategy"].unique()
        ]
        extension_steps = sorted(dataset_df["extension_step"].unique())

        if not extension_steps:
            print(f"No data to plot for {dataset_name}. Skipping plot generation.")
            continue

        x = np.arange(len(extension_steps))
        n_strategies = len(strategies)
        bar_width = 0.8 / n_strategies if n_strategies > 0 else 0.8

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
        fig.suptitle(
            f"{dataset_title_name} Dataset: Performance vs. Complexity",
            fontsize=18,
            fontweight="bold",
        )

        # Predictive Performance (F1-Score)
        ax1.set_title("Predictive Performance (F1-Score)", pad=15)
        for i, strategy in enumerate(strategies):
            subset = (
                dataset_df[dataset_df["strategy"] == strategy]
                .set_index("extension_step")
                .reindex(extension_steps)
            )

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
            ax1.set_ylim(
                top=max(1.0, (dataset_df["mean_f1"] + dataset_df["std_f1"]).max() * 1.1)
            )

        # Model Complexity (Number of Patterns)
        ax2.set_title("Model Complexity (Number of Patterns)", pad=15)
        for i, strategy in enumerate(strategies):
            subset = (
                dataset_df[dataset_df["strategy"] == strategy]
                .set_index("extension_step")
                .reindex(extension_steps)
            )

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
        ax2.set_xticks(x)
        ax2.set_xticklabels(extension_steps)

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

        # Mark BPIC11's early stopping
        if dataset_name == "bpic11_trunc.csv" and max(extension_steps) == 0:
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

        plt.tight_layout(
            rect=[0, 0.03, 1, 0.95]
        )  # This looks so funny, Ruff linter doesn't like it
        plot_filename = f"replication_plot_{dataset_title_name.replace(' ', '_')}.png"
        plot_path = os.path.join(OUTPUT_DIR, plot_filename)
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")
        plt.close()


if __name__ == "__main__":
    main()
