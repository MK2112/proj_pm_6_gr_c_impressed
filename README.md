# Process Mining - Project 6 (IMPresseD)

## Setup

1. Ensure you have Python 3.11 and all required packages installed. Ideally use [Miniconda](https://repo.anaconda.com/miniconda/).
```python
conda create --name impressed python=3.11
conda activate impressed
pip install -r requirements.txt
# Run the replication experiment from the project root directory:
python replication_experiment/run_experiment.py
```

Our replication script is configured to use the three datasets located in `replication_experiment/data/`: `production_trunc.csv`, `bpic12_trunc.csv`, and `bpic11_trunc.csv`. The script `replication_experiment/run_experiment.py` will automatically handle the preprocessing and label derivation.

The latter is the core reason for this script. The original project GUI's role ends with data preparation. It does not calculate performance metrics like the F1-score from training a decision tree classifier. The `run_experiment.py` script trains and tests the model 5 times on different subsets of the data and aggregates the results, intended to serve robust results while not being too expensive.

For `production_trunc.csv`:
- **Case ID column**: `case:concept:name` (index 0)
- **Activity column**: `concept:name` (index 1)
- **Outcome column**: `label` (index 9)
- **Outcome type**: *Binary*
- **Timestamp column**: `Complete Timestamp` (index 3)
- **Delta time (in second)**: `activity_duration` (index 14)

For `bpic11_trunc.csv`:
- **Case ID column**: `case` (index 0)
- **Activity column**: `event` (index 1)
- **Outcome column**: `Diagnosis` (index 20) (we deemed most likely fit, as its representing the patient's diagnosis)
- **Outcome type**: *Categorical*
- **Timestamp column**: `completeTime` (index 3)
- **Delta time (in second)**: *Not available*, the timestamps in this dataset only contain date information, not time, so a delta time could not be calculated

For `bpic12_trunc.csv`:
- **Case ID column**: `case` (index 0)
- **Activity column**: `event` (index 1)
- **Outcome column**: Derived from `event` column via inspecting the last event of each case (e.g., 'A_ACCEPTED', 'A_DECLINED')
- **Outcome type**: *Binary*
- **Timestamp column**: `completeTime` (index 3)
- **Delta time (in second)**: Calculated as the difference between `completeTime` and `startTime`