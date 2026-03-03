# Freezing Behavior Analysis Pipeline

A Python pipeline for automated detection and quantification of freezing behavior from raw movement signals (from threshold voltages load cells and/or video threshold analysis, e.g. MED-PC boxes, and/or VideoFreeze systems). Supports multi-subject batch processing, block-averaged analyses, and interactive visualization through a graphical interface.

---

## Overview

This pipeline processes raw movement signals recorded during fear conditioning or related behavioral paradigms. Each session is segmented into user-defined epochs (baseline, CS, ITI, or any custom event), and freezing bouts are detected within each epoch using a threshold-and-duration criterion. All detection is performed globally on the full session first and then mapped to individual epochs, ensuring that bout identities are consistent across the entire recording.

The pipeline runs either interactively through a GUI (`App_Behavior.py`) or programmatically by calling `behavior_analyse` directly, making it suitable for both exploratory use and automated processing of large datasets.

---

## Repository Structure

```
.
├── App_Behavior.py         GUI control center for the full pipeline
├── behavior_analyse.py     Core analysis function (multi-subject, block analysis)
├── detect_bouts.py         Low-level bout detection from a binary signal
└── plot_behavior_batch.py  Figure generation and export
```

---

## Dependencies

### Required Libraries

| Library | Version | Purpose |
|---|---|---|
| `numpy` | ≥ 1.21 | Array operations, signal processing |
| `pandas` | ≥ 1.3 | Excel export, events file loading |
| `matplotlib` | ≥ 3.4 | Figure generation and export |
| `scipy` | ≥ 1.7 | Signal smoothing (`uniform_filter1d`) |
| `openpyxl` | ≥ 3.0 | Writing `.xlsx` files |
| `PyQt5` | ≥ 5.15 | Graphical user interface |

### Installation

It is recommended to use a dedicated virtual environment or conda environment.

**Using pip:**

```bash
pip install numpy pandas matplotlib scipy openpyxl PyQt5
```

**Using conda:**

```bash
conda install numpy pandas matplotlib scipy openpyxl pyqt
```

### Inter-file Dependency

`detect_bouts.py` must be importable from the same directory as `behavior_analyse.py`, or present on the Python path. No other cross-file dependencies exist.

Tested on Python 3.9 and later.

---

## Quick Start

### Interactive Mode (GUI)

```bash
python App_Behavior.py
```

1. Use **File > Load Data Files** to select one or more raw movement files (`.out`, `.txt`, or `.csv`).
2. Load or manually fill the Events table (columns: Event Label, Onset in seconds, Offset in seconds).
3. Set basic parameters (sampling rate, freeze threshold, minimum duration, baseline duration) and up to five block grouping definitions.
4. Click **RUN ANALYSIS**.
5. Use **File > Export Results** or the **Plot Viewer** panel to inspect and export results.

### Standalone Analysis

```python
from behavior_analyse import behavior_analyse

params = {
    'fs':           5,        # sampling rate (Hz)
    'thr_low':      10,       # freeze threshold (% movement)
    'thr_dur':      1,        # minimum freeze duration (s)
    'baseline_dur': 180,      # baseline duration (s)
    'events_sec':   [[180, 190], [190, 250]],
    'event_names':  ['CS1', 'ITI1'],
}

data, parameters = behavior_analyse(raw_signal, params)
```

### Generating Figures

```python
from plot_behavior_batch import plot_behavior_batch

data_results = {
    'parameters': parameters,
    'my_file':    data
}

plot_behavior_batch(data_results)
```

Figures are saved as 300 dpi PNG files to the user's Desktop.

---

## Input Format

### Raw Data Files

Each data file should be a numeric matrix where:

- **Column 1** is the timestamp (automatically removed during file parsing in the GUI).
- **Remaining columns** are movement signals, one column per subject.

Accepted formats: `.out`, `.txt`, `.csv`. The parser handles comma, semicolon, or space delimiters automatically and skips any non-numeric header lines.

When calling `behavior_analyse` directly, pass a NumPy array of shape `(M, S)` where M is the number of samples and S is the number of subjects. One-dimensional arrays are accepted and reshaped to `(M, 1)` internally.

### Events File

A plain-text or CSV file with exactly three columns and no header row:

```
CS1     180   190
ITI1    190   220
CS2     220   230
```

---

## Parameters

| Parameter | Description | Default (GUI) |
|---|---|---|
| Sampling rate (Hz) | Samples per second of the input signal | 5 |
| Freeze threshold (%) | Samples at or below this normalised movement value are classified as frozen | 10 |
| Min freeze duration (s) | Bouts shorter than this after epoch clipping are discarded | 1 |
| Baseline duration (s) | Duration of the pre-event period (Index 1 in all outputs) | 180 |
| Block prefix | String prefix used to identify events belonging to a block (e.g., `CS`) | — |
| Block size | Number of consecutive matching events to average into one block | — |

The signal is normalised independently per subject to 0–100 % of its own maximum movement before any threshold is applied, making the freeze threshold comparable across animals and sessions.

---

## Output Structure

All results are returned in the `data` dict. List indices follow a consistent epoch convention:

- **Index 0** — Full session (entire recording)
- **Index 1** — Baseline (samples 0 to `baseline_dur × fs − 1`)
- **Indices 2 to N** — Experimental events (one entry per event defined in the events table)

### data['behavior_freezing'] (N epochs × 7 elements)

| Index | Content | Type |
|---|---|---|
| 0 | Raw bout durations | list of S arrays (seconds) |
| 1 | Mean bout duration | S-length array (seconds) |
| 2 | Number of bouts | S-length array (count) |
| 3 | Total freeze time | S-length array (seconds) |
| 4 | Freeze percentage | S-length array (%) |
| 5 | Mean inter-bout Delta T | S-length array (seconds); NaN if fewer than 2 bouts |
| 6 | Raw inter-bout Delta T | list of S arrays (seconds) |

S = number of subjects.

### data['behavior_nonfreezing'] (N epochs × 1 element)

Each entry `[0]` contains a list of S arrays with non-freeze bout durations in seconds, one array per subject.

### data['events_behavior_idx'] (N epochs × 1 element)

Each entry `[0]` contains an S-length list where each subject entry is a 3-element list:

- `[s][0]` — Freeze index pairs: B-by-2 array `[start, end]` in global 0-based sample indices. Contains only bouts that started within this epoch.
- `[s][1]` — Non-freeze index pairs: K-by-2 array `[start, end]` in global 0-based sample indices.
- `[s][2]` — Binary freeze mask: 1-D boolean array for this epoch (True = frozen).

Global indices are directly usable to slice any co-recorded signal of the same length (LFP, pupil, etc.).

### data['behavior_epochs'] (N epochs)

Each entry is an S-by-M NumPy array of the normalised movement signal for the corresponding epoch. Useful for epoch-level visualisation and sanity checks.

### data['blocks'] (List of dicts)

One element per active block definition. Fields:

| Field | Content |
|---|---|
| `prefix` | Matched event prefix string |
| `size` | Block size (number of events per block) |
| `labels` | List of label strings (e.g., `CS 1-5`) |
| `freeze` | S-by-B array of mean freeze percentage per block |
| `bout` | S-by-B array of summed bout count per block |
| `dur` | S-by-B array of mean bout duration per block |
| `delta_t` | S-by-B array of mean inter-bout Delta T per block |

---

## Excel Export

Running the export via the GUI menu produces one workbook per input file, saved in the same folder as the source data.

### Standard Sheets

| Sheet | Content |
|---|---|
| 1_Freezing_Percentage | Freeze % per epoch per subject |
| 2_Total_Bouts | Number of freeze bouts per epoch per subject |
| 3_Mean_Bout_Duration(s) | Mean bout duration per epoch per subject |
| 4_Bout_Duration(s) | All individual bout durations (comma-separated) |
| 5_Mean_Bout_DeltaT(s) | Mean inter-bout Delta T per epoch per subject |
| 6_Bout_DeltaT(s) | All individual Delta T values (comma-separated) |

### Block Sheets

One set of four sheets per active block definition, named using the format `Blk<N>_<Prefix>_<Metric>` (e.g., `Blk1_CS_Freezing_Per`). Sheet names are truncated to comply with Excel's 31-character limit.

### Timestamp Export

A separate workbook (`<file>_Timestamps.xlsx`) contains the raw onset and offset sample indices for every freeze and non-freeze bout across all subjects. Each subject occupies two columns (Onset and Offset), using 0-based global sample indices. This file is suitable for cross-referencing with other simultaneously recorded signals.

### Pickle Export

The full `data_results` dict can be saved to a `.pkl` file via **File > Save Results (.pkl)**. This allows reloading results into Python without rerunning the analysis:

```python
import pickle

with open('Behavior_Results.pkl', 'rb') as f:
    data_results = pickle.load(f)
```

---

## Figures

Calling `plot_behavior_batch(data_results)` or using the GUI's Plot Viewer generates one figure per file. Figure height scales automatically with the number of active block analyses.

- **Row 0** — Full-session smoothed movement traces (individual subjects in grey, group median in black), freeze threshold reference line, and per-subject freeze raster above the movement axis.
- **Row 1** — Event-by-event freeze percentage (individual subjects + mean ± SEM) and three summary pie charts (total bout count, mean bout duration, and mean Delta T across all epochs).
- **Rows 2 and beyond** — One row per block analysis: block-averaged freeze percentage line plot and three pie charts summarising bouts, duration, and Delta T by block.

Figures are exported as 300 dpi PNG files to the user's Desktop.

---

## Freeze Detection Logic

Detection follows a two-step approach designed to ensure consistency across epochs:

1. A binary freeze mask is computed for the full session by thresholding the normalised signal at `thr_low`. `detect_bouts` then finds all contiguous runs of frozen samples that meet the minimum duration criterion.
2. During per-epoch processing, each globally detected bout is intersected with the epoch boundaries. Bouts that straddle a boundary are clipped to the epoch. Clipped fragments shorter than the minimum duration are discarded. Statistics (total freeze time, percentage) use clipped durations from all valid bouts, including those that started in a previous epoch, so that freeze time is never under-reported at epoch boundaries. However, `events_behavior_idx` records only bouts that started within the epoch, preventing double-counting across epochs.

> **Note on indexing:** All sample indices throughout the pipeline are **0-based** (Python convention), unlike the original MATLAB version which used 1-based indexing.

---

## Block Analysis

Block analysis groups consecutive events that share a common name prefix (e.g., all events starting with `CS`) into windows of a fixed size and computes aggregate statistics per window. Up to five independent block definitions can be active simultaneously.

Aggregation rules:

- Freeze percentage — mean across events in the block
- Bout count — sum across events (total bouts in the block)
- Mean bout duration — `nanmean` across events (set to 0 if all values are NaN)
- Mean Delta T — `nanmean` across events (set to 0 if all values are NaN)

Block labels follow the format `<Prefix> <first>-<last>` (e.g., `CS 1-5`, `CS 6-10`).

---

## Author

Flavio Mourao (mourao.fg@gmail.com)

Maren Lab, Department of Psychological and Brain Sciences, Texas A&M University  
Beckman Institute, University of Illinois Urbana-Champaign  
Federal University of Minas Gerais, Brazil

Development started: December 2023  
Last update: February 2026
