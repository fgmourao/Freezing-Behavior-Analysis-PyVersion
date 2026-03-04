# Freezing Behavior Analysis Pipeline

A Python pipeline for automated detection and quantification of freezing behavior from raw movement signals (threshold voltages from load cells and/or video-based threshold analysis, e.g., MED-PC boxes and/or VideoFreeze systems). Supports multi-subject batch processing, block-averaged analyses, and interactive visualization through a graphical interface.

---

## Overview

This pipeline processes raw movement signals recorded during fear conditioning or related behavioral paradigms. Each session is segmented into user-defined epochs (baseline, CS, ITI, or any custom event), and freezing bouts are detected within each epoch using a threshold-and-duration criterion. All detection is performed globally on the full session first and then mapped to individual epochs, ensuring that bout identities are consistent across the entire recording.

The pipeline runs either interactively through a GUI (`app_behavior.py`) or programmatically by calling `behavior_analyse` directly, making it suitable for both exploratory use and automated processing of large datasets. An auxiliary tool, `BehaviorSync.py`, is a companion standalone GUI for video-synchronized event annotation. It can be launched from within `app_behavior.py` via the main menu or independently.

---

## Repository Structure

```
.
├── app_behavior.py         GUI control center for the full pipeline
├── BehaviorSync.py         Standalone GUI for video / neural synchronization
│                           and manual behavioral event extraction
├── behavior_analyse.py     Core analysis function (multi-subject, block analysis)
├── detect_bouts.py         Low-level bout detection from a binary signal
└── plot_behavior_batch.py  Figure generation and export
```

---

## Dependencies

### Required Libraries

| Library | Version | Purpose |
|---|---|---|
| `numpy` | ≥ 1.26, < 2.0 | Array operations, signal processing |
| `pandas` | ≥ 1.3 | Excel export, events file loading |
| `matplotlib` | ≥ 3.4 | Figure generation and export |
| `scipy` | ≥ 1.7 | Signal smoothing (`uniform_filter1d`) |
| `openpyxl` | ≥ 3.0 | Writing `.xlsx` files |
| `PyQt5` | ≥ 5.15 | Graphical user interface |
| `pyqtgraph` | ≥ 0.13 | Real-time signal plots (BehaviorSync only) |
| `opencv-python` | < 4.10 | Video decoding and frame display (BehaviorSync only) |

> **Note:** `pyqtgraph` and `opencv-python` are only required if you intend to use `BehaviorSync.py`. The core analysis pipeline (`app_behavior.py`, `behavior_analyse.py`) runs without them.

### Installation

It is recommended to use a dedicated virtual environment or conda environment.

**Using pip (recommended):**

```bash
pip install "numpy>=1.26,<2.0" pandas matplotlib scipy openpyxl PyQt5 pyqtgraph "opencv-python<4.10"
```

Or install all dependencies at once from the requirements file:

```bash
pip install -r requirements.txt
```

**Using conda:**

```bash
conda install "numpy>=1.26,<2.0" pandas matplotlib scipy openpyxl pyqt
pip install pyqtgraph "opencv-python<4.10"
```

### Inter-file Dependencies

All files must reside in the same directory (or be present on the Python path):

- `detect_bouts.py` is required by `behavior_analyse.py`.
- `behavior_analyse.py` and `plot_behavior_batch.py` are required by `app_behavior.py`.
- `BehaviorSync.py` is launched as a standalone window from **Tools > BehaviorSync Interface** inside `app_behavior.py`, and can also be run independently.

Tested on Python 3.9 and later.

---

## Quick Start

### Interactive Mode (GUI)

```bash
python app_behavior.py
```

1. Use **File > Load Data Files** to select one or more raw movement files (`.out`, `.txt`, or `.csv`).
2. *(Optional)* Use **Tools > BehaviorSync Interface** to load the corresponding video alongside neural or behavioral recordings, navigate frame by frame, and visually annotate event onsets and offsets. Export the result as `<file>_events.csv` and load it directly into the Events table in step 3.
3. Load or manually fill the Events table (columns: Event Label, Onset in seconds, Offset in seconds).
4. Set basic parameters (sampling rate, freeze threshold, minimum duration, baseline duration) and up to five block grouping definitions.
5. Click **RUN ANALYSIS**.
6. Use **File > Export Results** or the **Plot Viewer** panel to inspect and export results.

### BehaviorSync (Standalone)

```bash
python BehaviorSync.py
```

Launches the video synchronization and event annotation GUI independently, with no dependency on `app_behavior.py` or any other pipeline function. See the [BehaviorSync](#behaviorsync) section for full details.

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

## BehaviorSync

BehaviorSync is a **standalone** companion GUI for synchronized visualization and annotation of behavioral events directly from video, alongside simultaneously recorded neural or behavioral signals (e.g., LFP, Fiber Photometry, load cell, VideoFreeze/MED-PC output).

It can be launched independently from the command line or from within `app_behavior.py` via **Tools > BehaviorSync Interface**:

```bash
python BehaviorSync.py
```

### Features

**Video playback**
- Load and play video files (`.mp4`, `.avi`, `.wmv`, `.mov`).
- Adjustable playback speed: 0.25×, 0.5×, 1×, 2×, 4×, 10×, 20×.
- Frame-by-frame navigation via arrow keys or the timeline slider.

**Signal visualization**
- Load neural and behavioral recordings independently (`.csv` or `.txt`). The last column is used as the signal; if two columns are present, the first is treated as the time series. Only one experimental subject per file is supported.
- Time vectors are built automatically from sample count and user-supplied Fs — input files do **not** require a time column.
- Scrolling time-window view on both signal axes, synchronized with video playback.
- Adjustable time window width and independent Y-axis scaling for each signal.
- Red cursor line tracks video time in real time on both signal axes.

**Event annotation**
- Mark Onset and Offset events frame by frame via buttons or keyboard shortcuts.
- Smart toggle key [M]: marks Onset if pairs are balanced, Offset otherwise.
- Live list widgets showing all marked onsets, offsets, and computed durations.
- Delete last marked event at any time with [Del].
- Export annotated events to CSV — fully compatible with the Events table in `app_behavior.py`.

**Epoch-based analysis**
- Define analysis epochs manually in the built-in table (Label | Onset | Offset) or load them from a 3-column `.csv` / `.txt` file.
- Epoch boundaries are drawn as dashed blue lines on both signal axes for visual reference.
- A "Full Session" epoch is always included automatically.
- Epochs can be named freely to match the experimental design — typical examples include `Baseline`, `CS1`, `ITI1`, `CS2`, or any custom label. Each named epoch is analyzed independently, so trial-by-trial metrics (e.g., freezing on CS1 vs CS2, or across ITI periods) are directly available in the output without any post-processing.
- Run Analysis computes the following metrics per epoch, using the manually marked bouts:

| Metric | Description |
|---|---|
| Freezing Percentage | Total time in behavioral bouts / epoch duration (%) |
| Total Bouts | Number of discrete behavioral episodes |
| Mean Bout Duration | Average duration of individual bouts (s) |
| Bout Duration (raw) | All individual bout durations (s) |
| Mean Inter-Bout Interval (ΔT) | Average gap between consecutive bouts (s) |
| Inter-Bout Interval (raw) | All individual inter-bout intervals (s) |

### Output Files

| Format | Content | How to export |
|---|---|---|
| `.csv` | Annotated event timestamps (frame + time, onset/offset/duration) with metadata header | File > Export Behavior Timestamps |
| `.xlsx` | One sheet per metric, rows = epochs, one column per subject | File > Export Results |
| `.pkl` | Dictionary `analysis_results` with all computed metrics (pickle format) | File > Save Results |

**CSV format:**
```
# BehaviorSync export | Video: 30.0000 fps | Neural Fs: 1000 Hz | Behavior Fs: 5 Hz
Frame (sample) onset, Frame (sample) offset, Onset (seconds), Offset (seconds), Duration (seconds)
```

### Keyboard Shortcuts

| Key | Action |
|---|---|
| Space | Play / Pause |
| I | Mark Onset |
| O | Mark Offset |
| M | Smart toggle (Onset if balanced, Offset otherwise) |
| ← / → | Step one frame backward / forward |
| Del | Delete last marked event |

### Known Limitations

> Synchronization between signals with different recording start times is not yet supported. A Time Offset field per signal is planned for a future version.

---

## Author

Flavio Mourao (mourao.fg@gmail.com)

Maren Lab, Department of Psychological and Brain Sciences, Texas A&M University  
Beckman Institute, University of Illinois Urbana-Champaign  
Federal University of Minas Gerais, Brazil

Development started: December 2023  
Last update: March 2026

- `detect_bouts.py` is required by `behavior_analyse.py`.
- `behavior_analyse.py` and `plot_behavior_batch.py` are required by `app_behavior.py`.
- `BehaviorSync.py` is launched as a standalone window from **Tools > BehaviorSync Interface** inside `app_behavior.py`, and can also be run independently.

Tested on Python 3.9 and later.

---

## Quick Start

### Interactive Mode (GUI)

```bash
python app_behavior.py
```

1. Use **File > Load Data Files** to select one or more raw movement files (`.out`, `.txt`, or `.csv`).
2. Load or manually fill the Events table (columns: Event Label, Onset in seconds, Offset in seconds).
   - *Tip: Use **Tools > BehaviorSync Interface** to visually extract onset/offset times directly from a video recording.*
3. Set basic parameters (sampling rate, freeze threshold, minimum duration, baseline duration) and up to five block grouping definitions.
4. Click **RUN ANALYSIS**.
5. Use **File > Export Results** or the **Plot Viewer** panel to inspect and export results.

### BehaviorSync (Video + Signal Visualization)

```bash
python BehaviorSync.py
```

Or launch from within the main GUI via **Tools > BehaviorSync Interface**.

1. Set Fs (Hz) for each recording **before** loading the file.
2. Load Video → Load Neural → Load Behavior (order is flexible).
3. Use Play/Pause [Space bar] or arrow keys to navigate the video.
4. Mark Onset **[I]** and Offset **[O]** or (M] 2x for both at the desired frames.
5. Export timestamps via **File > Export Behavior Timestamps (.csv)**.
6. Use **File > Export Results** 

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

### Neural and Behavioral Recordings (BehaviorSync)

Each file must contain one experimental subject. The last column is always used as the signal, making the format robust to files with non-numeric leading columns. If two columns are present, the first is treated as the time series; otherwise, time is reconstructed automatically from the user-supplied sampling frequency (Fs). Accepted formats: `.csv`, `.txt`.

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

### BehaviorSync Timestamp Export

`BehaviorSync.py` exports a separate `.csv` file (`<file>_events.csv`) containing behavioral events extracted through visual inspection of the video. The file includes a metadata header (video fps, neural Fs, behavior Fs) and the following columns: Frame onset | Frame offset | Onset (s) | Offset (s) | Duration (s). The exported onset/offset times in seconds can be loaded directly into the Events table of `app_behavior.py`.

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
Last update: March 2026
