"""
app_behavior.py - GUI Control Center for the Behavior Analysis Pipeline

DESCRIPTION:
    Graphical User Interface (GUI) control center for the behavior analysis
    pipeline. Provides interactive panels for loading data, configuring
    parameters, running analysis, and exporting results — without requiring
    Qt Designer or .ui files.
    All UI state is managed through the 'app_data' dictionary, which is
    accessible by all methods of the AppBehavior class.

LAYOUT:
    Panel 1 (top-left)   - Basic Parameters (fs, threshold, duration, baseline)
    Panel 2 (top-right)  - Block Analysis definitions (up to 5 prefix/size pairs)
    Panel 3 (middle)     - Events table (editable; loadable from CSV/TXT)
    Panel 4 (bottom)     - Plot Viewer (dropdown + Show Figure button)
    Status bar           - Color-coded feedback label
    Run button           - Triggers the full analysis pipeline
    Menu > File          - Load data, export Excel, save .pkl, save timestamps
    Menu > Tools         - Launch BehaviorSync — standalone GUI for video /
                           neural visualization and behavioral event extraction

WORKFLOW:
    1. Load one or more raw data files via the File menu.
    2. Fill or load the events table (Name, Onset_s, Offset_s).
       * Tip: Use Tools > Open BehaviorSync to visually extract these events
              from video recordings.
    3. Set basic and block parameters in the panels.
    4. Click RUN ANALYSIS.
    5. Use the Plot Viewer or File menu to export results.

OUTPUTS (via File menu after analysis):
    <file>_Results.xlsx      - Freeze metrics per epoch per subject
    <file>_Timestamps.xlsx   - Freeze / non-freeze onset-offset pairs
    Behavior_Results.pkl     - Full data_results dict (pickle format)

    BehaviorSync output (via Tools menu)  [under construction]
    <file>_events.csv        - Behavioral events extracted from video through
                               visual inspection, containing:
                               Row 1  : Metadata (video fps, neural Fs, behavior Fs)
                               Columns: Frame onset | Frame offset |
                                        Onset (s) | Offset (s) | Duration (s)
                               * Exported Onset/Offset times can be loaded
                                 directly into Panel 3.

KNOWN LIMITATIONS:
    - If the behavioral or neural recording does not start at the same
      real-world time as the video, a manual Time Offset (s) field will be
      required for precise alignment. This feature is under development.

NOTE:
    BehaviorSync is still under active development.
    Synchronization of signals with different start times will be addressed
    in future versions.

REQUIRES:
    Internal modules : behavior_analyse.py, plot_behavior_batch.py,
                       BehaviorSync.py

    External packages: Install via pip (recommended: use a virtual environment)

        pip install numpy pandas openpyxl PyQt5

    Pinned versions (for reproducibility):

        numpy>=1.26.0,<2.0.0
        pandas
        openpyxl
        PyQt5

    Or install all at once from the requirements file:

        pip install -r requirements.txt

AUTHOR:
    Flavio Mourao  (mourao.fg@gmail.com)
    Texas A&M University     - Department of Psychological and Brain Sciences
    Beckman Institute / UIUC - University of Illinois Urbana-Champaign
    Federal University of Minas Gerais (UFMG) - Brazil

Started:     12/2023
Last update: 03/2026
"""

import sys
import os
import numpy as np
import pandas as pd
import pickle

from behavior_analyse import behavior_analyse
from plot_behavior_batch import plot_behavior_batch

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QGridLayout, QGroupBox, QLabel,
    QLineEdit, QPushButton, QTableWidget, QTableWidgetItem,
    QFileDialog, QComboBox, QMessageBox
)
from PyQt5.QtCore import Qt


class AppBehavior(QMainWindow):
    """Main application window for the Behavior Analysis pipeline."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Behavior Analysis')
        self.setGeometry(100, 100, 800, 650)

        # Shared application state accessible by all methods
        self.app_data = {
            'file_names':   [],
            'path_name':    '',
            'data_results': {},
            'P':            {}
        }

        # BehaviorSync memory lock
        self.bs_window = None

        self.init_ui()


    # ---------------------------------------------------------------
    #  UI Construction


    def init_ui(self):
        """Build the main window layout and all panels."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        self.create_menu()

        top_panels_layout = QHBoxLayout()
        self.create_basic_parameters_panel(top_panels_layout)
        self.create_block_analysis_panel(top_panels_layout)
        main_layout.addLayout(top_panels_layout)

        self.create_events_table_panel(main_layout)
        self.create_status_and_run_panel(main_layout)
        self.create_plot_viewer_panel(main_layout)

    def create_menu(self):
        """Create the File and Tools menus."""
        menubar   = self.menuBar()
        
        # --- Menu File ---
        file_menu = menubar.addMenu('File')

        load_action = file_menu.addAction('1. Load Data Files (.out, .csv)')
        load_action.triggered.connect(self.load_files)

        # Export and save actions are disabled until analysis completes
        self.export_action = file_menu.addAction('2. Export Results (.xls)')
        self.export_action.setEnabled(False)
        self.export_action.triggered.connect(self.export_excel)

        self.save_pkl_action = file_menu.addAction('3. Save Results (.pkl)')
        self.save_pkl_action.setEnabled(False)
        self.save_pkl_action.triggered.connect(self.save_pkl_file)

        self.save_ts_action = file_menu.addAction('4. Export freeze timestamps (.xls)')
        self.save_ts_action.setEnabled(False)
        self.save_ts_action.triggered.connect(self.save_timestamps)

        # --- Menu Tools ---
        tools_menu = menubar.addMenu('Tools')
        
        behavior_sync_action = tools_menu.addAction('BehaviorSync Interface')
        behavior_sync_action.triggered.connect(self.launch_behavior_sync)

    def create_basic_parameters_panel(self, parent_layout):
        """Basic Parameters panel: sampling rate, threshold, duration, baseline."""
        group_box = QGroupBox("Basic Parameters")
        layout    = QGridLayout()

        layout.addWidget(QLabel("Baseline Dur. (s):"),    0, 0)
        self.edt_base    = QLineEdit("180")
        layout.addWidget(self.edt_base,    0, 1)

        layout.addWidget(QLabel("Min Freeze Dur. (s):"),  1, 0)
        self.edt_min_dur = QLineEdit("1")
        layout.addWidget(self.edt_min_dur, 1, 1)

        layout.addWidget(QLabel("Freeze Threshold (%):"), 2, 0)
        self.edt_thr     = QLineEdit("10")
        layout.addWidget(self.edt_thr,     2, 1)

        layout.addWidget(QLabel("Sampling Rate (Hz):"),   3, 0)
        self.edt_fs      = QLineEdit("5")
        layout.addWidget(self.edt_fs,      3, 1)

        group_box.setLayout(layout)
        parent_layout.addWidget(group_box)

    def create_block_analysis_panel(self, parent_layout):
        """Block Analysis panel: up to 5 prefix/size pairs for block grouping."""
        group_box = QGroupBox("Block Analysis (Prefix / Size)")
        layout    = QGridLayout()

        self.edt_prefixes = []
        self.edt_sizes    = []

        # Default values for the first 3 blocks; rows 4-5 are left empty
        defaults = [("CS", "5"), ("ITI", "5"), ("Trial", "5"), ("", ""), ("", "")]

        for i in range(5):
            layout.addWidget(QLabel(f"Block {i+1}:"), i, 0)

            edt_pref = QLineEdit(defaults[i][0])
            self.edt_prefixes.append(edt_pref)
            layout.addWidget(edt_pref, i, 1)

            edt_size = QLineEdit(defaults[i][1])
            self.edt_sizes.append(edt_size)
            layout.addWidget(edt_size, i, 2)

        group_box.setLayout(layout)
        parent_layout.addWidget(group_box)

    def create_events_table_panel(self, parent_layout):
        """
        Events table panel: editable 3-column table (Name, Onset_s, Offset_s).
        Pre-filled with two example rows; can be loaded from a CSV/TXT file.
        """
        group_box = QGroupBox("Events Definition (Name, Onset_s, Offset_s)")
        layout    = QVBoxLayout()

        btn_load_events = QPushButton("Load Events File (.txt / .csv)")
        btn_load_events.clicked.connect(self.load_events_file)
        layout.addWidget(btn_load_events, alignment=Qt.AlignRight)

        self.table_events = QTableWidget(200, 3)
        self.table_events.setHorizontalHeaderLabels(
            ["Event Label", "Onset (s)", "Offset (s)"]
        )

        # Pre-fill with two example rows so the table is not empty on first launch
        defaults = [("CS1", "180", "190"), ("ITI1", "190", "250")]
        for row, (name, on, off) in enumerate(defaults):
            self.table_events.setItem(row, 0, QTableWidgetItem(name))
            self.table_events.setItem(row, 1, QTableWidgetItem(on))
            self.table_events.setItem(row, 2, QTableWidgetItem(off))

        layout.addWidget(self.table_events)
        group_box.setLayout(layout)
        parent_layout.addWidget(group_box)

    def create_status_and_run_panel(self, parent_layout):
        """Status label (color-coded) and Run Analysis button."""
        layout = QHBoxLayout()

        self.lbl_status = QLabel("Status: Waiting for data files...")
        self.lbl_status.setStyleSheet("color: red; font-weight: bold;")
        layout.addWidget(self.lbl_status)

        # Run button is disabled until at least one data file is loaded
        self.btn_run = QPushButton("RUN ANALYSIS")
        self.btn_run.setEnabled(False)
        self.btn_run.setFixedSize(150, 40)
        self.btn_run.setStyleSheet("font-weight: bold;")
        self.btn_run.clicked.connect(self.run_analysis)
        layout.addWidget(self.btn_run)

        parent_layout.addLayout(layout)

    def create_plot_viewer_panel(self, parent_layout):
        """Plot Viewer panel: dropdown of processed files + Show Figure button."""
        group_box = QGroupBox("Plot Viewer")
        layout    = QHBoxLayout()

        # Dropdown is populated with processed file names after analysis completes
        self.dd_files = QComboBox()
        self.dd_files.addItem("Run analysis first...")
        self.dd_files.setEnabled(False)
        layout.addWidget(self.dd_files)

        self.btn_plot = QPushButton("SHOW FIGURE")
        self.btn_plot.setEnabled(False)
        self.btn_plot.setFixedSize(150, 40)
        self.btn_plot.setStyleSheet("font-weight: bold;")
        self.btn_plot.clicked.connect(self.show_figure)
        layout.addWidget(self.btn_plot)

        group_box.setLayout(layout)
        parent_layout.addWidget(group_box)


    # ---------------------------------------------------------------
    #  Callbacks

    def launch_behavior_sync(self):
        """Instantiates and shows the BehaviorSync interface."""
        try:
            # Importa localmente para evitar carregamento circular ou desnecessário
            from BehaviorSync import BehaviorSync
            
            # Se a janela já existe e está fechada, podemos recriá-la ou apenas mostrá-la
            if self.bs_window is None or not self.bs_window.isVisible():
                self.bs_window = BehaviorSync()
                self.bs_window.show()
            else:
                # Traz para frente se já estiver aberta
                self.bs_window.raise_()
                self.bs_window.activateWindow()
                
        except ImportError as e:
            QMessageBox.critical(
                self, "Import Error", 
                f"Could not load BehaviorSync.\nMake sure 'BehaviorSync.py' is in the same folder as this script.\n\nDetails: {str(e)}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while launching BehaviorSync:\n{str(e)}")

    def load_files(self):
        """Select one or more raw data files and store their paths."""
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select the DATA files", "",
            "Data Files (*.out *.txt *.csv)"
        )
        if not files:
            return

        # Note: path_name is derived from the first selected file.
        # All files are assumed to reside in the same directory.
        self.app_data['file_names'] = [os.path.basename(f) for f in files]
        self.app_data['path_name']  = os.path.dirname(files[0])

        self.lbl_status.setText(f"Status: {len(files)} file(s) loaded. Ready to run.")
        self.lbl_status.setStyleSheet("color: green; font-weight: bold;")
        self.btn_run.setEnabled(True)

    def load_events_file(self):
        """Load event definitions from a CSV or TXT file into the events table."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select the EVENTS file", "",
            "Event Timings File (*.csv *.txt)"
        )
        if not file_path:
            return

        try:
            df = pd.read_csv(file_path, header=None)

            if df.shape[1] < 3:
                QMessageBox.critical(
                    self, "Format Error",
                    "The events file must have 3 columns: Name, Onset, Offset."
                )
                return

            # Clear existing table content and repopulate from file
            self.table_events.clearContents()

            for row in range(len(df)):
                name = str(df.iloc[row, 0])
                on   = str(df.iloc[row, 1])
                off  = str(df.iloc[row, 2])

                if pd.notna(on) and pd.notna(off):
                    self.table_events.setItem(row, 0, QTableWidgetItem(name.strip()))
                    self.table_events.setItem(row, 1, QTableWidgetItem(on))
                    self.table_events.setItem(row, 2, QTableWidgetItem(off))

            self.lbl_status.setText(
                f"Status: Successfully loaded {len(df)} events from file."
            )
            self.lbl_status.setStyleSheet("color: blue; font-weight: bold;")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load events:\n{str(e)}")

    def run_analysis(self):
        """
        Collect parameters from the UI, run behavior_analyse on all loaded
        files, and update the interface with the results.
        """
        self.lbl_status.setText("Status: Running analysis... Please wait.")
        self.lbl_status.setStyleSheet("color: black; font-weight: bold;")
        QApplication.processEvents()

        try:
            # Collect basic parameters
            P = {
                'fs':             float(self.edt_fs.text()),
                'thr_low':        float(self.edt_thr.text()),
                'thr_dur':        float(self.edt_min_dur.text()),
                'baseline_dur':   float(self.edt_base.text()),
                'block_prefixes': [edt.text().strip() for edt in self.edt_prefixes],
                'block_sizes':    []
            }

            # Empty block size fields are stored as NaN and ignored during analysis
            for edt in self.edt_sizes:
                val = edt.text().strip()
                P['block_sizes'].append(float(val) if val else np.nan)

            # Parse events table
            # Keep only rows with a non-empty label and valid numeric onset/offset
            events_sec  = []
            event_names = []

            for row in range(self.table_events.rowCount()):
                it_name = self.table_events.item(row, 0)
                it_on   = self.table_events.item(row, 1)
                it_off  = self.table_events.item(row, 2)

                if it_name and it_on and it_off and it_name.text().strip():
                    try:
                        events_sec.append([float(it_on.text()), float(it_off.text())])
                        event_names.append(it_name.text().strip())
                    except ValueError:
                        continue

            if not events_sec:
                QMessageBox.warning(
                    self, "Input Error",
                    "No valid events found in the table. Fill at least one row."
                )
                self.lbl_status.setText("Status: Ready.")
                return

            P['events_sec']  = events_sec
            P['event_names'] = event_names
            self.app_data['P'] = P

            # Process each file
            data_results = {}

            for i, f_name in enumerate(self.app_data['file_names']):

                file_path = os.path.join(self.app_data['path_name'], f_name)

                # Parse raw data: handle comma, semicolon, or space delimiters.
                # Lines that cannot be fully parsed as floats are skipped (e.g. headers).
                data_rows = []
                with open(file_path, 'r') as f:
                    for line in f:
                        parts = line.replace(',', ' ').replace(';', ' ').split()
                        if not parts:
                            continue
                        try:
                            data_rows.append([float(p) for p in parts])
                        except ValueError:
                            continue

                raw_data = np.array(data_rows)

                # NOTE: timestamp column removal is handled inside behavior_analyse.
                # The line below is intentionally disabled; re-enable if the input
                # files include a leading timestamp column not stripped by the parser.
                # if raw_data.ndim > 1 and raw_data.shape[1] > 1:
                #     raw_data = raw_data[:, 1:]

                data, params_out = behavior_analyse(raw_data, P)

                # Build a safe dict key from the filename (no spaces or hyphens)
                safe_name = os.path.splitext(f_name)[0].replace(" ", "_").replace("-", "_")
                data_results[safe_name] = data

                # Save shared parameters from the first file only
                if i == 0:
                    data_results['parameters'] = params_out

            self.app_data['data_results'] = data_results

            # Update UI after successful analysis
            valid_fields = [k for k in data_results if k != 'parameters']
            self.dd_files.clear()
            self.dd_files.addItems(valid_fields)
            self.dd_files.setEnabled(True)
            self.btn_plot.setEnabled(True)
            self.export_action.setEnabled(True)
            self.save_pkl_action.setEnabled(True)
            self.save_ts_action.setEnabled(True)

            self.lbl_status.setText("Status: Analysis complete! Ready to plot or export.")
            self.lbl_status.setStyleSheet("color: green; font-weight: bold;")

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Analysis Error", f"An error occurred:\n{str(e)}")
            self.lbl_status.setText("Status: Error during analysis.")
            self.lbl_status.setStyleSheet("color: red; font-weight: bold;")

    def show_figure(self):
        """
        Generate and display the summary figure for the selected file.
        Passes a minimal data_results dict containing only the selected file
        so that plot_behavior_batch renders a single-file figure.
        """
        selected_file = self.dd_files.currentText()
        if not selected_file:
            return

        temp_results = {
            'parameters':  self.app_data['P'],
            selected_file: self.app_data['data_results'][selected_file]
        }

        try:
            plot_behavior_batch(temp_results)
            QMessageBox.information(
                self, "Success", f"Plot generated for {selected_file}!"
            )
        except Exception as e:
            QMessageBox.critical(self, "Plot Error", f"Failed to generate plot:\n{str(e)}")

    def save_pkl_file(self):
        """Save the full data_results dictionary to a pickle file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Data as Pickle", "Behavior_Results.pkl",
            "Pickle File (*.pkl)"
        )
        if not file_path:
            return

        with open(file_path, 'wb') as f:
            pickle.dump(self.app_data['data_results'], f)

        self.lbl_status.setText(
            f"Status: Results successfully saved as {os.path.basename(file_path)}!"
        )
        self.lbl_status.setStyleSheet("color: green; font-weight: bold;")

    def export_excel(self):
        """
        Export all freeze metrics to per-file Excel workbooks.

        behavior_freezing index layout (referenced by position below):
            [i][0]  Raw bout durations      (list of arrays, one per subject)
            [i][1]  Mean bout duration      (array, seconds)
            [i][2]  Number of bouts         (array, count)
            [i][3]  Total freeze time       (array, seconds)
            [i][4]  Freeze percentage       (array, %)
            [i][5]  Mean inter-bout Delta T (array, seconds; NaN if < 2 bouts)
            [i][6]  Raw inter-bout Delta T  (list of arrays, one per subject)
        """
        self.lbl_status.setText("Status: Exporting to Excel... Please wait.")
        self.lbl_status.setStyleSheet("color: blue; font-weight: bold;")
        QApplication.processEvents()

        try:
            valid_fields = [k for k in self.app_data['data_results'] if k != 'parameters']
            P = self.app_data['P']

            for f_name in valid_fields:

                data         = self.app_data['data_results'][f_name]
                n_epochs     = len(data['behavior_freezing'])
                num_subjects = len(data['behavior_freezing'][0][1])

                # Build epoch row labels (Full Session, Baseline, then event names)
                epoch_labels = ['Full Session', 'Baseline']
                if 'event_names' in P:
                    epoch_labels.extend(P['event_names'])
                epoch_labels = epoch_labels[:n_epochs]

                subj_cols = [f'Subject {s+1}' for s in range(num_subjects)]

                # Pre-allocate one DataFrame per metric
                df_freeze   = pd.DataFrame(index=epoch_labels, columns=subj_cols)
                df_bouts    = pd.DataFrame(index=epoch_labels, columns=subj_cols)
                df_mean_dur = pd.DataFrame(index=epoch_labels, columns=subj_cols)
                df_raw_dur  = pd.DataFrame(index=epoch_labels, columns=subj_cols)
                df_mean_dt  = pd.DataFrame(index=epoch_labels, columns=subj_cols)
                df_raw_dt   = pd.DataFrame(index=epoch_labels, columns=subj_cols)

                for i in range(n_epochs):

                    # Unpack epoch-level data using the documented index layout above
                    bf_raw_dur  = data['behavior_freezing'][i][0]
                    bf_mean     = data['behavior_freezing'][i][1]
                    bf_num      = data['behavior_freezing'][i][2]
                    bf_freeze   = data['behavior_freezing'][i][4]
                    bf_mean_dt  = data['behavior_freezing'][i][5]
                    bf_raw_dt_i = data['behavior_freezing'][i][6]

                    # Scalar metrics (one value per subject)
                    df_freeze.iloc[i]   = bf_freeze
                    df_bouts.iloc[i]    = bf_num
                    df_mean_dur.iloc[i] = bf_mean
                    df_mean_dt.iloc[i]  = bf_mean_dt

                    # Variable-length arrays: serialised as comma-separated strings
                    for s in range(num_subjects):

                        r_dur = bf_raw_dur[s]
                        df_raw_dur.iloc[i, s] = (
                            '0' if len(r_dur) == 0
                            else ', '.join([f"{x:.2f}" for x in r_dur])
                        )

                        r_dt = bf_raw_dt_i[s]
                        df_raw_dt.iloc[i, s] = (
                            'NaN' if len(r_dt) == 0
                            else ', '.join([f"{x:.2f}" for x in r_dt])
                        )

                # Write standard metric sheets
                out_xlsx = os.path.join(
                    self.app_data['path_name'], f"{f_name}_Results.xlsx"
                )

                with pd.ExcelWriter(out_xlsx, engine='openpyxl') as writer:
                    df_freeze.to_excel(writer,   sheet_name='1_Freezing_Percentage')
                    df_bouts.to_excel(writer,    sheet_name='2_Total_Bouts')
                    df_mean_dur.to_excel(writer, sheet_name='3_Mean_Bout_Duration(s)')
                    df_raw_dur.to_excel(writer,  sheet_name='4_Bout_Duration(s)')
                    df_mean_dt.to_excel(writer,  sheet_name='5_Mean_Bout_DeltaT(s)')
                    df_raw_dt.to_excel(writer,   sheet_name='6_Bout_DeltaT(s)')

                    # Write block analysis sheets (one set per active block).
                    # Sheet names are truncated to stay within Excel's 31-character limit.
                    if 'blocks' in data and data['blocks']:
                        for b_i, block in enumerate(data['blocks']):
                            pref     = block['prefix']
                            b_labels = block['labels']

                            # Block matrices are (subjects x blocks); transpose for Excel rows
                            df_b_freeze = pd.DataFrame(block['freeze'].T,  index=b_labels, columns=subj_cols)
                            df_b_bouts  = pd.DataFrame(block['bout'].T,    index=b_labels, columns=subj_cols)
                            df_b_dur    = pd.DataFrame(block['dur'].T,     index=b_labels, columns=subj_cols)
                            df_b_dt     = pd.DataFrame(block['delta_t'].T, index=b_labels, columns=subj_cols)

                            df_b_freeze.to_excel(writer, sheet_name=f'Blk{b_i+1}_{pref[:10]}_Freezing_Per')
                            df_b_bouts.to_excel(writer,  sheet_name=f'Blk{b_i+1}_{pref[:10]}_Total_Bouts')
                            df_b_dur.to_excel(writer,    sheet_name=f'Blk{b_i+1}_{pref[:10]}_Mean_Bout_Dur')
                            df_b_dt.to_excel(writer,     sheet_name=f'Blk{b_i+1}_{pref[:10]}_Mean_DeltaT')

            self.lbl_status.setText("Status: Excel Export Complete!")
            self.lbl_status.setStyleSheet("color: green; font-weight: bold;")
            QMessageBox.information(
                self, "Export Complete",
                "All Excel files have been successfully exported."
            )

        except Exception as e:
            QMessageBox.critical(
                self, "Export Error",
                f"An error occurred during Excel export:\n{str(e)}"
            )
            self.lbl_status.setText("Status: Error during export.")
            self.lbl_status.setStyleSheet("color: red; font-weight: bold;")

    def save_timestamps(self):
        """
        Export freeze and non-freeze onset-offset pairs to a per-file Excel workbook.

        Output format:
            Each subject occupies 2 columns: Onset | Offset (sample indices).
            Row 1 is the column header; rows 2..N contain the index pairs.
            The number of rows is set by the subject with the most bouts.
            Full-session index pairs (index 0 of events_behavior_idx) are exported.

        events_behavior_idx structure per epoch:
            [epoch][0]  -> list of S subjects, each entry containing:
                [s][0]  Freeze index pairs     (N-by-2 array, global sample indices)
                [s][1]  Non-freeze index pairs (N-by-2 array, global sample indices)
                [s][2]  Binary freeze mask     (1-D boolean array)
        """
        self.lbl_status.setText("Status: Exporting timestamps... Please wait.")
        self.lbl_status.setStyleSheet("color: blue; font-weight: bold;")
        QApplication.processEvents()

        try:
            valid_fields = [k for k in self.app_data['data_results'] if k != 'parameters']

            for f_name in valid_fields:

                data = self.app_data['data_results'][f_name]

                if 'events_behavior_idx' not in data or len(data['events_behavior_idx']) == 0:
                    continue

                # Row 0 holds full-session index pairs (S-length list of subject data)
                base_ts      = data['events_behavior_idx'][0][0]
                num_subjects = len(base_ts)

                # Determine the maximum number of bouts across subjects
                # to set the number of data rows in each export table
                max_f  = max([len(base_ts[s][0]) for s in range(num_subjects)] + [0])
                max_nf = max([len(base_ts[s][1]) for s in range(num_subjects)] + [0])

                # Build column headers: 2 columns per subject (Onset, Offset)
                cols = []
                for s in range(num_subjects):
                    cols.extend([f'Subj {s+1} Onset', f'Subj {s+1} Offset'])

                df_f  = pd.DataFrame(index=range(max_f),  columns=cols)
                df_nf = pd.DataFrame(index=range(max_nf), columns=cols)

                for s in range(num_subjects):

                    # Fill freeze onset/offset pairs (global sample indices)
                    f_pairs = base_ts[s][0]
                    if len(f_pairs) > 0:
                        df_f.iloc[:len(f_pairs), s * 2]     = f_pairs[:, 0]
                        df_f.iloc[:len(f_pairs), s * 2 + 1] = f_pairs[:, 1]

                    # Fill non-freeze onset/offset pairs (global sample indices)
                    nf_pairs = base_ts[s][1]
                    if len(nf_pairs) > 0:
                        df_nf.iloc[:len(nf_pairs), s * 2]     = nf_pairs[:, 0]
                        df_nf.iloc[:len(nf_pairs), s * 2 + 1] = nf_pairs[:, 1]

                out_xlsx = os.path.join(
                    self.app_data['path_name'], f"{f_name}_Timestamps.xlsx"
                )

                with pd.ExcelWriter(out_xlsx, engine='openpyxl') as writer:
                    df_f.to_excel(writer,  sheet_name='freezing timestamps',     index=False)
                    df_nf.to_excel(writer, sheet_name='non freezing timestamps', index=False)

            self.lbl_status.setText("Status: Timestamps Excel Export Complete!")
            self.lbl_status.setStyleSheet("color: green; font-weight: bold;")
            QMessageBox.information(
                self, "Export Complete",
                "Freeze timestamps exported successfully."
            )

        except Exception as e:
            QMessageBox.critical(
                self, "Export Error",
                f"An error occurred during timestamps export:\n{str(e)}"
            )
            self.lbl_status.setText("Status: Error during timestamps export.")
            self.lbl_status.setStyleSheet("color: red; font-weight: bold;")


# ---------------------------------------------------------------
#  Entry Point

if __name__ == '__main__':
    # Guard against re-creating a QApplication instance when running inside
    # IDEs like Spyder that may already have one active.
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    window = AppBehavior()
    window.show()

    app.exec_()