"""
BehaviorSync.py - Interface for Video and Neural Recording Visualization

DESCRIPTION:
    GUI tool for synchronized visualization and analysis of behavioral video
    alongside neural recordings and behavioral time-series data (e.g., Load
    Cells, VideoFreeze, MED-PC systems).
    The time vector is built automatically from the number of samples and the
    user-supplied sample rate (Fs), so the input file does NOT need a time
    column — only the signal column is required.
    By default, the script reads the LAST column of the file as the signal,
    making it robust to files that contain non-numeric leading columns.
    Only one experimental subject per file is supported.

INPUT FILE FORMAT:
    - CSV or TXT with at least one numeric column (the signal).
    - Any number of leading columns is accepted; only the last column is used.
    - Example (neural,   1000 Hz): one column of voltage samples.
    - Example (behavior,    5 Hz): one column of position / force samples.

WORKFLOW:
    1. Set Fs (Hz) for each recording BEFORE loading the file.
    2. Load Video -> Load Neural -> Load Behavior  (order is flexible).
    3. Use Play/Pause or arrow keys to navigate the video.
    4. Mark Onset [I] and Offset [O] (or [M] twice) at the desired frames.
    5. (Optional) Define analysis epochs manually in the table or load from file.
    6. Click RUN ANALYSIS to compute behavioral metrics per epoch.
    7. Export results via the File menu (.xlsx or .pkl).

KEYBOARD SHORTCUTS:
    Space      - Play / Pause
    I          - Mark Onset
    O          - Mark Offset
    M          - Smart toggle: Onset if balanced, Offset otherwise
    Left/Right - Step one frame backward / forward
    Del        - Delete last marked event

ANALYSIS (Events Definition & Analysis panel):
    Epochs can be defined manually in the table or loaded from a 3-column file
    (Label, Onset, Offset). A "Full Session" epoch is always included
    automatically. For each epoch, the following metrics are computed:
    - Freezing Percentage         : total time in behavioral bouts / epoch duration (%)
    - Total Bouts                 : number of discrete behavioral episodes
    - Mean Bout Duration          : average duration of individual bouts (s)
    - Bout Duration (raw)         : all individual bout durations (s)
    - Mean Inter-Bout Interval    : average gap between consecutive bouts (s)
    - Inter-Bout Interval (raw)   : all individual inter-bout intervals (s)

OUTPUT:
    1. Behavior Timestamps (.csv) — via File menu:
       Header with metadata (video fps, neural Fs, behavior Fs).
       Columns: Frame onset | Frame offset | Onset (s) | Offset (s) | Duration (s)

    2. Analysis Results (.xlsx) — via File menu:
       One sheet per metric; rows = epochs, columns = subjects.
       Sheets: Behavior_Percentage | Total_Bouts | Mean_Bout_Duration |
               Bout_Durations | Mean_Bout_Latency | Bout_Latencies

    3. Analysis Results (.pkl) — via File menu:
       Dictionary 'analysis_results' with all computed metrics (pickle format).

KNOWN LIMITATIONS:
    - If the behavioral or neural recording does not start at the same
      real-world time as the video, a manual Time Offset (s) field will be
      required for precise alignment. This feature is under development.

NOTE:
    This script is still under active development.
    Synchronization of signals with different start times will be addressed
    in future versions.

REQUIREMENTS:
    Python >= 3.8
    PyQt5, pyqtgraph, opencv-python, numpy, pandas, openpyxl, pickle

AUTHOR:
    Flavio Mourao  (mourao.fg@gmail.com)
    Texas A&M University     - Department of Psychological and Brain Sciences
    Beckman Institute / UIUC - University of Illinois Urbana-Champaign
    Federal University of Minas Gerais (UFMG) - Brazil

Started:     09/2019
Last update: 03/2026
"""

import sys
import os
import cv2
import pickle
import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore, QtGui


class BehaviorSync(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BehaviorSync")
        self.resize(1400, 850)

        # --- State Variables ---
        self.video_cap      = None
        self.video_filename = ""
        self.is_playing     = False
        self.curr_frame     = 1
        self.total_frames   = 1
        self.fps            = 30
        self.playback_speed = 1.0
        self.window_scale   = 10.0

        self.neuro_time = np.array([])
        self.neuro_data = np.array([])
        self.behav_time = np.array([])
        self.behav_data = np.array([])

        self.onsets_sec          = []
        self.offsets_sec         = []
        self.onsets_frame        = []
        self.offsets_frame       = []
        self.event_lines_neural  = []
        self.event_lines_behav   = []

        self.analysis_results = None

        self.setup_gui()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)

    # ---------------------------------------------------------------
    # GUI SETUP

    def setup_gui(self):
        #  Menu: File
        menubar   = self.menuBar()
        file_menu = menubar.addMenu('File')

        self.action_export_xls = QtWidgets.QAction('1. Export Results (.xlsx)', self)
        self.action_export_xls.setEnabled(False)
        self.action_export_xls.triggered.connect(self.export_excel)
        file_menu.addAction(self.action_export_xls)

        self.action_save_pkl = QtWidgets.QAction('2. Save Results (.pkl)', self)
        self.action_save_pkl.setEnabled(False)
        self.action_save_pkl.triggered.connect(self.save_pkl_file)
        file_menu.addAction(self.action_save_pkl)

        self.action_export_csv = QtWidgets.QAction('3. Export Behavior Timestamps (.csv)', self)
        self.action_export_csv.triggered.connect(self.export_behav_ts)
        file_menu.addAction(self.action_export_csv)

        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QGridLayout(central_widget)


        # Video Panel
        vid_group  = QtWidgets.QGroupBox("Video")
        vid_layout = QtWidgets.QVBoxLayout()

        self.video_label = QtWidgets.QLabel("Video Player")
        self.video_label.setAlignment(QtCore.Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")

        controls_layout = QtWidgets.QHBoxLayout()

        self.btn_load_vid = QtWidgets.QPushButton("Load Video")
        self.btn_load_vid.clicked.connect(self.load_video)

        self.btn_play = QtWidgets.QPushButton("Play / Pause [Space]")
        self.btn_play.setCheckable(True)
        self.btn_play.clicked.connect(self.toggle_play)

        self.lbl_time = QtWidgets.QLabel("Time: -- / --")

        self.combo_speed = QtWidgets.QComboBox()
        self.combo_speed.addItems(["0.25x", "0.5x", "1x", "2x", "4x", "10x", "20x"])
        self.combo_speed.setCurrentIndex(2)
        self.combo_speed.currentIndexChanged.connect(self.change_speed)

        controls_layout.addWidget(self.btn_load_vid)
        controls_layout.addWidget(self.btn_play)
        controls_layout.addWidget(self.lbl_time)
        controls_layout.addWidget(QtWidgets.QLabel("Speed:"))
        controls_layout.addWidget(self.combo_speed)

        self.slider_vid = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_vid.sliderMoved.connect(self.slider_callback)

        vid_layout.addWidget(self.video_label)
        vid_layout.addLayout(controls_layout)
        vid_layout.addWidget(self.slider_vid)
        vid_group.setLayout(vid_layout)


        # Signal Plots (PyQtGraph)
        graphs_layout = QtWidgets.QVBoxLayout()

        window_layout = QtWidgets.QHBoxLayout()
        window_layout.addStretch()
        window_layout.addWidget(QtWidgets.QLabel("Time Window (s):"))
        self.edit_window = QtWidgets.QLineEdit(str(self.window_scale))
        self.edit_window.setFixedWidth(50)
        self.edit_window.editingFinished.connect(self.update_window_scale)
        window_layout.addWidget(self.edit_window)
        graphs_layout.addLayout(window_layout)

        # Neural plot
        self.plot_neural  = pg.PlotWidget(title="Neural Recording")
        self.plot_neural.setBackground('w')
        self.curve_neural = self.plot_neural.plot(pen=pg.mkPen('#0072BD', width=1))
        self.cursor_neural = pg.InfiniteLine(
            angle=90, movable=False, pen=pg.mkPen('#A2142F', width=2.5))
        self.plot_neural.addItem(self.cursor_neural)

        neural_ctrl_layout = QtWidgets.QHBoxLayout()
        btn_load_neuro = QtWidgets.QPushButton("Load Neural (*.CSV / *.TXT)")
        btn_load_neuro.clicked.connect(lambda: self.load_data('neuro'))
        self.edit_fs_neuro = QtWidgets.QLineEdit("1000")
        self.edit_fs_neuro.setFixedWidth(60)
        self.edit_fs_neuro.editingFinished.connect(self.setFocus)
        self.edit_y_neuro = QtWidgets.QLineEdit("Auto")
        self.edit_y_neuro.setFixedWidth(60)
        self.edit_y_neuro.editingFinished.connect(self.update_y_axes)

        neural_ctrl_layout.addWidget(btn_load_neuro)
        neural_ctrl_layout.addWidget(QtWidgets.QLabel("Fs (Hz):"))
        neural_ctrl_layout.addWidget(self.edit_fs_neuro)
        neural_ctrl_layout.addWidget(QtWidgets.QLabel("Set Y scale:"))
        neural_ctrl_layout.addWidget(self.edit_y_neuro)
        neural_ctrl_layout.addStretch()

        # Behavior plot
        self.plot_behav  = pg.PlotWidget(title="Behavior Recording")
        self.plot_behav.setBackground('w')
        self.curve_behav = self.plot_behav.plot(pen=pg.mkPen('#808080', width=1.5))
        self.cursor_behav = pg.InfiniteLine(
            angle=90, movable=False, pen=pg.mkPen('#A2142F', width=2.5))
        self.plot_behav.addItem(self.cursor_behav)

        behav_ctrl_layout = QtWidgets.QHBoxLayout()
        btn_load_behav = QtWidgets.QPushButton("Load Behavior (*.CSV / *.TXT)")
        btn_load_behav.clicked.connect(lambda: self.load_data('behav'))
        self.edit_fs_behav = QtWidgets.QLineEdit("5")
        self.edit_fs_behav.setFixedWidth(60)
        self.edit_fs_behav.editingFinished.connect(self.setFocus)
        self.edit_y_behav = QtWidgets.QLineEdit("Auto")
        self.edit_y_behav.setFixedWidth(60)
        self.edit_y_behav.editingFinished.connect(self.update_y_axes)

        behav_ctrl_layout.addWidget(btn_load_behav)
        behav_ctrl_layout.addWidget(QtWidgets.QLabel("Fs (Hz):"))
        behav_ctrl_layout.addWidget(self.edit_fs_behav)
        behav_ctrl_layout.addWidget(QtWidgets.QLabel("Set Y scale:"))
        behav_ctrl_layout.addWidget(self.edit_y_behav)
        behav_ctrl_layout.addStretch()

        graphs_layout.addWidget(self.plot_neural)
        graphs_layout.addLayout(neural_ctrl_layout)
        graphs_layout.addWidget(self.plot_behav)
        graphs_layout.addLayout(behav_ctrl_layout)


        # Event Marking Panel
        events_group = QtWidgets.QGroupBox(
            "Event Marking  —  Shortcuts:  I = Onset  |  O = Offset  |  M = Toggle  |  Del = Undo")
        events_layout = QtWidgets.QHBoxLayout()

        btn_ev_layout = QtWidgets.QVBoxLayout()
        btn_onset = QtWidgets.QPushButton("ONSET [I]")
        btn_onset.setStyleSheet(
            "background-color: #A2142F; color: white; font-weight: bold; height: 30px;")
        btn_onset.clicked.connect(lambda: self.mark_event('onset'))

        btn_offset = QtWidgets.QPushButton("OFFSET [O]")
        btn_offset.setStyleSheet(
            "background-color: #D95319; color: white; font-weight: bold; height: 30px;")
        btn_offset.clicked.connect(lambda: self.mark_event('offset'))

        btn_del = QtWidgets.QPushButton("Delete Last [Del]")
        btn_del.clicked.connect(self.delete_last_event)

        btn_ev_layout.addWidget(btn_onset)
        btn_ev_layout.addWidget(btn_offset)
        btn_ev_layout.addWidget(btn_del)

        self.list_onset    = QtWidgets.QListWidget()
        self.list_offset   = QtWidgets.QListWidget()
        self.list_duration = QtWidgets.QListWidget()

        events_layout.addLayout(btn_ev_layout)
        for label, lst in [
            ("Onsets (s)",   self.list_onset),
            ("Offsets (s)",  self.list_offset),
            ("Duration (s)", self.list_duration)
        ]:
            vbox = QtWidgets.QVBoxLayout()
            vbox.addWidget(QtWidgets.QLabel(label))
            vbox.addWidget(lst)
            events_layout.addLayout(vbox)
        events_group.setLayout(events_layout)


        # Analysis Panel
        analysis_group  = QtWidgets.QGroupBox("Events Definition & Analysis")
        analysis_layout = QtWidgets.QHBoxLayout()

        self.table_events = QtWidgets.QTableWidget(200, 3)
        self.table_events.setHorizontalHeaderLabels(["Event Label", "Onset (s)", "Offset (s)"])
        self.table_events.horizontalHeader().setStretchLastSection(True)
        analysis_layout.addWidget(self.table_events)

        btn_analysis_layout = QtWidgets.QVBoxLayout()
        btn_load_epochs = QtWidgets.QPushButton("Load Events File (.txt / .csv)")
        btn_load_epochs.clicked.connect(self.load_events_file)
        btn_run = QtWidgets.QPushButton("RUN ANALYSIS")
        btn_run.setStyleSheet(
            "background-color: #CCCCCC; font-weight: bold; height: 50px;")
        btn_run.clicked.connect(self.run_analysis)

        btn_analysis_layout.addWidget(btn_load_epochs)
        btn_analysis_layout.addWidget(btn_run)
        analysis_layout.addLayout(btn_analysis_layout)
        analysis_group.setLayout(analysis_layout)


        # Main Grid Assembly
        main_layout.addWidget(vid_group,       0, 0)
        main_layout.addLayout(graphs_layout,   0, 1)
        main_layout.addWidget(events_group,    1, 0)
        main_layout.addWidget(analysis_group,  1, 1)
        main_layout.setRowStretch(0, 3)
        main_layout.setRowStretch(1, 1)


        # Focus Management
        # The main window captures all keyboard shortcuts. Remove focus from
        # interactive widgets that do not require keyboard input.
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        for widget in self.findChildren(QtWidgets.QWidget):
            if isinstance(widget, (
                QtWidgets.QPushButton, QtWidgets.QSlider,
                QtWidgets.QComboBox, QtWidgets.QListWidget,
                pg.PlotWidget
            )):
                widget.setFocusPolicy(QtCore.Qt.NoFocus)

    def mousePressEvent(self, event):
        """Return focus to the main window on background click."""
        self.setFocus()
        super().mousePressEvent(event)

    # ---------------------------------------------------------------
    # KEYBOARD SHORTCUTS

    def keyPressEvent(self, event):
        """Handle all global keyboard shortcuts."""
        if self.video_cap is None:
            return super().keyPressEvent(event)

        # Do not intercept shortcuts while the user is typing in the events table
        if isinstance(QtWidgets.QApplication.focusWidget(), QtWidgets.QTableWidget):
            return super().keyPressEvent(event)

        key = event.key()

        if key == QtCore.Qt.Key_Space:
            self.btn_play.setChecked(not self.btn_play.isChecked())
            self.toggle_play()
        elif key == QtCore.Qt.Key_I:
            self.mark_event('onset')
        elif key == QtCore.Qt.Key_O:
            self.mark_event('offset')
        elif key == QtCore.Qt.Key_M:
            if len(self.onsets_sec) == len(self.offsets_sec):
                self.mark_event('onset')
            else:
                self.mark_event('offset')
        elif key == QtCore.Qt.Key_Left and self.curr_frame > 1:
            self.curr_frame -= 1
            self.update_frame_manually()
        elif key == QtCore.Qt.Key_Right and self.curr_frame < self.total_frames:
            self.curr_frame += 1
            self.update_frame_manually()
        elif key in (QtCore.Qt.Key_Backspace, QtCore.Qt.Key_Delete):
            self.delete_last_event()
        else:
            super().keyPressEvent(event)

    # ---------------------------------------------------------------
    # VIDEO & DATA LOADING

    def load_video(self):
        """Open a video file and initialize playback controls."""
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Video", "",
            "Video Files (*.mp4 *.avi *.wmv *.mov)")
        if not filename:
            return
        self.video_filename = filename
        self.video_cap      = cv2.VideoCapture(filename)
        self.total_frames   = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps            = self.video_cap.get(cv2.CAP_PROP_FPS)
        self.curr_frame     = 1
        self.slider_vid.setRange(1, self.total_frames)
        self.update_frame_manually()

    def change_speed(self):
        """Update playback speed from the combo box selection."""
        speeds = [0.25, 0.5, 1.0, 2.0, 4.0, 10.0, 20.0]
        self.playback_speed = speeds[self.combo_speed.currentIndex()]
        if self.is_playing:
            self.timer.start(int(1000 / (self.fps * self.playback_speed)))

    def load_data(self, d_type):
        """
        Load a neural or behavioral recording from a .csv or .txt file.

        The last column is used as the signal. If two columns are present,
        the first column is treated as the time series; otherwise, time is
        reconstructed from the provided sampling frequency (Fs). Only one
        experimental subject per file is supported.
        """
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Data File", "",
            "Data Files (*.csv *.txt)")
        if not filename:
            return

        df     = pd.read_csv(filename, header=None)
        signal = df.iloc[:, -1].apply(pd.to_numeric, errors='coerce').dropna().values

        if d_type == 'neuro':
            fs = float(self.edit_fs_neuro.text())
            self.neuro_time = np.arange(len(signal)) / fs
            self.neuro_data = signal
            self.curve_neural.setData(self.neuro_time, self.neuro_data)
        else:
            fs = float(self.edit_fs_behav.text())
            self.behav_time = np.arange(len(signal)) / fs
            self.behav_data = signal
            self.curve_behav.setData(self.behav_time, self.behav_data)

        self.update_y_axes()
        self.sync_data_axes()

    # ---------------------------------------------------------------
    # PLOT CONTROLS

    def update_y_axes(self):
        """Apply manual Y-axis scaling or auto-range if input is not numeric."""
        try:
            self.plot_neural.setYRange(
                -float(self.edit_y_neuro.text()),
                 float(self.edit_y_neuro.text()))
        except ValueError:
            self.plot_neural.autoRange()

        try:
            self.plot_behav.setYRange(0, float(self.edit_y_behav.text()))
        except ValueError:
            self.plot_behav.autoRange()

        self.setFocus()

    def update_window_scale(self):
        """Update the visible time window width (in seconds)."""
        try:
            self.window_scale = float(self.edit_window.text())
            self.sync_data_axes()
        except ValueError:
            pass
        self.setFocus()

    # ---------------------------------------------------------------
    # PLAYBACK

    def toggle_play(self):
        """Start or pause video playback."""
        if self.video_cap is None:
            return
        self.is_playing = self.btn_play.isChecked()
        if self.is_playing:
            self.timer.start(int(1000 / (self.fps * self.playback_speed)))
        else:
            self.timer.stop()

    def update_frame(self):
        """Advance one frame during timed playback."""
        if self.curr_frame >= self.total_frames:
            self.btn_play.setChecked(False)
            self.toggle_play()
            return
        ret, frame = self.video_cap.read()
        if ret:
            self.curr_frame += 1
            self.display_frame(frame)
            if self.curr_frame % 3 == 0:
                self.sync_data_axes()

    def update_frame_manually(self):
        """Seek to the current frame and refresh the display."""
        if self.video_cap is None:
            return
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, self.curr_frame - 1)
        ret, frame = self.video_cap.read()
        if ret:
            self.display_frame(frame)
            self.sync_data_axes()
            self.slider_vid.setValue(self.curr_frame)

    def display_frame(self, frame):
        """Render a video frame in the video label widget."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch  = frame_rgb.shape
        q_img     = QtGui.QImage(
            frame_rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
        self.video_label.setPixmap(
            QtGui.QPixmap.fromImage(q_img).scaled(
                self.video_label.size(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation))
        elapsed = (self.curr_frame - 1) / self.fps
        total   = self.total_frames / self.fps
        self.lbl_time.setText(f"Time: {elapsed:.2f} s / {total:.2f} s")

    def slider_callback(self, value):
        """Respond to manual slider movement."""
        self.curr_frame = value
        self.update_frame_manually()

    def sync_data_axes(self):
        """Center both signal plots on the current video time position."""
        c_time = (self.curr_frame - 1) / self.fps
        x_min  = max(0, c_time - self.window_scale / 2)
        x_max  = c_time + self.window_scale / 2
        self.plot_neural.setXRange(x_min, x_max, padding=0)
        self.plot_behav.setXRange(x_min, x_max, padding=0)
        self.cursor_neural.setValue(c_time)
        self.cursor_behav.setValue(c_time)

    # ---------------------------------------------------------------
    # EVENT MARKING

    def mark_event(self, e_type):
        """Mark an onset or offset event at the current video timestamp."""
        if self.video_cap is None:
            return
        c_time = (self.curr_frame - 1) / self.fps

        if e_type == 'onset':
            self.onsets_sec.append(c_time)
            self.onsets_frame.append(self.curr_frame)
            color = '#A2142F'
        elif e_type == 'offset':
            if len(self.offsets_sec) >= len(self.onsets_sec):
                QtWidgets.QMessageBox.warning(
                    self, "Order Error",
                    "An Onset must be marked before each Offset.")
                return
            self.offsets_sec.append(c_time)
            self.offsets_frame.append(self.curr_frame)
            color = '#D95319'

        line_n = pg.InfiniteLine(
            pos=c_time, angle=90, movable=False,
            pen=pg.mkPen(color, width=2))
        line_b = pg.InfiniteLine(
            pos=c_time, angle=90, movable=False,
            pen=pg.mkPen(color, width=2))
        self.plot_neural.addItem(line_n)
        self.event_lines_neural.append(line_n)
        self.plot_behav.addItem(line_b)
        self.event_lines_behav.append(line_b)
        self.update_lists()

    def delete_last_event(self):
        """Remove the most recently marked onset or offset."""
        if not self.onsets_sec:
            return
        if len(self.offsets_sec) < len(self.onsets_sec):
            self.onsets_sec.pop()
            self.onsets_frame.pop()
        else:
            self.offsets_sec.pop()
            self.offsets_frame.pop()

        if self.event_lines_neural:
            self.plot_neural.removeItem(self.event_lines_neural.pop())
        if self.event_lines_behav:
            self.plot_behav.removeItem(self.event_lines_behav.pop())
        self.update_lists()

    def update_lists(self):
        """Refresh the onset, offset, and duration list widgets."""
        self.list_onset.clear()
        self.list_offset.clear()
        self.list_duration.clear()

        self.list_onset.addItems([f"{x:.3f}" for x in self.onsets_sec])
        self.list_offset.addItems([f"{x:.3f}" for x in self.offsets_sec])

        n_pairs   = min(len(self.onsets_sec), len(self.offsets_sec))
        durations = (np.array(self.offsets_sec[:n_pairs])
                     - np.array(self.onsets_sec[:n_pairs]))
        self.list_duration.addItems([f"{x:.3f}" for x in durations])

    # ---------------------------------------------------------------
    # ANALYSIS & EXPORT

    def load_events_file(self):
        """
        Load an events/epochs file into the analysis table.

        Expected format: three columns — Event Label | Onset (s) | Offset (s)
        """
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Events File", "",
            "Text Files (*.csv *.txt)")
        if not filename:
            return

        df = pd.read_csv(filename, header=None)
        if df.shape[1] < 3:
            QtWidgets.QMessageBox.critical(
                self, "Format Error",
                "The file must contain 3 columns: Event Label, Onset (s), Offset (s).")
            return

        for i, row in df.iterrows():
            if pd.isna(row[1]) or pd.isna(row[2]):
                continue
            self.table_events.setItem(i, 0, QtWidgets.QTableWidgetItem(str(row[0])))
            self.table_events.setItem(i, 1, QtWidgets.QTableWidgetItem(str(row[1])))
            self.table_events.setItem(i, 2, QtWidgets.QTableWidgetItem(str(row[2])))

            line_n = pg.InfiniteLine(
                pos=float(row[1]), angle=90, movable=False,
                pen=pg.mkPen('b', style=QtCore.Qt.DashLine))
            self.plot_neural.addItem(line_n)

            line_b = pg.InfiniteLine(
                pos=float(row[1]), angle=90, movable=False,
                pen=pg.mkPen('b', style=QtCore.Qt.DashLine))
            self.plot_behav.addItem(line_b)

    def run_analysis(self):
        """
        Compute behavioral metrics for each defined epoch.

        Metrics include: percentage of time in behavior, total bout count,
        mean bout duration, inter-bout latencies, and their raw distributions.
        """
        n_pairs = min(len(self.onsets_sec), len(self.offsets_sec))
        if n_pairs == 0:
            QtWidgets.QMessageBox.warning(
                self, "Warning",
                "No complete behavioral bouts have been marked.")
            return

        bouts_on  = np.array(self.onsets_sec[:n_pairs])
        bouts_off = np.array(self.offsets_sec[:n_pairs])

        # Default epoch: full session
        epochs_name = ["Full Session"]
        epochs_on   = [0.0]
        max_time    = max(
            np.max(bouts_off),
            self.total_frames / self.fps if self.fps > 0 else 0)
        epochs_off  = [max_time]

        # Additional epochs from the events table
        for i in range(self.table_events.rowCount()):
            item_name = self.table_events.item(i, 0)
            item_on   = self.table_events.item(i, 1)
            item_off  = self.table_events.item(i, 2)
            if item_name and item_on and item_off:
                try:
                    epochs_name.append(item_name.text())
                    epochs_on.append(float(item_on.text()))
                    epochs_off.append(float(item_off.text()))
                except ValueError:
                    pass

        res = {
            'epoch_labels': epochs_name,
            'pct':      [], 'bouts':    [], 'mean_dur': [],
            'raw_dur':  [], 'mean_lat': [], 'raw_lat':  []
        }

        for t_on, t_off in zip(epochs_on, epochs_off):
            ep_dur = t_off - t_on
            if ep_dur <= 0:
                continue

            adj_on  = np.maximum(bouts_on,  t_on)
            adj_off = np.minimum(bouts_off, t_off)
            valid   = adj_on < adj_off
            v_on, v_off = adj_on[valid], adj_off[valid]
            durations   = v_off - v_on

            res['bouts'].append(len(durations))
            res['pct'].append(
                (np.sum(durations) / ep_dur) * 100 if ep_dur > 0 else 0)
            res['raw_dur'].append(durations.tolist())
            res['mean_dur'].append(np.mean(durations) if len(durations) > 0 else 0)

            if len(v_on) > 1:
                latencies = v_on[1:] - v_off[:-1]
                latencies = latencies[latencies >= 0]
                res['raw_lat'].append(latencies.tolist())
                res['mean_lat'].append(
                    np.mean(latencies) if len(latencies) > 0 else 0)
            else:
                res['raw_lat'].append([])
                res['mean_lat'].append(0)

        self.analysis_results = res
        QtWidgets.QMessageBox.information(
            self, "Analysis Complete",
            "Analysis finished successfully. Results are ready to export.")
        self.action_export_xls.setEnabled(True)
        self.action_save_pkl.setEnabled(True)

    def export_excel(self):
        """Export analysis results to a multi-sheet Excel workbook (.xlsx)."""
        if not self.analysis_results:
            return
        default_name = (self.video_filename.replace('.mp4', '_Results.xlsx')
                        if self.video_filename else "Results.xlsx")
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Excel File", default_name,
            "Excel Files (*.xlsx)")
        if not filename:
            return

        res    = self.analysis_results
        epochs = res['epoch_labels']

        def fmt_list(lst):
            return ", ".join([f"{x:.2f}" for x in lst]) if lst else "0"

        sheets = {
            '1_Behavior_Percentage':  pd.DataFrame({'Epoch': epochs, 'Subject 1': res['pct']}),
            '2_Total_Bouts':          pd.DataFrame({'Epoch': epochs, 'Subject 1': res['bouts']}),
            '3_Mean_Bout_Duration':   pd.DataFrame({'Epoch': epochs, 'Subject 1': res['mean_dur']}),
            '4_Bout_Durations':       pd.DataFrame({'Epoch': epochs, 'Subject 1': [fmt_list(r) for r in res['raw_dur']]}),
            '5_Mean_Bout_Latency':    pd.DataFrame({'Epoch': epochs, 'Subject 1': res['mean_lat']}),
            '6_Bout_Latencies':       pd.DataFrame({'Epoch': epochs, 'Subject 1': [fmt_list(r) if r else "NaN" for r in res['raw_lat']]})
        }

        with pd.ExcelWriter(filename) as writer:
            for sheet_name, df in sheets.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)

        QtWidgets.QMessageBox.information(
            self, "Export Successful",
            f"Excel file saved to:\n{filename}")

    def save_pkl_file(self):
        """Serialize analysis results to a Python pickle file (.pkl)."""
        if not self.analysis_results:
            return
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Results", "BehaviorSync_Results.pkl",
            "Pickle Files (*.pkl)")
        if not filename:
            return
        with open(filename, 'wb') as f:
            pickle.dump(self.analysis_results, f)
        QtWidgets.QMessageBox.information(
            self, "Saved",
            "Results serialized and saved to .pkl successfully.")

    def export_behav_ts(self):
        """Export behavioral event timestamps (frames and seconds) to CSV."""
        if not self.onsets_sec:
            return
        n_onsets = len(self.onsets_sec)
        os_arr   = self.onsets_sec.copy()
        ofs_arr  = self.offsets_sec.copy()
        on_arr   = self.onsets_frame.copy()
        of_arr   = self.offsets_frame.copy()

        # Pad offsets if the last bout has no closing offset
        while len(ofs_arr) < n_onsets:
            ofs_arr.append(np.nan)
            of_arr.append(np.nan)

        dur = np.array(ofs_arr) - np.array(os_arr)

        default_name = (self.video_filename.replace('.mp4', '_events.csv')
                        if self.video_filename else "behavior_events.csv")
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Timestamps", default_name,
            "CSV Files (*.csv)")
        if not filename:
            return

        df = pd.DataFrame({
            'Frame onset':  on_arr,
            'Frame offset': of_arr,
            'Onset (s)':    os_arr,
            'Offset (s)':   ofs_arr,
            'Duration (s)': dur
        })

        with open(filename, 'w') as f:
            f.write(f"# BehaviorSync export | Video: {self.fps:.4f} fps\n")
        df.to_csv(filename, mode='a', index=False)

        QtWidgets.QMessageBox.information(
            self, "Export Successful",
            f"Exported {n_onsets} event(s) to:\n{filename}")


# ---------------------------------------------------------------
# ENTRY POINT

if __name__ == "__main__":
    # Reuse an existing QApplication instance when running inside Spyder
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)

    window = BehaviorSync()
    window.show()

    # Use exec_() instead of sys.exit() to preserve the Spyder console session
    app.exec_()