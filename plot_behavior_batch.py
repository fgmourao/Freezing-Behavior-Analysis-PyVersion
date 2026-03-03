"""
plot_behavior_batch.py

DESCRIPTION
    Generates and saves one summary figure per processed file.
    The figure layout scales dynamically with the number of block analyses:

        Row 0       - Full-session movement trace (median + individual subjects)
                      and per-subject freeze raster above the movement axis.
        Row 1       - Event-by-event freeze % line plot + three summary pie
                      charts (total bouts, mean bout duration, mean Delta T).
        Rows 2..N   - One row per active block analysis, following the same
                      4-panel pattern as Row 1 (line plot + three pie charts).

    Each figure is saved as a 300 dpi PNG to the user's Desktop.
    The figure window is displayed and blocks execution until closed
    (plt.show(block=True)), allowing visual inspection before the next
    file is processed.

USAGE
    plot_behavior_batch(data_results)

INPUT
    data_results - dict returned by behavior_analyse (or batch wrapper).
                   Must contain a 'parameters' key and one key per file.

OUTPUT
    One PNG file per processed file saved to ~/Desktop.
    Filename format: '<field_name>_Plot.png'  (300 dpi)

REQUIRES
    numpy, matplotlib, scipy

AUTHOR
    Flavio Mourao (mourao.fg@gmail.com)
    Texas A&M University - Department of Psychological and Brain Sciences
    University of Illinois Urbana-Champaign - Beckman Institute
    Federal University of Minas Gerais - Brazil

Started:     12/2023
Last update: 02/2026
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines   import Line2D
from matplotlib.patches import Patch
from scipy.ndimage      import uniform_filter1d


# =============================================================================
#  Helper: _plot_pie
# =============================================================================

def _plot_pie(ax, values, labels, title):
    """
    Draw a proportional pie chart on the given axes.

    Only slices with values > 0 are included. If all values are zero or
    negative, the axes are hidden. Labels are formatted as
    '<event_name> (<value:.1f>)'.

    Parameters
    ----------
    ax     : matplotlib.axes.Axes
    values : array-like  Numeric values for each slice.
    labels : list of str Corresponding label strings.
    title  : str         Axes title.
    """
    values = np.array(values).flatten()
    idx    = values > 0

    if np.sum(values[idx]) > 0:
        val_filt = values[idx]
        lab_filt = [
            f"{labels[i]} ({val_filt[j]:.1f})"
            for j, i in enumerate(np.where(idx)[0])
        ]
        colors = plt.cm.pink(np.linspace(0.2, 0.8, len(val_filt))).tolist()
        ax.pie(val_filt, labels=lab_filt, colors=colors, textprops={'fontsize': 6})
        ax.set_title(title, fontsize=11, fontweight='bold')
    else:
        ax.axis('off')


# =============================================================================
#  Main Function: plot_behavior_batch
# =============================================================================

def plot_behavior_batch(data_results):
    """
    Generate and save one summary figure per processed file in data_results.

    Parameters
    ----------
    data_results : dict
        Must contain a 'parameters' key and one additional key per file.
    """

    # =========================================================================
    #  1. Validate Input and Extract Shared Parameters
    # =========================================================================

    if 'parameters' not in data_results:
        raise ValueError("'parameters' key not found in data_results.")

    P          = data_results['parameters']
    file_names = [k for k in data_results if k != 'parameters']

    # Output directory: user Desktop (cross-platform via os.path.expanduser)
    desktop_path = os.path.expanduser("~/Desktop")


    # =========================================================================
    #  2. Per-File Plotting Loop
    # =========================================================================

    for f_name in file_names:

        data = data_results[f_name]
        print(f"Plotting and saving {f_name}...")

        # --- Shared signal data ------------------------------------------

        # behavior_epochs[0] contains the full-session signal (S-by-M)
        raw_matrix              = data['behavior_epochs'][0]
        num_subjects, n_samples = raw_matrix.shape

        fs = P.get('fs', 1.0)
        t  = np.arange(n_samples) / fs

        # Smooth each subject's trace with a 1-second moving average window.
        # max(1, ...) guards against fs < 0.5 rounding to 0, which would
        # cause uniform_filter1d to raise a ValueError.
        smooth_win      = max(1, int(round(fs)))
        smoothed_matrix = uniform_filter1d(raw_matrix, size=smooth_win, axis=1)
        median_trace    = np.median(smoothed_matrix, axis=0).flatten()

        # Event onset/offset times (seconds) for shaded regions
        events_sec = np.array(P.get('events_sec', []))
        if events_sec.ndim == 2 and events_sec.shape[1] == 2:
            ev_on_s  = events_sec[:, 0].flatten()
            ev_off_s = events_sec[:, 1].flatten()
        else:
            ev_on_s = ev_off_s = []

        # --- Dynamic figure height ---------------------------------------
        # Base layout has 2 rows; each block analysis adds one additional row.
        blocks          = data.get('blocks', []) or []
        num_block_types = len(blocks)
        total_rows      = 2 + num_block_types
        fig_height      = max(10.0, total_rows * 4.5)

        fig = plt.figure(figsize=(18, fig_height), facecolor='white')

        # NOTE: underscores in f_name are intentionally NOT escaped.
        # Escaping with \_ is only needed when usetex=True in rcParams,
        # which is not the default. Using '\\_' without usetex renders the
        # literal string '\_' in the figure title instead of an underscore.
        fig.suptitle(f_name, fontsize=14, fontweight='bold', y=0.98)

        gs = GridSpec(total_rows, 5, figure=fig, hspace=0.7, wspace=0.4)


        # =====================================================================
        #  Row 0: Full-Session Movement Trace + Freeze Raster
        # =====================================================================
        #
        #  The top panel spans all 5 subplot columns and shows:
        #    - Individual smoothed traces (light grey)
        #    - Group median trace (black)
        #    - Freeze threshold reference line (dashed)
        #    - Shaded event regions (light red)
        #    - Per-subject freeze raster above y=100
        #
        #  Individual traces are plotted in a loop to avoid Matplotlib 2-D
        #  broadcast errors that occur when passing a 2-D array to ax.plot().

        ax1 = fig.add_subplot(gs[0, :])

        # Raster band sits above the 0-100% movement axis
        raster_start_y = 110
        raster_height  = 100
        raster_step    = raster_height / num_subjects
        max_y_axis     = raster_start_y + raster_height + 10

        # Shaded event regions
        for on, off in zip(ev_on_s, ev_off_s):
            ax1.axvspan(on, off, color='mistyrose', alpha=0.4, lw=0)

        # Individual traces (loop avoids 2-D broadcast issue)
        for s in range(num_subjects):
            ax1.plot(t, smoothed_matrix[s, :], color='lightgray', alpha=0.5, lw=0.5)

        # Group median and freeze threshold
        ax1.plot(t, median_trace, 'k', lw=2)
        ax1.axhline(P.get('thr_low', 10), color='k', linestyle='--', lw=1.2)

        # Freeze raster: one horizontal segment per freeze bout per subject
        for s in range(num_subjects):
            y_val   = float(raster_start_y + ((s + 1) * raster_step))
            f_pairs = data['events_behavior_idx'][0][0][s][0]

            # Subject index label to the right of the raster band
            ax1.text(
                t[-1] + (t[-1] * 0.01), y_val, str(s + 1),
                fontsize=7, fontweight='bold', color='dimgray', va='center'
            )

            if len(f_pairs) > 0:
                starts = t[f_pairs[:, 0].astype(int)].flatten()
                ends   = t[f_pairs[:, 1].astype(int)].flatten()

                # hlines requires a y-array of the same length as xmin/xmax
                y_arr = np.full(starts.shape, y_val)
                ax1.hlines(y_arr, starts, ends,
                           color='darkred', lw=max(1.5, raster_step * 0.7))

        ax1.set_xlabel('Time (s)',                fontsize=11, fontweight='bold')
        ax1.set_ylabel('Movement (%)  |  Raster', fontsize=11, fontweight='bold')
        ax1.tick_params(axis='both', which='major', labelsize=8)
        ax1.set_xlim(t[0] - 2, t[-1] + (t[-1] * 0.05))
        ax1.set_ylim(0, max_y_axis)
        ax1.set_yticks(np.arange(0, 101, 20))
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        legend_elements = [
            Line2D([0], [0], color='lightgray', lw=1,         label='Individual traces'),
            Line2D([0], [0], color='k',         lw=2,         label='Median'),
            Patch( facecolor='mistyrose',        alpha=0.4,    label='Events'),
            Line2D([0], [0], color='darkred',   lw=2,         label='Freeze bouts'),
            Line2D([0], [0], color='k',         linestyle='--',
                   lw=1.2,                                     label='Freeze threshold'),
        ]
        ax1.legend(
            handles=legend_elements, loc='upper center',
            bbox_to_anchor=(0.5, -0.25), ncol=5, frameon=False
        )


        # =====================================================================
        #  Row 1: Event-by-Event Freeze % + Summary Pie Charts
        # =====================================================================
        #
        #  Left panel (cols 0-1): individual + mean±SEM line plot per epoch.
        #  Right panels (cols 2-4): pie charts for total bouts, mean duration,
        #                           and mean inter-bout Delta T.

        n_epochs   = len(data['behavior_freezing'])
        num_events = n_epochs - 1   # Full Session (index 0) is excluded here

        if num_events > 0:

            # --- Line plot: freeze % per epoch ---------------------------
            ax2 = fig.add_subplot(gs[1, 0:2])

            freeze_matrix = np.zeros((num_subjects, num_events))
            x_labels      = []

            for e in range(1, n_epochs):
                freeze_matrix[:, e - 1] = data['behavior_freezing'][e][4]
                if e == 1:
                    x_labels.append('Baseline')
                else:
                    event_names = P.get('event_names', [])
                    name        = event_names[e - 2] if (e - 2) < len(event_names) else f'Event {e-1}'
                    x_labels.append(str(name))

            x_vals = np.arange(num_events)

            # Individual subject lines (loop avoids 2-D broadcast issue)
            for s in range(num_subjects):
                ax2.plot(x_vals, freeze_matrix[s, :], color='lightgray', alpha=0.6, lw=1)

            # Group mean ± SEM
            mean_freeze = np.mean(freeze_matrix, axis=0).flatten()
            sem_freeze  = (
                np.std(freeze_matrix, axis=0, ddof=1) / np.sqrt(num_subjects)
            ).flatten() if num_subjects > 1 else np.zeros(num_events)

            ax2.errorbar(
                x_vals, mean_freeze, yerr=sem_freeze,
                fmt='-ko', lw=2, markersize=8,
                markerfacecolor='k', markeredgecolor='k', zorder=5
            )

            ax2.set_ylabel('Freezing (%)', fontsize=9,  fontweight='bold')
            ax2.set_title('Event-by-Event',  fontsize=11)
            ax2.tick_params(axis='both', which='major', labelsize=8)  # fixed: was ax1
            ax2.set_xticks(x_vals)
            ax2.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9)
            ax2.set_xlim(-0.5, num_events - 0.5)
            ax2.set_ylim(-5, 105)
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
                                            
            # --- Pie charts: total bouts, mean duration, mean Delta T ----
            ax3 = fig.add_subplot(gs[1, 2])
            bouts_sum = [np.sum(data['behavior_freezing'][e][2]) for e in range(1, n_epochs)]
            _plot_pie(ax3, bouts_sum, x_labels, 'Total Bouts (All)')

            ax4 = fig.add_subplot(gs[1, 3])
            with np.errstate(invalid='ignore'):
                mean_dur_sum = [
                    np.nanmean(data['behavior_freezing'][e][1]) for e in range(1, n_epochs)
                ]
            _plot_pie(ax4, np.nan_to_num(mean_dur_sum), x_labels, 'Mean Bout Dur. (All)')

            ax5 = fig.add_subplot(gs[1, 4])
            with np.errstate(invalid='ignore'):
                mean_dt_sum = [
                    np.nanmean(data['behavior_freezing'][e][5]) for e in range(1, n_epochs)
                ]
            _plot_pie(ax5, np.nan_to_num(mean_dt_sum), x_labels, 'Mean Delta T (All)')


        # =====================================================================
        #  Rows 2+: Dynamic Block Analysis Rows
        # =====================================================================
        #
        #  One row is added per active block definition.
        #  Each row follows the same 5-column pattern as Row 1:
        #    Cols 0-1 : mean±SEM freeze % per block
        #    Col  2   : pie chart of total bouts per block
        #    Col  3   : pie chart of mean duration per block
        #    Col  4   : pie chart of mean Delta T per block
        #
        #  _plot_pie is defined at module level (not nested inside the loop),
        #  so it is always accessible here even when num_events == 0 and the
        #  Row 1 block is skipped entirely.

        for b_i, block in enumerate(blocks):

            r      = 2 + b_i
            num_b  = len(block['labels'])
            x_vals = np.arange(num_b)

            # --- Line plot: block freeze % -------------------------------
            ax_L = fig.add_subplot(gs[r, 0:2])

            # Individual subject lines (loop avoids 2-D broadcast issue)
            for s in range(num_subjects):
                ax_L.plot(x_vals, block['freeze'][s, :],
                          color='lightgray', alpha=0.6, lw=1)

            # Group mean ± SEM
            mean_blk = np.mean(block['freeze'], axis=0).flatten()
            sem_blk  = (
                np.std(block['freeze'], axis=0, ddof=1) / np.sqrt(num_subjects)
            ).flatten() if num_subjects > 1 else np.zeros(num_b)

            ax_L.errorbar(
                x_vals, mean_blk, yerr=sem_blk,
                fmt='-ko', lw=2, markersize=8,
                markerfacecolor='k', markeredgecolor='k', zorder=5
            )

            ax_L.set_ylabel('Freezing (%)',             fontsize=11, fontweight='bold')
            ax_L.set_title(f"Block: {block['prefix']}", fontsize=11)
            ax_L.tick_params(axis='both', which='major', labelsize=8)  # fixed: was ax1
            ax_L.set_xticks(x_vals)
            ax_L.set_xticklabels(block['labels'], rotation=20, ha='right', fontsize=11)
            ax_L.set_xlim(-0.5, num_b - 0.5)
            ax_L.set_ylim(-5, 105)
            ax_L.spines['top'].set_visible(False)
            ax_L.spines['right'].set_visible(False)
            
            # --- Pie charts: total bouts, mean duration, mean Delta T ----
            ax_P1 = fig.add_subplot(gs[r, 2])
            b_sum = np.sum(block['bout'], axis=0)
            _plot_pie(ax_P1, b_sum, block['labels'], f"Total Bouts ({block['prefix']})")

            ax_P2 = fig.add_subplot(gs[r, 3])
            with np.errstate(invalid='ignore'):
                d_mean = np.nanmean(block['dur'], axis=0)
            _plot_pie(ax_P2, np.nan_to_num(d_mean), block['labels'],
                      f"Mean Bout Dur. ({block['prefix']})")

            ax_P3 = fig.add_subplot(gs[r, 4])
            with np.errstate(invalid='ignore'):
                dt_mean = np.nanmean(block['delta_t'], axis=0)
            _plot_pie(ax_P3, np.nan_to_num(dt_mean), block['labels'],
                      f"Mean Delta T ({block['prefix']})")


        # =====================================================================
        #  Export Figure
        # =====================================================================

        out_filename = os.path.join(desktop_path, f"{f_name}_Plot.png")
        plt.savefig(out_filename, dpi=300, bbox_inches='tight')

        # Display the figure and block execution until the user closes the window.
        # This allows visual inspection before the next file is processed.
        # To process all files without pausing, replace with: plt.close(fig)
        plt.show(block=True)

    print("All plots successfully generated and saved to Desktop!")