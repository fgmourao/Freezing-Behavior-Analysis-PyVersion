"""
behavior_analyse.py

DESCRIPTION
    Identifies and quantifies freezing and non-freezing bouts from a raw
    movement signal. Natively supports multiple subjects (columns) and
    includes optional Block Analysis for grouping consecutive events.

    Each subject's signal is normalised independently to 0-100% of its own
    maximum movement. Freezing bouts are detected once globally per subject
    and then mapped to each epoch, ensuring consistency across all outputs.

    All indices are 0-based (Python convention), unlike the MATLAB version
    which uses 1-based indexing.

USAGE
    data, parameters = behavior_analyse(raw_signal, params)

INPUT
    raw_signal - M-by-S numpy array: M samples, S subjects.
                 1-D arrays are accepted and reshaped to M-by-1 internally.
    params     - dict with analysis parameters:
                   Required keys:
                     'fs'             Sampling rate (Hz)
                     'thr_low'        Freeze threshold (% movement)
                     'thr_dur'        Minimum freeze duration (s)
                     'baseline_dur'   Baseline duration (s)
                     'events_sec'     List of [onset_s, offset_s] pairs
                     'event_names'    List of event label strings
                   Optional keys (for block analysis):
                     'block_prefixes' List of prefix strings (up to 5)
                     'block_sizes'    List of block sizes (N events per block)

OUTPUT
    data - dict with keys:
        'behavior_freezing'    List of N epochs, each a 7-element list:
            [0]  Raw bout durations      (list of arrays, one per subject, seconds)
            [1]  Mean bout duration      (S-length array, seconds)
            [2]  Number of bouts         (S-length array, count)
            [3]  Total freeze time       (S-length array, seconds)
            [4]  Freeze percentage       (S-length array, %)
            [5]  Mean inter-bout Delta T (S-length array, seconds; NaN if < 2 bouts)
            [6]  Raw inter-bout Delta T  (list of arrays, one per subject, seconds)

        'behavior_nonfreezing' List of N epochs, each a 1-element list:
            [0]  Non-freeze durations    (list of arrays, one per subject, seconds)

        'events_behavior_idx'  List of N epochs, each a 1-element list:
            [0]  S-by-3 list per subject:
                 [s][0]  Freeze index pairs     (N-by-2 array, global 0-based)
                 [s][1]  Non-freeze index pairs (N-by-2 array, global 0-based)
                 [s][2]  Binary freeze mask     (1-D boolean array)

        'behavior_epochs'      List of N epochs:
            [i]  S-by-M array of normalised signal for this epoch

        'blocks'               List of dicts, one per active block definition:
            'prefix'   Matched event prefix string
            'size'     Block size (N events per block)
            'labels'   List of label strings (e.g., 'CS 1-5')
            'freeze'   S-by-B array of mean freeze % per block
            'bout'     S-by-B array of summed bout count per block
            'dur'      S-by-B array of mean bout duration per block
            'delta_t'  S-by-B array of mean inter-bout Delta T per block

    parameters - copy of the input params dict, with 'events_idx' added.

EPOCH ROW INDEXING
    Index 0 - Full Session  (entire recording)
    Index 1 - Baseline      (samples 0 to baseline_dur * fs - 1)
    Index 2+ - Experimental Events (one entry per event in events_sec)

REQUIRES
    detect_bouts.py must be importable from the same directory or Python path.

AUTHOR
    Flavio Mourao (mourao.fg@gmail.com)
    Texas A&M University - Department of Psychological and Brain Sciences
    University of Illinois Urbana-Champaign - Beckman Institute
    Federal University of Minas Gerais - Brazil

Started:     12/2023
Last update: 03/2026
"""

import warnings
import numpy as np
from detect_bouts import detect_bouts


# ---------------------------------------------------------------
#  Helper: get_nf_pairs

def get_nf_pairs(f_pairs, ep_start, ep_end, min_dur):
    """
    Build non-freeze [start, end] pairs within an epoch (0-based indices).

    Produces index pairs for three types of non-freeze intervals:
        1. Segment from epoch start to the first freeze onset
        2. Gaps between consecutive freeze bouts
        3. Segment from the last freeze offset to epoch end

    Parameters
    ----------
    f_pairs  : ndarray, shape (B, 2)
        Freeze [start, end] pairs (0-based global indices).
        Should include ALL clipped fragments (including inherited bouts)
        so that gaps at epoch boundaries are temporally accurate.
    ep_start : int
        Global index of the first sample in this epoch.
    ep_end   : int
        Global index of the last sample in this epoch.
    min_dur  : int
        Minimum segment length in samples; shorter segments are discarded.

    Returns
    -------
    nf : ndarray, shape (K, 2)
        Non-freeze [start, end] pairs (0-based global indices).
        Empty array with shape (0, 2) if no valid segments exist.
    """

    # Special case: no freezing — entire epoch is one non-freeze segment
    if len(f_pairs) == 0:
        if (ep_end - ep_start + 1) >= min_dur:
            return np.array([[ep_start, ep_end]])
        else:
            return np.zeros((0, 2), dtype=int)

    # Sort by onset to guarantee chronological order
    f_pairs = f_pairs[np.argsort(f_pairs[:, 0])]

    # Build candidate non-freeze intervals from the spaces around freeze bouts:
    #   nf_s[0]    = epoch start
    #   nf_s[1:]   = each freeze offset + 1  (gap after each bout)
    #   nf_e[:-1]  = each freeze onset - 1   (gap before each bout)
    #   nf_e[-1]   = epoch end
    nf_s = np.concatenate(([ep_start],       f_pairs[:, 1] + 1))
    nf_e = np.concatenate((f_pairs[:, 0] - 1, [ep_end]        ))

    nf_raw = np.column_stack((nf_s, nf_e))

    # Clip to epoch boundaries (guards against inherited bouts at epoch edges)
    nf_raw[:, 0] = np.maximum(nf_raw[:, 0], ep_start)
    nf_raw[:, 1] = np.minimum(nf_raw[:, 1], ep_end)

    # Retain only segments meeting the minimum duration (end - start + 1 >= min_dur)
    valid = (nf_raw[:, 1] - nf_raw[:, 0] + 1) >= min_dur
    return nf_raw[valid]


# ---------------------------------------------------------------
#  Main Function: behavior_analyse

def behavior_analyse(raw_signal, params):
    """
    Identify and quantify freezing and non-freezing bouts from a raw signal.
    See module docstring for full input/output specification.
    """

    # ---------------------------------------------------------------
    #  1. Input Handling & Normalisation

    raw_signal = np.asarray(raw_signal, dtype=float)

    # Reshape 1-D input to a column vector (M-by-1)
    if raw_signal.ndim == 1:
        raw_signal = raw_signal.reshape(-1, 1)

    n_samples, num_subjects = raw_signal.shape

    # Work on a copy of params to avoid mutating the caller's dict
    P = params.copy()

    # Normalise each subject independently to 0-100% of their own max movement.
    # Vectorised across subjects (min/max operate column-wise).
    min_vals   = np.min(raw_signal, axis=0)
    max_vals   = np.max(raw_signal, axis=0)
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1   # guard against flat (zero-range) signals
    raw_signal = 100 * (raw_signal - min_vals) / range_vals


    # ---------------------------------------------------------------
    #  2. Parameters & Epoch Boundaries

    min_samples = int(np.round(P['thr_dur'] * P['fs']))

    # Epoch boundary list: [start_sample, end_sample] pairs (0-based, inclusive).
    #   Index 0 : Full session
    #   Index 1 : Baseline
    #   Index 2+: Experimental events
    #
    # NOTE: Python uses 0-based indexing, so no +1 offset is needed on onsets
    # (unlike the MATLAB version which converted from 0-based seconds to 1-based samples).
    epoch_bounds = [
        [0,          n_samples - 1],
        [0,          int(np.round(P['baseline_dur'] * P['fs'])) - 1]
    ]

    if 'events_sec' in P and len(P['events_sec']) > 0:
        events_idx = np.round(np.array(P['events_sec']) * P['fs']).astype(int)
        P['events_idx'] = events_idx
        for ev in events_idx:
            epoch_bounds.append([ev[0], ev[1]])

    epoch_bounds = np.array(epoch_bounds)
    n_epochs     = len(epoch_bounds)


    # ---------------------------------------------------------------
    #  3. Global Freeze Detection (per subject)

    # Build a binary freeze mask for the entire session (M-by-S boolean array).
    # Samples at or below thr_low are classified as frozen.
    freeze_mask_global = raw_signal <= P['thr_low']

    # Detect all freeze bouts globally for each subject.
    # detect_bouts returns a 3-by-B array: row 0 = onset, row 1 = duration (samples),
    # row 2 = duration (seconds).
    all_freezing_bouts = []
    for s in range(num_subjects):
        bouts = detect_bouts(freeze_mask_global[:, s], min_samples, P['fs'])
        all_freezing_bouts.append(bouts)


    # ---------------------------------------------------------------
    #  4. Output Pre-allocation

    # behavior_freezing index layout (per epoch):
    #   [0]  Raw bout durations      (list of arrays, one per subject)
    #   [1]  Mean bout duration      (array, seconds)
    #   [2]  Number of bouts         (array, count)
    #   [3]  Total freeze time       (array, seconds)
    #   [4]  Freeze percentage       (array, %)
    #   [5]  Mean inter-bout Delta T (array, seconds; NaN if fewer than 2 bouts)
    #   [6]  Raw inter-bout Delta T  (list of arrays, one per subject)
    data = {
        'behavior_freezing':   [[None] * 7 for _ in range(n_epochs)],
        'behavior_nonfreezing': [[None]     for _ in range(n_epochs)],
        'events_behavior_idx':  [[None]     for _ in range(n_epochs)],
        'behavior_epochs':      [None       for _ in range(n_epochs)],
        'blocks':               []
    }


    # ---------------------------------------------------------------
    #  5. Per-Epoch Processing Loop

    for i in range(n_epochs):

        # Epoch boundaries and duration
        start_sample = epoch_bounds[i, 0]
        end_sample   = epoch_bounds[i, 1]
        epoch_len_s  = (end_sample - start_sample + 1) / P['fs']

        # Store the normalised signal segment for all subjects (S-by-M)
        data['behavior_epochs'][i] = raw_signal[start_sample:end_sample + 1, :].T

        # Per-subject accumulators
        bf_raw_dur = []                       # raw bout durations (s) per subject
        bf_mean    = np.zeros(num_subjects)   # mean bout duration (s)
        bf_num     = np.zeros(num_subjects)   # number of bouts
        bf_tot     = np.zeros(num_subjects)   # total freeze time (s)
        bf_freeze  = np.zeros(num_subjects)   # freeze percentage (%)
        bf_mean_dt = np.zeros(num_subjects)   # mean inter-bout Delta T (s)
        bf_raw_dt  = []                       # raw inter-bout Delta T (s)

        bnf_dur = []                          # non-freeze durations (s) per subject
        ev_idx  = [[None, None, None] for _ in range(num_subjects)]

        # Inner loop: process each subject
        for s in range(num_subjects):

            subj_f_bouts   = all_freezing_bouts[s]
            f_pairs_for_nf = np.zeros((0, 2), dtype=int)  # default: no freeze pairs

            if subj_f_bouts.shape[1] > 0 and epoch_len_s > 0:

                # Compute global end sample for each bout (onset + duration - 1)
                all_bout_ends = subj_f_bouts[0, :] + subj_f_bouts[1, :] - 1

                # Find bouts that overlap with this epoch
                overlap_idx = (
                    (subj_f_bouts[0, :] <= end_sample) &
                    (all_bout_ends >= start_sample)
                )
                raw_f_bouts = subj_f_bouts[:, overlap_idx]
                raw_f_ends  = all_bout_ends[overlap_idx]

                if raw_f_bouts.shape[1] > 0:

                    # Clip bout boundaries to the current epoch
                    act_starts_clip = np.maximum(raw_f_bouts[0, :], start_sample)
                    act_ends_clip   = np.minimum(raw_f_ends, end_sample)
                    dur_smp_clip    = act_ends_clip - act_starts_clip + 1

                    # Discard clipped fragments shorter than the minimum duration
                    valid_clip = dur_smp_clip >= min_samples

                    if np.any(valid_clip):

                        # Unclipped onset/offset for valid bouts
                        starts_orig_valid = raw_f_bouts[0, valid_clip]
                        ends_orig_valid   = raw_f_ends[valid_clip]

                        # ev_idx[s][0]: only bouts that STARTED in this epoch.
                        # Inherited bouts are excluded to avoid cross-epoch double-counting.
                        is_new_bout  = starts_orig_valid >= start_sample
                        ev_idx[s][0] = np.column_stack((
                            starts_orig_valid[is_new_bout],
                            ends_orig_valid[is_new_bout]
                        ))

                        # Clipped data for ALL valid bouts (new + inherited).
                        # Used for statistics and non-freeze gap computation.
                        starts_clip_valid = act_starts_clip[valid_clip]
                        dur_clip_valid    = dur_smp_clip[valid_clip]

                        # Freeze statistics
                        bf_raw_dur.append(dur_clip_valid / P['fs'])
                        bf_mean[s] = np.mean(bf_raw_dur[-1])
                        bf_num[s]  = len(dur_clip_valid)
                        bf_tot[s]  = np.sum(dur_clip_valid) / P['fs']

                        # Delta T: time between end of one bout and start of the next.
                        # Requires at least 2 bouts; set to NaN otherwise.
                        if len(starts_clip_valid) > 1:
                            dt_smp = (
                                starts_clip_valid[1:] -
                                (starts_clip_valid[:-1] + dur_clip_valid[:-1] - 1)
                            )
                            bf_raw_dt.append(dt_smp / P['fs'])
                            bf_mean_dt[s] = np.mean(bf_raw_dt[-1])
                        else:
                            bf_raw_dt.append(np.array([]))
                            bf_mean_dt[s] = np.nan

                        # Clipped freeze pairs for non-freeze gap computation
                        f_pairs_for_nf = np.column_stack((
                            starts_clip_valid,
                            starts_clip_valid + dur_clip_valid - 1
                        ))

                    else:
                        # All clipped fragments too short — treat as no freezing
                        ev_idx[s][0] = np.zeros((0, 2), dtype=int)
                        bf_raw_dur.append(np.array([]))
                        bf_raw_dt.append(np.array([]))
                        bf_mean_dt[s] = np.nan

                else:
                    # No globally detected bouts overlap this epoch
                    ev_idx[s][0] = np.zeros((0, 2), dtype=int)
                    bf_raw_dur.append(np.array([]))
                    bf_raw_dt.append(np.array([]))
                    bf_mean_dt[s] = np.nan

            else:
                # Subject has no bouts, or epoch has zero duration
                ev_idx[s][0] = np.zeros((0, 2), dtype=int)
                bf_raw_dur.append(np.array([]))
                bf_raw_dt.append(np.array([]))
                bf_mean_dt[s] = np.nan

            # Binary freeze mask for this epoch and subject (local 0-based indices)
            ev_idx[s][2] = freeze_mask_global[int(start_sample):int(end_sample) + 1, s]

            # Freeze percentage (guard against zero-length epochs)
            if epoch_len_s > 0:
                bf_freeze[s] = (bf_tot[s] / epoch_len_s) * 100

            # Non-freeze index pairs (0-based global indices)
            ev_idx[s][1] = get_nf_pairs(f_pairs_for_nf, start_sample, end_sample, min_samples)

            # Non-freeze bout durations (seconds)
            if len(ev_idx[s][1]) > 0:
                bnf_dur.append((ev_idx[s][1][:, 1] - ev_idx[s][1][:, 0] + 1) / P['fs'])
            else:
                bnf_dur.append(np.array([]))

        # Store epoch-level results
        data['behavior_freezing'][i][0] = bf_raw_dur
        data['behavior_freezing'][i][1] = bf_mean
        data['behavior_freezing'][i][2] = bf_num
        data['behavior_freezing'][i][3] = bf_tot
        data['behavior_freezing'][i][4] = bf_freeze
        data['behavior_freezing'][i][5] = bf_mean_dt
        data['behavior_freezing'][i][6] = bf_raw_dt

        data['behavior_nonfreezing'][i][0] = bnf_dur
        data['events_behavior_idx'][i][0]  = ev_idx


    # ---------------------------------------------------------------
    #  6. Block Analysis (Optional)
    #
    #  Groups consecutive events that share a common prefix into blocks of
    #  size N, then computes per-block aggregates across subjects.
    #
    #  Aggregation rules per metric:
    #    freeze %   -> mean across events in block
    #    bout count -> sum  across events in block (total bouts)
    #    duration   -> nanmean across events (replaced with 0 if all NaN)
    #    Delta T    -> nanmean across events (replaced with 0 if all NaN)

    if (
        'block_prefixes' in P and
        'block_sizes'    in P and
        'event_names'    in P and
        len(P['event_names']) > 0
    ):
        event_names = np.array(P['event_names'], dtype=str)

        for b_i, pref in enumerate(P['block_prefixes']):

            sz = P['block_sizes'][b_i]

            # Skip this entry if the prefix is empty or the block size is invalid
            if not pref or np.isnan(sz) or sz <= 0:
                continue

            # Find events whose names start with the target prefix (case-insensitive)
            pref_lower = pref.lower()
            match_idx  = np.where(
                [str(name).lower().startswith(pref_lower) for name in event_names]
            )[0]
            num_matches = len(match_idx)

            if num_matches == 0:
                continue

            num_blocks = int(np.ceil(num_matches / sz))

            # Pre-allocate block result matrices (subjects x blocks)
            block_freeze  = np.zeros((num_subjects, num_blocks))
            block_bout    = np.zeros((num_subjects, num_blocks))
            block_dur     = np.zeros((num_subjects, num_blocks))
            block_delta_t = np.zeros((num_subjects, num_blocks))
            block_labels  = []

            for b in range(num_blocks):

                # Slice of matched events belonging to this block
                idx_start    = b * int(sz)
                idx_end      = min((b + 1) * int(sz), num_matches)
                curr_matches = match_idx[idx_start:idx_end]

                # Epoch rows are offset by 2 (0=Full, 1=Baseline, 2+=Events)
                curr_epochs = curr_matches + 2

                # Collect per-epoch statistics for all subjects in this block
                temp_freeze  = np.zeros((num_subjects, len(curr_epochs)))
                temp_bout    = np.zeros((num_subjects, len(curr_epochs)))
                temp_dur     = np.zeros((num_subjects, len(curr_epochs)))
                temp_delta_t = np.zeros((num_subjects, len(curr_epochs)))

                for k, ep in enumerate(curr_epochs):
                    temp_freeze[:,  k] = data['behavior_freezing'][ep][4]
                    temp_bout[:,    k] = data['behavior_freezing'][ep][2]
                    temp_dur[:,     k] = data['behavior_freezing'][ep][1]
                    temp_delta_t[:, k] = data['behavior_freezing'][ep][5]

                # Aggregate across events within the block
                block_freeze[:, b] = np.mean(temp_freeze, axis=1)
                block_bout[:, b]   = np.sum(temp_bout,    axis=1)

                # Duration and Delta T: NaN-safe mean; replace remaining NaN with 0
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    dur_tmp = np.nanmean(temp_dur,     axis=1)
                    dt_tmp  = np.nanmean(temp_delta_t, axis=1)

                dur_tmp[np.isnan(dur_tmp)] = 0
                dt_tmp[np.isnan(dt_tmp)]   = 0

                block_dur[:,     b] = dur_tmp
                block_delta_t[:, b] = dt_tmp

                # Label: '<Prefix> <first>-<last>'  (1-based for readability, e.g. 'CS 1-5')
                block_labels.append(f"{pref} {idx_start + 1}-{idx_end}")

            # Append this block definition to the output list
            data['blocks'].append({
                'freeze':   block_freeze,
                'bout':     block_bout,
                'dur':      block_dur,
                'delta_t':  block_delta_t,
                'labels':   block_labels,
                'prefix':   pref,
                'size':     sz
            })

    return data, P