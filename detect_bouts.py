"""
detect_bouts.py

DESCRIPTION
    Finds all contiguous runs of ones (active bouts) in a binary signal
    and returns those that meet or exceed a minimum duration threshold.

    Edge detection is performed via np.diff() on a zero-padded copy of the
    signal, so bouts that begin at index 0 or end at the last sample are
    correctly captured without special-case handling.

    All indices are 0-based (Python convention).

USAGE
    bouts = detect_bouts(binary_signal, min_dur_samples, sample_rate)

INPUT
    binary_signal    - 1-D array-like of 0s and 1s (logical or numeric).
                       Any shape is accepted (flattened internally).
    min_dur_samples  - Minimum bout length in samples.
                       Bouts shorter than this are discarded.
    sample_rate      - Sampling rate in Hz.
                       Used only to convert bout duration to seconds (row 2).

OUTPUT
    bouts  - 3-by-B numpy array, where B is the number of qualifying bouts.
               Row 0 : onset index    (samples, 0-based)
               Row 1 : duration       (samples)
               Row 2 : duration       (seconds)
             Returns np.zeros((3, 0)) if no qualifying bouts are found.

EXAMPLE
    signal = [0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0]
    bouts  = detect_bouts(signal, min_dur_samples=3, sample_rate=1)

    # Two bouts meet the 3-sample minimum:
    #   Bout 0: onset=2,  duration=3 samples, 3.0 s
    #   Bout 1: onset=9,  duration=4 samples, 4.0 s
    #
    #   bouts =
    #     [[ 2.   9. ]    <- row 0: onset (samples, 0-based)
    #      [ 3.   4. ]    <- row 1: duration (samples)
    #      [ 3.   4. ]]   <- row 2: duration (seconds)

AUTHOR
    Flavio Mourao (mourao.fg@gmail.com)
    Texas A&M University - Department of Psychological and Brain Sciences
    University of Illinois Urbana-Champaign - Beckman Institute
    Federal University of Minas Gerais - Brazil

Started:     12/2023
Last update: 03/2026
"""

import numpy as np


def detect_bouts(binary_signal, min_dur_samples, sample_rate):
    """
    Find all contiguous runs of ones in a binary signal.

    Parameters
    ----------
    binary_signal   : array-like
        1-D binary signal (0s and 1s). Any orientation is accepted.
    min_dur_samples : int
        Minimum bout length in samples. Shorter bouts are discarded.
    sample_rate     : float
        Sampling rate in Hz. Used to compute duration in seconds (row 2).

    Returns
    -------
    bouts : ndarray, shape (3, B)
        B = number of qualifying bouts.
        Row 0 : onset index (samples, 0-based)
        Row 1 : duration    (samples)
        Row 2 : duration    (seconds)
        Returns np.zeros((3, 0)) if no qualifying bouts are found.
    """

    # ---------------------------------------------------------------
    #  1. Input Guard

    # Return empty result immediately if signal is empty or contains no active samples
    if len(binary_signal) == 0 or not np.any(binary_signal):
        return np.zeros((3, 0))

    # Force a flat boolean array for consistent edge detection
    sig = np.asarray(binary_signal).flatten() > 0


    # ---------------------------------------------------------------
    #  2. Edge Detection

    # Pad with False on both sides so that bouts starting at index 0
    # or ending at the last sample are detected as proper rising/falling edges.
    padded = np.concatenate(([False], sig, [False])).astype(int)
    edges  = np.diff(padded)

    # Rising edge  (+1): sample where the signal transitions 0 -> 1 (bout onset)
    # Falling edge (-1): sample AFTER the last 1, so subtract 1 to get bout end
    starts    = np.where(edges ==  1)[0]
    ends      = np.where(edges == -1)[0] - 1
    durations = ends - starts + 1


    # ---------------------------------------------------------------
    #  3. Minimum Duration Filter

    keep = durations >= min_dur_samples

    if not np.any(keep):
        return np.zeros((3, 0))

    s = starts[keep]
    d = durations[keep]

    # Assemble output: onset (0-based) | duration (samples) | duration (seconds)
    return np.vstack((s, d, d / sample_rate))