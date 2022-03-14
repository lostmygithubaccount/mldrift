"""Some code to help visualize and run drift stuff. Should be reworked for actual usage.

Code written by a PM. Use at your own risk.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime, timedelta

from ..lightgbm_diff import DataDiff


def run_and_visualize(
    df: pd.DataFrame,
    start: datetime = None,
    end: datetime = None,
    baseline_df: pd.DataFrame = None,
    interval: timedelta = timedelta(days=30),
    rolling_baseline=False,
    plot_dark: bool = True,
):
    """Helper function for demonstration purposes to run the diff calculations and visualize results.

    return: Results
    rtype: Results
    """

    # use the first interval as the baseline
    if baseline_df is None:
        baseline_df = df[start : start + interval]

    results = {}
    current = start
    while current < end:
        test_df = df[current : current + interval]

        diff = DataDiff(baseline_df, test_df)
        metrics = diff.run()
        results[current] = metrics

        if rolling_baseline:
            baseline_df = test_df

        current += interval

    if plot_dark:
        plt.style.use("dark_background")

    _visualize(results)

    return results


def _visualize(results):

    metrics_x = list(results.keys())
    metrics_drift_y = [results[x][0].value for x in metrics_x]

    fig, ax = plt.subplots(figsize=(16, 8))

    ax.set_xlabel("time")
    ax.set_ylabel("drift metrics")
    ax.set_title("drift metric over time")

    plt.plot(metrics_x, metrics_drift_y)
