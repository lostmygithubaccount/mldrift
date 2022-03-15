# imports
import os
import mlflow
import argparse

import pandas as pd

from datetime import datetime, timedelta

from mldrift.tabular.lightgbm_diff import DataDiff

BASELINE_START = datetime(2000, 1, 1)
BASELINE_END = datetime(2000, 4, 1)
INTERVAL = timedelta(days=30)

# define functions
def main(args):
    # read in data
    print("reading in data...")
    df = pd.read_csv(args.data, index_col="date")
    df.index = pd.to_datetime(df.index)
    print(df.head())

    # hardcode the baseline
    print("getting baseline and test...")
    baseline = df[BASELINE_START:BASELINE_END]
    test = df[args.date - INTERVAL : args.date]

    # run the diff
    print("running data diff...")
    diff = DataDiff(baseline, test)

    with mlflow.start_run():
        diff.run()

    # return None
    pritn("done, exiting...")
    return None


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--data", type=str)
    parser.add_argument("--date", type=str)

    # parse args
    args = parser.parse_args()

    # convert date to datetime
    args.date = datetime.strptime(args.date, "%Y-%m-%d")

    # return args
    return args


# run script
if __name__ == "__main__":
    # parse args
    args = parse_args()

    # run main function
    main(args)
