# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Utils functions"""
import pandas as pd

PD_INVALID_PERCENT_THRESHOLD = 0.15
PD_CATEGORICAL_CARD_MAX_PERCENT_TOTAL = 0.05
PD_CATEGORICAL_CARD_MAX = 100


class ColumnSummary:
    def __init__(
        self,
        name,
        total_ct,
        distinct_ct,
        valid_ct,
        invalid_ct,
        inferred_dtype,
        pd_dtype,
    ):
        self.name = name
        self.total_row_count = total_ct
        self.distinct_count = distinct_ct
        self.valid_row_count = valid_ct
        self.invalid_row_count = invalid_ct
        self.inferred_dtype = inferred_dtype
        self.pd_dtype = pd_dtype


def get_inferred_categorical_columns(summaries):
    return [
        c.name
        for c in summaries
        if pd.api.types.is_categorical_dtype(c.inferred_dtype)
    ]


def get_inferred_fillna_columns(summaries):
    return [
        c.name
        for c in summaries
        if pd.api.types.is_numeric_dtype(c.inferred_dtype)
        and c.invalid_row_count != 0
        and c.invalid_row_count / c.total_row_count
        < PD_INVALID_PERCENT_THRESHOLD
    ]


def get_inferred_drop_columns(summaries):
    return [
        c.name
        for c in summaries
        if (
            pd.api.types.is_numeric_dtype(c.inferred_dtype)
            and c.invalid_row_count / c.total_row_count
            >= PD_INVALID_PERCENT_THRESHOLD
        )
        or (
            not pd.api.types.is_numeric_dtype(c.inferred_dtype)
            and not pd.api.types.is_categorical_dtype(c.inferred_dtype)
        )
    ]


def get_pandas_df_summary(df):
    total_row_counts = len(df.index)
    max_cat_set_size = min(
        int(total_row_counts * PD_CATEGORICAL_CARD_MAX_PERCENT_TOTAL),
        PD_CATEGORICAL_CARD_MAX,
    )
    valid_counts_dict = df.count().to_dict()
    dtypes_dict = _get_dtypes_dict(df)

    summaries = []
    for col in df.columns:
        dcounts = len(df[col].unique())

        inferred_dtype = dtypes_dict[col]
        if (
            pd.api.types.is_numeric_dtype(inferred_dtype)
            or pd.api.types.is_string_dtype(inferred_dtype)
        ) and dcounts <= max_cat_set_size:
            inferred_dtype = pd.api.types.CategoricalDtype()

        summaries.append(
            ColumnSummary(
                name=col,
                total_ct=total_row_counts,
                distinct_ct=dcounts,
                valid_ct=valid_counts_dict[col],
                invalid_ct=total_row_counts - valid_counts_dict[col],
                inferred_dtype=inferred_dtype,
                pd_dtype=dtypes_dict[col],
            )
        )

    return summaries


def _get_dtypes_dict(df):
    dtypes_dict = {}
    for c, t in df.dtypes.iteritems():
        dtypes_dict[c] = t

    return dtypes_dict
