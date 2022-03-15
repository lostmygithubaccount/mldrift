"""Defines the data diff logic between two datasets, relies on the DataSets API."""

import mlflow
import lightgbm
import numpy as np
import pandas as pd
import sklearn.ensemble
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import json
import copy
from datetime import datetime
from ._utils.constants import (
    METRIC_STATISTICAL_DISTANCE_ENERGY,
    METRIC_STATISTICAL_DISTANCE_WASSERSTEIN,
    METRIC_TYPE,
    METRIC_SCHEMA_VERSION_CUR,
    COLUMN_NAME,
    OUTPUT_METRIC_DRIFT_COEFFICIENT,
    OUTPUT_METRIC_DRIFT_CONTRIBUTION,
    DEBUG_DATASHIFT_MCC_TRAIN,
    DEBUG_DATASHIFT_MCC_TEST,
    DEBUG_DATASHIFT_MCC_ALL,
    SCORING_DATE,
    PIPELINE_START_TIME,
)
from enum import Enum
from scipy.stats import energy_distance
from scipy.stats import wasserstein_distance


# TODO - fix this hack
def get_distributions(pd_dataset, decimal=3):
    distributions = []

    for column in pd_dataset:
        binlabel, weight = np.unique(
            np.around(pd_dataset[column].values, decimal), return_counts=True
        )
        distributions.append(Distribution(column, binlabel, weight))

    return distributions


class Distribution:
    """Class represents distribution of a pandas data frame."""

    def __init__(self, name, binlabel, weight):
        """Distribution constructor.

        Creates distribution from list of binlabel and corresponding weight.

        :param name: Name of column of the distribution
        :type name: str
        :param binlabel: Array of numeric bin label which is central value of a bin
        :type binlabel: numpy.array[float]
        :param weight: Array of weight of each corresponding bin label at the same index
        :type weight: numpy.array[float]
        :return: A distribution object
        :rtype: Distribution
        """
        self.name = name
        self.binlabel = binlabel
        self.weight = weight


class MetricType(Enum):
    """Defines types of metrics returned in a data drift analysis.

    Use the :meth:`azureml.datadrift.DataDriftDetector.get_output` method of a
    :class:`azureml.datadrift.DataDriftDetector` object to return metrics.
    """

    column = 1
    dataset = 2


class Metric:
    """Represents a metric returned in a data drift analysis.

    The Metric class is for internal usage only. Use the :meth:`azureml.datadrift.DataDriftDetector.get_output`
    method of a :class:`azureml.datadrift.DataDriftDetector` object to return metrics.

    :param name: The name of the metric.
    :type name: str
    :param value: The value of the metric.
    :type value: float
    :param extended_properties: A dictionary of string to Python primitive type (int, float, str).
    :type extended_properties: dict
    :param schema_version: Optional schema version of the metric. Set to METRIC_SCHEMA_VERSION_CUR
        if no specific value set.
    :type schema_version: str
    """

    def __init__(
        self,
        name,
        value,
        extended_properties,
        schema_version=METRIC_SCHEMA_VERSION_CUR,
    ):
        """Metric constructor.

        :param name: The name of the metric.
        :type name: str
        :param value: The value of the metric.
        :type value: float
        :param extended_properties: A dictionary of string to Python primitive type (int, float, str).
        :type extended_properties: dict
        :param schema_version: Optional schema version of the metric. Set to METRIC_SCHEMA_VERSION_CUR
            if no specific value set.
        :type schema_version: str
        :return: A Metric object.
        :rtype: Metric
        """
        if name is None or value is None or extended_properties is None:
            raise ValueError(
                "name, value and extended_properties must not be None"
            )

        if not isinstance(extended_properties, dict):
            raise TypeError("extended_properties must be dictionary type")

        for k, v in extended_properties.items():
            self.validate_extended_properties_type(k, v)

        self.name = str(name)
        self.value = float(value)
        self.schema_version = str(schema_version)
        self.extended_properties = extended_properties

    def get_extended_properties(self):
        datetime_string_formats = [
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%SZ",
        ]
        export_ep = copy.deepcopy(self.extended_properties)
        datetime_keys = [SCORING_DATE, PIPELINE_START_TIME]
        for k in datetime_keys:
            if k in export_ep and isinstance(export_ep[k], str):
                for fmr in datetime_string_formats:
                    try:
                        export_ep[k] = datetime.strptime(export_ep[k], fmr)
                        break
                    except ValueError:
                        pass

        return export_ep

    def add_extended_properties(self, ep):
        if not isinstance(ep, dict):
            raise TypeError("extended_properties must be dictionary type")
        for k, v in ep.items():
            self.validate_extended_properties_type(k, v)
        self.extended_properties.update(ep)

    def validate_extended_properties_type(self, key, value):
        if not isinstance(key, str):
            raise TypeError("extended_properties key must be string type")
        if not isinstance(
            value, (int, float, str, bool, type(None), datetime)
        ):
            raise TypeError(
                "extended_properties value must be python built in type"
            )

    def __str__(self):
        return json.dumps(self, default=lambda o: o.__dict__)

    def __repr__(self):
        return self.__str__()


class DataDiff:
    """Class represents data diff service."""

    def __init__(self, base_dataset, diff_dataset):
        """Datadiff constructor.

        :param base_dataset: Dataframe of dataset for diff
        :type base_dataset: pandas.DataFrame
        :param diff_dataset: Dataframe of dataset for diff
        :type diff_dataset: pandas.DataFrame
        :return: A DataDiff object
        :rtype: DataDiff
        """
        # hack here
        # verify input dataset has same columns
        if base_dataset is None or diff_dataset is None:
            raise ValueError("Inputs cannot be None")
        if not DataDiff._is_dataframe_columns_equal(
            base_dataset, diff_dataset
        ):
            raise AssertionError(
                "Input datasets do not share identical columns"
            )
        if not DataDiff._is_dataframe_dtypes_supported(base_dataset):
            raise AssertionError(
                "Input data contains unsupported datatype column"
            )

        base_distributions = get_distributions(base_dataset)
        diff_distributions = get_distributions(diff_dataset)

        self.base_dataset = base_dataset
        self.diff_dataset = diff_dataset
        self.base_distributions = base_distributions
        self.diff_distributions = diff_distributions
        self.metrics = None
        self.model_dict = {}

    @staticmethod
    def get_supported_dtypes():
        """Get supported datatypes.

        :return: List of string
        :rtype: list(str)
        """
        return [
            "int16",
            "int32",
            "int64",
            "float16",
            "float32",
            "float64",
            "category",
        ]

    @staticmethod
    def is_supported_dtype(dtype):
        """Check if dtype is supported.

        :param dtype: dtype of a column
        :type dtype: pandas.DataFrame.dtype
        :return: whether given dtype is supported
        :rtype: boolean
        """
        if pd.api.types.is_numeric_dtype(dtype):
            return True

        if pd.api.types.is_categorical_dtype(dtype):
            return True

        return False

    @staticmethod
    def get_metrics_list():
        """Get all metrics name

        :return: List of tuples (metric_name, metric_type)
        :rtype: list(tuple)
        """

        metrics_list = [
            (OUTPUT_METRIC_DRIFT_COEFFICIENT, MetricType.dataset.name),
            (OUTPUT_METRIC_DRIFT_CONTRIBUTION, MetricType.column.name),
            (METRIC_STATISTICAL_DISTANCE_ENERGY, MetricType.column.name),
            (METRIC_STATISTICAL_DISTANCE_WASSERSTEIN, MetricType.column.name),
        ]

        return metrics_list

    @staticmethod
    def _is_dataframe_dtypes_supported(df):
        supported = True
        for c in df.dtypes:
            if not DataDiff.is_supported_dtype(c):
                supported = False
                break

        return supported

    @staticmethod
    def _is_dataframe_columns_equal(df1, df2):
        column_list1 = []
        column_list2 = []

        for i, v in df1.dtypes.iteritems():
            column_list1.append(
                "{}_{}".format(i, DataDiff._get_generic_type(v))
            )

        for i, v in df2.dtypes.iteritems():
            column_list2.append(
                "{}_{}".format(i, DataDiff._get_generic_type(v))
            )

        return set(column_list1) == set(column_list2)

    @staticmethod
    def _get_generic_type(str):
        if str in ["int16", "int32", "int64", "float16", "float32", "float64"]:
            return "numeric"
        else:
            return str

    @staticmethod
    def get_wasserstein_distance(x, y, x_weight, y_weight):
        """Calculate Wasserstein distance.

        :param x: distribution x bin central value
        :type x: numpy.array[float]
        :param y: distribution y bin central value
        :type y: numpy.array[float]
        :param x_weight: weight of bins at the corresponding
        index of distribution x
        :type x_weight: numpy.array[float]
        :param y_weight: weight of bins at the corresponding
        index of distribution y
        :type y_weight: numpy.array[float]
        :return: distance
        :rtype: float
        """
        return wasserstein_distance(x, y, x_weight, y_weight)

    @staticmethod
    def get_energy_distance(x, y, x_weight, y_weight):
        """Calculate energy distance.

        :param x: distribution x bin central value
        :type x: numpy.array[float]
        :param y: distribution y bin central value
        :type y: numpy.array[float]
        :param x_weight: weight of bins at the corresponding
        index of distribution x
        :type x_weight: numpy.array[float]
        :param y_weight: weight of bins at the corresponding
        index of distribution y
        :type y_weight: numpy.array[float]
        :return: distance
        :rtype: float
        """
        return energy_distance(x, y, x_weight, y_weight)

    @staticmethod
    def get_nicolo_datashift_detection(
        X1, X2, learner=lightgbm.LGBMClassifier
    ):
        """Run nicolo datashift detection.

        :param X1: Dataframe 1 for diff
        :type X1: pandas.DataFrame
        :param X2: Dataframe 2 for diff
        :type X2: pandas.DataFrame
        :param learner: classifier that supports fit and predict method call
        :type learner: sklearn.classifier
        :return: train_mcc
        :rtype: float
        :return: test_mcc
        :rtype: float
        :return: all_mcc
        :rtype: float
        :return: fea_imp_vec
        :rtype: numpy.array[float]
        """
        assert X1.shape[1] == X2.shape[1]

        data1 = X1
        data2 = X2

        target1 = np.zeros((data1.shape[0], 1))
        target2 = np.ones((data2.shape[0], 1))

        data = pd.concat([data1, data2], ignore_index=True)
        target = np.concatenate((target1, target2), 0)

        sss = sklearn.model_selection.StratifiedShuffleSplit(
            n_splits=1, test_size=0.5
        )

        for train, test in sss.split(data, target):
            clf1 = learner(n_jobs=-1)
            clf1.fit(data.iloc[train], target[train].flatten())

            all_preds = clf1.predict(data)
            all_mcc = sklearn.metrics.matthews_corrcoef(target, all_preds)

            test_preds = clf1.predict(data.iloc[test])
            test_mcc = sklearn.metrics.matthews_corrcoef(
                target[test], test_preds
            )

            train_preds = clf1.predict(data.iloc[train])
            train_mcc = sklearn.metrics.matthews_corrcoef(
                target[train], train_preds
            )

        fea_imp_dict = {}
        for index in range(len(clf1.feature_importances_)):
            fea_imp_dict[
                data.columns.values[index]
            ] = clf1.feature_importances_[index]

        return train_mcc, test_mcc, all_mcc, fea_imp_dict, clf1

    def get_nicolo_datashift_metrics(self, X1, X2):
        """Get nicolo datashift detection metrics and append to metrics object.

        :param X1: Dataframe 1 for diff
        :type X1: pandas.DataFrame
        :param X2: Dataframe 2 for diff
        :type X2: pandas.DataFrame
        :return: metrics
        :rtype: list[Metrics]
        """
        metrics = []

        (
            train_mcc,
            test_mcc,
            all_mcc,
            feature_importances,
            model,
        ) = DataDiff.get_nicolo_datashift_detection(X1, X2)

        self.model_dict["ds_model"] = model

        mtx = Metric(
            OUTPUT_METRIC_DRIFT_COEFFICIENT,
            test_mcc,
            {METRIC_TYPE: MetricType.dataset.name},
        )

        mtx.extended_properties.update({DEBUG_DATASHIFT_MCC_TRAIN: train_mcc})
        mtx.extended_properties.update({DEBUG_DATASHIFT_MCC_TEST: test_mcc})
        mtx.extended_properties.update({DEBUG_DATASHIFT_MCC_ALL: all_mcc})
        metrics.append(mtx)

        for key, value in feature_importances.items():
            metrics.append(
                Metric(
                    OUTPUT_METRIC_DRIFT_CONTRIBUTION,
                    value,
                    {METRIC_TYPE: MetricType.column.name, COLUMN_NAME: key},
                )
            )

        return metrics

    def get_statistical_distance_metrics(
        self, base_distributions, diff_distributions
    ):
        """Get distance metrics.

        Get all statistical distance metrics for all columns in input
        distributions and append to metrics object.

        :param base_distributions: List of distribuiton object for diff.
        :type base_distributions: list[Distribution]
        :param diff_distributions: List of distribuiton object for diff.
        :type diff_distributions: list[Distribution]
        :return: metrics
        :rtype: list[Metrics]
        """

        metrics = []

        for b_d in base_distributions:

            d_d = [d for d in diff_distributions if d.name == b_d.name][0]

            metrics.append(
                Metric(
                    METRIC_STATISTICAL_DISTANCE_WASSERSTEIN,
                    DataDiff.get_wasserstein_distance(
                        b_d.binlabel, d_d.binlabel, b_d.weight, d_d.weight
                    ),
                    {
                        METRIC_TYPE: MetricType.column.name,
                        COLUMN_NAME: b_d.name,
                    },
                )
            )
            metrics.append(
                Metric(
                    METRIC_STATISTICAL_DISTANCE_ENERGY,
                    DataDiff.get_energy_distance(
                        b_d.binlabel, d_d.binlabel, b_d.weight, d_d.weight
                    ),
                    {
                        METRIC_TYPE: MetricType.column.name,
                        COLUMN_NAME: b_d.name,
                    },
                )
            )

        return metrics

    def run(self):
        """Run all diff calculation and return metrics object.

        :return: metrics
        :rtype: list[Metrics]
        """
        # autolog for good measure
        mlflow.lightgbm.autolog()

        metrics = []
        metrics.extend(
            self.get_nicolo_datashift_metrics(
                self.base_dataset, self.diff_dataset
            )
        )
        metrics.extend(
            self.get_statistical_distance_metrics(
                self.base_distributions, self.diff_distributions
            )
        )

        self.metrics = metrics

        # log to mlflow
        # this is probably not efficient
        # list comprehension for the comedy
        [
            mlflow.log_metric(metric.name, metric.value)
            if "column" not in metric.extended_properties["metric_type"]
            else mlflow.log_metric(
                metric.extended_properties["column_name"] + "_" + metric.name,
                metric.value,
            )
            for metric in metrics
        ]

        return metrics
