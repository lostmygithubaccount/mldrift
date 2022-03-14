# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Parse, reorganize and present data drift calculation results."""

import os
import shutil
import tempfile
import warnings
from datetime import datetime, timedelta
from collections import OrderedDict

import matplotlib.pyplot as plt
import pandas as pd
from azureml.core import Datastore
from azureml.data import _dataset_diff as dsd
from azureml.datadrift._logging._telemetry_logger import _TelemetryLogger
from azureml.datadrift._utils.constants import (
    DATADRIFT_TYPE_DATASET,
    RUN_TYPE_KEY,
    RUN_TYPE_ADHOC,
    HAS_DRIFT,
    COLUMN_NAME,
    DEBUG_DATASHIFT_MCC_ALL,
    DEBUG_DATASHIFT_MCC_TEST,
    DEBUG_DATASHIFT_MCC_TRAIN,
    DRIFT_THRESHOLD,
    DRIFT_MAGNITUDE_TITLE,
    DRIFT_CONTRIBUTION_TITLE,
    Y_LABEL_PERCENTAGE,
    METRIC_COLUMN_METRICS,
    METRIC_DATASET_METRICS,
    METRIC_DATASHIFT_FEATURE_IMPORTANCE,
    METRIC_DATASHIFT_MCC_ALL,
    METRIC_DATASHIFT_MCC_TEST,
    METRIC_DATASHIFT_MCC_TRAIN,
    METRIC_SCHEMA_VERSION_DFT,
    METRIC_SCHEMA_VERSION_KEY,
    METRIC_TYPE,
    METRIC_TYPE_COLUMN,
    METRIC_TYPE_DATASET,
    OUTPUT_METRIC_DRIFT_COEFFICIENT,
    OUTPUT_METRIC_DRIFT_CONTRIBUTION,
    METRIC_FROM_DATASET,
    METRIC_FROM_DATASET_BOTH,
    PIPELINE_START_TIME,
    RUN_ID,
    SCORING_DATE,
    DUMMY_SERVICE,
    TGT_DST_START_DATE,
    TGT_DST_END_DATE,
    KEY_NAME_Drift_TYPE,
    KEY_NAME_BASE_DATASET_ID,
    KEY_NAME_TARGET_DATASET_ID,
)

from azureml.datadrift._utils.parameter_validator import ParameterValidator

module_logger = _TelemetryLogger.get_telemetry_logger(__name__)


def _all_outputs(
    datadriftdetector, start_time, end_time, run_id=None, activity_logger=None
):
    """Get a tuple of the drift results and metrics in a given time window.

        .. remarks::
            Given there are three run types, adhoc run, scheduled run and backfill run. This attribute will be used
            to retrieve corresponding results in different ways:

            * To retrieve adhoc run results, there is only one way: run_id should be a valid guid run id.
            * To retrieve scheduled runs and backfill runs' results, there are two different ways: assign a valid guid
                run id to run_id, or assign specific start_time and/or end_time while keeping run_id as None;
            * If run_id and start_time/end_time are not None in the same invoking, parameter validation exception
                will be thrown.

            The get_output attribute can be used to retrieve all outputs or partial outputs of scheduled runs in a
            specific time range between 'start_tzime' and 'end_time' (boundary included). User can also limited to
            results of an individual adhoc 'run_id'.

            * Principle for filtering is "overlapping": as long as there is an overlap between the actual result time
                (target dataset's [start date, end date] for dataset based drift)
                and the given [start_time, end_time], that result will be picked up.

            * The result contents for dataset based data drift instance will look like:

        .. code-block:: python

            # results : [{'drift_type': 'DatasetBased',
            #             'result':[{'has_drift': True, 'drift_threshold': 0.3,
            #                        'start_date': '2019-04-03', 'end_date': '2019-04-04',
            #                        'base_dataset_id': '4ac144ef-c86d-4c81-b7e5-ea6bbcd2dc7d',
            #                        'target_dataset_id': '13445141-aaaa-bbbb-cccc-ea23542bcaf9'}]}]
            # metrics : [{'drift_type': 'DatasetBased',
            #             'metrics': [{'schema_version': '0.1',
            #                          'start_date': '2019-04-03', 'end_date': '2019-04-04',
            #                          'baseline_dataset_id': '4ac144ef-c86d-4c81-b7e5-ea6bbcd2dc7d',
            #                          'target_dataset_id': '13445141-aaaa-bbbb-cccc-ea23542bcaf9'
            #                          'dataset_metrics': [{'name': 'datadrift_coefficient', 'value': 0.53459}],
            #                          'column_metrics': [{'feature1': [{'name': 'datadrift_contribution',
            #                                                            'value': 288.0},
            #                                                           {'name': 'wasserstein_distance',
            #                                                            'value': 4.858040000000001},
            #                                                           {'name': 'energy_distance',
            #                                                            'value': 2.7204799576545313}]}]}]}]

    :param logger:
    :param drift_type:
    :param baseline_dataset_id:
    :param target_dataset_id:
    :param start_time:
    :param end_time:
    :param run_id:
    :param activity_logger:
    :param daily_latest_only: flag of whether dedup to pick only latest results.
    :return:
    """
    drift_type = datadriftdetector.drift_type
    logger = activity_logger

    # Filter based on input params
    if run_id:
        metrics_got = _get_metrics(
            datadriftdetector,
            start_time,
            end_time,
            run_id,
            daily_latest_only=False,
            logger=logger,
        )
        metrics_valid = [
            m
            for m in metrics_got
            if _properties(m.get_extended_properties())[RUN_ID] == run_id
        ]
        if len(metrics_valid) < 1:
            msg = "No result found for run id query. Please make sure {} a valid run id.".format(
                run_id
            )
            logger.error(msg)
            raise ValueError(msg)
        else:
            logger.debug(
                "Run id results found. Total {} records for run id {}.".format(
                    len(metrics_valid), run_id
                )
            )
    # if run_id is None, pick only scheduled runs' results
    else:
        metrics_got = _get_metrics(
            datadriftdetector,
            start_time,
            end_time,
            run_id,
            daily_latest_only=True,
            logger=logger,
        )
        if len(metrics_got) < 1:
            logger.error(
                "No resoults found for scheduled/backfill runs. All results are from adhoc runs."
            )
            raise FileNotFoundError(
                "No resoults found for scheduled/backfill runs. All results are from adhoc runs."
            )
        else:

            logger.debug(
                "Results found for scheduled/backfill runs. Filtered in {} records.".format(
                    len(metrics_got)
                )
            )

        # Filter in given time range, only for schedule runs, if input run_id only, then no need to check time range.
        if drift_type == DATADRIFT_TYPE_DATASET:
            metrics_valid = [
                m
                for m in metrics_got
                if (
                    start_time
                    <= _properties(m.get_extended_properties())[
                        TGT_DST_START_DATE
                    ]
                    <= end_time
                    or start_time
                    <= _properties(m.get_extended_properties())[
                        TGT_DST_END_DATE
                    ]
                    <= end_time
                )
            ]
        else:
            metrics_valid = metrics_got

        if len(metrics_valid) < 1:
            msg = "Results found but not in given time range. Given time range: [{}, {}].".format(
                start_time, end_time
            )
            logger.error(msg)
            raise ValueError(msg)
        else:
            logger.debug(
                "Valid results found in given time range. Filtered in {} records between [{}, {}].".format(
                    len(metrics_valid), start_time, end_time
                )
            )

    return _build_results(drift_type, metrics_valid), _build_metrics(
        drift_type, metrics_valid
    )


# To avoid massive downloading:
# For output of specific run id, check if file name includes run id
# For output in a time range, add start_time / end_time
def _get_metrics(
    datadriftdetector,
    start_time,
    end_time,
    run_id,
    daily_latest_only=True,
    with_adhoc=False,
    logger=None,
):
    """

    :param datadriftdetector:
    :param start_time:
    :param end_time:
    :param run_id:
    :param daily_latest_only: pick only latest outputs for each single target date. For scheduled/backfill runs only.
    :param with_adhoc: if adhoc should be included in latest output dedup. True for _show() only.
    :return:
    """
    workspaceid = datadriftdetector.workspace.get_details()["workspaceid"]
    drift_type = datadriftdetector.drift_type
    did = datadriftdetector._id

    local_temp_root = os.path.join(tempfile.gettempdir(), workspaceid)
    os.makedirs(local_temp_root, exist_ok=True)

    # metrics will be a list of object DiffMetric, it has a member 'name' of the measurement and corresponding 'value',
    # as for detailed information such as date, or base/target dataset, they are in extended properties.
    metrics = []

    # For dedup. Considering there could be mixed results from scheduled run and adhoc run on each date, should find
    # latest pipeline start time for each date and:
    #
    #   for dataset based drift, there is actually no 'service' so it's just per date. The date is the 'start_date'
    #   in extended properties.
    #
    # Moreover, to make codes aligned, fake a service name for dataset based drift by using is drift id.
    # then dataset based drift latest pipeline times are all per service per date.
    latest_pipeline_times = {}

    if drift_type == DATADRIFT_TYPE_DATASET:
        metrics, latest_pipeline_times = _download_from_blob_metrics(
            datadriftdetector,
            local_temp_root,
            start_time,
            end_time,
            run_id,
            with_adhoc,
            logger,
        )
    else:
        return None

    # dedup data based on pipeline start time.
    # NOTICE:
    #  It's possible there will be multiple results in same target date folder, thus dedup is needed.
    #  (target date marked by 'start_date' for dataset based drift)
    #
    #  The principle of dedup is to pick the latest results. The latest one can be tell from latest pipeline start time
    #  on target dataset start_date (dataset based drift).
    if daily_latest_only is True:
        # daily_latest_only is True means retrieving scheduled/backfill runs only, therefore filter out adhoc runs.
        if with_adhoc is False:
            metrics = [
                m
                for m in metrics
                if m.extended_properties[RUN_TYPE_KEY] != RUN_TYPE_ADHOC
            ]

        if len(metrics) > 0:
            # then pick only latest outputs for each target date.
            if len(latest_pipeline_times) > 0:
                ondate = TGT_DST_START_DATE
                metrics = [
                    m
                    for m in metrics
                    if m.extended_properties[PIPELINE_START_TIME]
                    == latest_pipeline_times[did][
                        m.extended_properties[ondate]
                    ]
                ]
            logger.debug(
                "Results of Scheduled/backfill runs dedupled. Got {} latest records.".format(
                    len(metrics)
                )
            )
        else:
            error_msg = "No Scheduled/backfill run outputs found. Time range: {} to {}.".format(
                start_time, end_time
            )
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

    # Refine metrics and be compatible for old data:
    #   1. Assign default schema version (0.1) if unavailable in old data
    #   2. Refresh data drift coefficient key if unavailable and put mcc pairs in extended property
    mcc_train_value = 0.0
    mcc_all_value = 0.0
    for m in metrics:
        if m.name == METRIC_DATASHIFT_MCC_TRAIN:
            mcc_train_value = m.value
            metrics.remove(m)
        if m.name == METRIC_DATASHIFT_MCC_ALL:
            mcc_all_value = m.value
            metrics.remove(m)

    for m in metrics:
        if hasattr(m, METRIC_SCHEMA_VERSION_KEY) is False:
            m.schema_version = METRIC_SCHEMA_VERSION_DFT
            if m.name == METRIC_DATASHIFT_MCC_TEST:
                ep = _properties(m.get_extended_properties())
                m.name = OUTPUT_METRIC_DRIFT_COEFFICIENT
                ep[DEBUG_DATASHIFT_MCC_TEST] = m.value
                ep[DEBUG_DATASHIFT_MCC_TRAIN] = mcc_train_value
                ep[DEBUG_DATASHIFT_MCC_ALL] = mcc_all_value
            if m.name == METRIC_DATASHIFT_FEATURE_IMPORTANCE:
                m.name = OUTPUT_METRIC_DRIFT_CONTRIBUTION

    # tempfile.gettempdir() always points to same folder with same guid
    # thus empty temp folder to ensure new contents will be downloaded in each running.
    shutil.rmtree(local_temp_root, ignore_errors=True)

    logger.debug(
        "Valid deduped results found in raw diff outputs. Return {} records.".format(
            len(metrics)
        )
    )
    return metrics


def _download_from_blob_metrics(
    datadriftdetector,
    local_temp_root,
    start_time,
    end_time,
    run_id,
    with_adhoc,
    logger,
):
    data_store = Datastore.get_default(datadriftdetector.workspace)
    drift_type = datadriftdetector.drift_type
    datadrift_id = datadriftdetector._id
    latest_run_time = datadriftdetector._latest_run_time
    baseline_dataset_id = datadriftdetector._baseline_dataset_id
    target_dataset_id = datadriftdetector._target_dataset_id

    target_metrics = []
    latest_pipeline_times = {}
    # Considering there could be mixed results from scheduled run and adhoc run on each date, should find
    # latest pipeline start time for each date and:
    #
    #   for dataset based drift, the date is the 'start_date' in extended properties.
    #
    # Moreover, to make codes aligned, fake a service name for dataset based drift by using is drift id.
    # then dataset based drift latest pipeline times are all per service per scoring/start date.
    service_name = datadrift_id
    latest_pipeline_times[service_name] = {}

    metrics_rel_base = _get_metrics_path(
        drift_type=drift_type,
        datadrift_id=datadrift_id,
        datastore=data_store,
        logger=logger,
    )

    logger.info(
        "Relative metrics path confirmed. Drift id = {}, path = {}".format(
            datadrift_id, metrics_rel_base
        )
    )

    logger.info(
        "Looking for results. blob = {}, container = {}.".format(
            metrics_rel_base, data_store.container_name
        )
    )

    metrics_list = []
    # to avoid massive download, check all output json file lists with full path
    # (considering run id based query could be in random day so have to retrive all output file list)
    # if it's for a specific run id, just pick file name contains this run id and download it
    # if it's for a time range, then check yyyy/mm/dd in path and only download outputs in time range.
    blob_list = data_store.blob_service.list_blobs(
        container_name=data_store.container_name, prefix=metrics_rel_base
    )
    if len(data_store._filter_conflicting_blobs(blob_list)) > 0:
        for f in blob_list:
            # find outputs of current datadrift
            # handle different separator to be aligned with metrics_rel_base
            f_name = f.name.replace("\\", "/")
            if metrics_rel_base in f_name and f_name.endswith(".json"):
                # get date from path
                ymd_position = f_name.find(metrics_rel_base) + len(
                    metrics_rel_base
                )
                dt_str = os.path.split(f_name[ymd_position:])[0]
                dt = datetime.strptime(dt_str, "%Y/%m/%d")
                # if it's a valid target, download it and add local path to metrics list
                if (run_id and run_id in f.name) or (
                    not run_id and start_time <= dt <= end_time
                ):
                    data_store.download(
                        target_path=local_temp_root,
                        prefix=f.name,
                        show_progress=False,
                    )
                    metrics_full_path = os.path.join(
                        local_temp_root, *f_name.split("/")
                    )
                    metrics_list.append(metrics_full_path)
        logger.info(
            "Download list is ready. In total {} files under blob {} in container {}".format(
                len(metrics_list), metrics_rel_base, data_store.container_name
            )
        )

    count = 0
    for f in metrics_list:
        with open(f, "r") as metrics_json:
            data = metrics_json.read()
            metric = _decode_metric(data)

            target_metrics += metric
            count += len(metric)

            # find the latest pipeline start time
            for m in metric:
                if drift_type == DATADRIFT_TYPE_DATASET:
                    target_date = m.extended_properties[TGT_DST_START_DATE]

                if target_date not in latest_pipeline_times[service_name]:
                    latest_pipeline_times[service_name][
                        target_date
                    ] = datetime.min

                if (
                    PIPELINE_START_TIME in m.extended_properties
                    and m.extended_properties[PIPELINE_START_TIME]
                    > latest_pipeline_times[service_name][target_date]
                ):
                    run_type = (
                        m.extended_properties[RUN_TYPE_KEY]
                        if RUN_TYPE_KEY in m.extended_properties
                        else RUN_TYPE_ADHOC
                    )
                    if with_adhoc is True or (
                        with_adhoc is False and run_type != RUN_TYPE_ADHOC
                    ):
                        latest_pipeline_times[service_name][
                            target_date
                        ] = m.extended_properties[PIPELINE_START_TIME]
                        logger.info(
                            "Update latest pipeline time. For date {}, latest pipeline time is {}. Ahdoc {}.".format(
                                target_date,
                                m.extended_properties[PIPELINE_START_TIME],
                                with_adhoc,
                            )
                        )

    # found metrics (before filtering by time range or run id)
    if count > 0:
        logger.debug(
            "Download done. Before applying filters, downloaded {} rows of data diff results under path {}.".format(
                count, metrics_rel_base
            )
        )
    # find nothing, which means no files on blob storage, means outputs were not generated.
    else:
        error_msg = (
            "No metrics output found in container without applying any filter. "
            "blob = {}, container = {}. ".format(
                metrics_rel_base, data_store.container_name
            )
        )

        if run_id:
            error_msg += "For run id {} and ".format(run_id)
        else:
            error_msg += "In given time range [{} ... {}] for ".format(
                start_time, end_time
            )

        if drift_type == DATADRIFT_TYPE_DATASET:
            error_msg += "dataset based drift detector {} base dataset: {} and target dataset: {}.".format(
                datadrift_id, baseline_dataset_id, target_dataset_id
            )

        if latest_run_time:
            error_msg += (
                " Given latest run accomplished at {}, please check run details see if data of target dates"
                " are processed by the run, or if the run failed.".format(
                    latest_run_time
                )
            )
        else:
            error_msg = " Run history is empty, please accomplish at lease one run before retrieving outputs."
            raise FileNotFoundError(error_msg)

        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    return target_metrics, latest_pipeline_times


# To be compatible, set default value of drift type as None, then SDK will be able to know if invocation is from
# older version '_generate_script.py'. It will always go to datadrift_id path.
def _get_metrics_path(
    target_date=None,
    drift_type=None,
    datadrift_id=None,
    datastore=None,
    logger=None,
):
    """Get the metric path for a given drift type, instance target date and frequency of diff.

    :param service: datastore instance
    :type service: AbstractDatastore
    :param target_date: Diff instance start time. If none datetime portion is omitted.
    :type target_date: datetime.datetime
    :return: Relative paths to metric on datastore
    :rtype: str
    """
    # by default reach general output folder
    metrics_output_path = "datadrift/metrics/{}/".format(datadrift_id)

    # validate folder exists in blob storage
    if datastore:
        blobs = datastore.blob_service.list_blobs(
            container_name=datastore.container_name, prefix=metrics_output_path
        )
        blobs = datastore._filter_conflicting_blobs(blobs)
        if len(blobs) == 0:
            if logger:
                logger.error(
                    "Get output path failed. Container = {}, prefix = {}, drift id = {}".format(
                        datastore.container_name,
                        metrics_output_path,
                        datadrift_id,
                    )
                )

            return metrics_output_path

    if target_date is not None:
        metrics_output_path += "{}/".format(target_date.strftime("%Y/%m/%d"))

    return metrics_output_path


def _extended_property_datetime_from_str(extended_property):
    datetime_string_formats = [
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d",
    ]
    datetime_keys = [
        SCORING_DATE,
        PIPELINE_START_TIME,
        TGT_DST_START_DATE,
        TGT_DST_END_DATE,
    ]
    for k in datetime_keys:
        if k in extended_property and isinstance(extended_property[k], str):
            for fmr in datetime_string_formats:
                try:
                    extended_property[k] = datetime.strptime(
                        extended_property[k], fmr
                    )
                    break
                except ValueError:
                    pass
    return extended_property


def _decode_metric(json):
    """decode json into Metric object.

    :param json:
    :return:
    """
    decoded_metric = dsd.DiffMetric._list_from_json(json)

    for dm in decoded_metric:
        _properties(dm.extended_properties)

    return decoded_metric


def _properties(diff_extended_properties_got):
    """For dataset based results, DiffMetric attribute retruns a dict with information listed directly there,
    and retrieving can be done with return_dict['key']:

    :param diff_extended_properties_got:
    :return:
    """

    return _extended_property_datetime_from_str(diff_extended_properties_got)


def _build_single_result_content(drift_type, value, extended_properties):
    """Build a result dict.

    :param value: dataset level drift
    :param extended_properties: extended properties includes information such as scoring time, dataset id...
    :return: result dict.
    """
    result_content = {
        HAS_DRIFT: value > extended_properties[DRIFT_THRESHOLD],
        DRIFT_THRESHOLD: extended_properties[DRIFT_THRESHOLD],
    }

    # TODO: move strings to constants before check in
    if drift_type == DATADRIFT_TYPE_DATASET:
        result_content[TGT_DST_START_DATE] = extended_properties[
            TGT_DST_START_DATE
        ]
        result_content[TGT_DST_END_DATE] = extended_properties[
            TGT_DST_END_DATE
        ]
        result_content[KEY_NAME_BASE_DATASET_ID] = extended_properties[
            KEY_NAME_BASE_DATASET_ID
        ]
        result_content[KEY_NAME_TARGET_DATASET_ID] = extended_properties[
            KEY_NAME_TARGET_DATASET_ID
        ]

    return result_content


def _build_results(drift_type, raw_metrics):
    """Generate all results for queried time window or run id of some a datadriftdetector.

    :param raw_metrics: origin data diff calculation results.
    :return: a list of result dict.
    """
    results = []

    for metric in raw_metrics:
        ep = _properties(metric.get_extended_properties())
        if metric.name == OUTPUT_METRIC_DRIFT_COEFFICIENT:
            # Overall drift coefficient; add to results return object
            create_new_component = True

            if create_new_component:
                res = {KEY_NAME_Drift_TYPE: drift_type}

                # attach result content
                result_list = []
                result_list.append(
                    _build_single_result_content(drift_type, metric.value, ep)
                )
                res["result"] = result_list

                results.append(res)
    return results


def _create_metric_dict(drift_type, metric):
    """Create metrics dictionary.

    :param metric: Metric object
    :type metric: azureml.contrib.datadrift._datadiff.Metric
    :return: Dictionary of metrics delineated by service
    :rtype: dict()
    """
    ep = _properties(metric.get_extended_properties())
    _metric = {"name": metric.name, "value": metric.value}

    # general items
    metric_dict = (
        {METRIC_SCHEMA_VERSION_KEY: metric.schema_version}
        if hasattr(metric, "schema_version")
        else {}
    )

    # drift type sensitive items
    if drift_type == DATADRIFT_TYPE_DATASET:
        metric_dict[TGT_DST_START_DATE] = ep[TGT_DST_START_DATE]
        metric_dict[TGT_DST_END_DATE] = ep[TGT_DST_END_DATE]
        metric_dict[KEY_NAME_BASE_DATASET_ID] = ep[KEY_NAME_BASE_DATASET_ID]
        metric_dict[KEY_NAME_TARGET_DATASET_ID] = ep[
            KEY_NAME_TARGET_DATASET_ID
        ]

    # detail metrics
    # Use user friendly key without postfix such as "(Test)"
    if metric.name == METRIC_DATASHIFT_MCC_TEST:
        _metric["name"] = OUTPUT_METRIC_DRIFT_COEFFICIENT

    if ep[METRIC_TYPE] == METRIC_TYPE_DATASET:
        if METRIC_DATASET_METRICS not in metric_dict:
            metric_dict[METRIC_DATASET_METRICS] = []
        metric_dict[METRIC_DATASET_METRICS].append(_metric)

    elif ep[METRIC_TYPE] == METRIC_TYPE_COLUMN:
        column_dict = {ep[COLUMN_NAME]: [_metric]}
        # TODO: refine results returned from get_output for a special case after adding this protection:
        # if whole result json doesn't contain 'dataset' level drift result
        # then the 'results' from get_output will be empty while metrics still contains column level numbers.
        if METRIC_COLUMN_METRICS not in metric_dict:
            metric_dict[METRIC_COLUMN_METRICS] = []
        metric_dict[METRIC_COLUMN_METRICS].append(column_dict)

    return metric_dict


def _check_if_is_expected_metric(drift_type, output_metric, metric):
    """Check if the metric is in expectation for same baseline/target datasets.

    :param output_metric: metric with final outputs
    :param metric: metric records content from data diff results
    :return: bool
    """
    ep = _properties(metric.get_extended_properties())

    is_exected_metric = len(ep) > 0

    if is_exected_metric and drift_type == DATADRIFT_TYPE_DATASET:
        is_exected_metric &= (
            output_metric[TGT_DST_START_DATE] == ep[TGT_DST_START_DATE]
        )
        is_exected_metric &= (
            output_metric[TGT_DST_END_DATE] == ep[TGT_DST_END_DATE]
        )
        is_exected_metric &= (
            output_metric[KEY_NAME_BASE_DATASET_ID]
            == ep[KEY_NAME_BASE_DATASET_ID]
        )
        is_exected_metric &= (
            output_metric[KEY_NAME_TARGET_DATASET_ID]
            == ep[KEY_NAME_TARGET_DATASET_ID]
        )

    return is_exected_metric


def _build_metrics(drift_type, raw_metrics):
    """Build output dict.

    :param raw_metrics: origin data diff calculation results.
    :return: output metric list.
    """
    output_metrics = []

    for metric in raw_metrics:
        ep = _properties(metric.get_extended_properties())

        # Add to metrics return object
        create_new_component = True
        for m in output_metrics:
            if drift_type == DATADRIFT_TYPE_DATASET:
                create_new_component = False
                met_metric_exists = False
                for output_metric in m["metrics"]:
                    if _check_if_is_expected_metric(
                        drift_type, output_metric, metric
                    ):
                        name_prefix = (
                            (ep[METRIC_FROM_DATASET] + "_")
                            if (
                                METRIC_FROM_DATASET in ep
                                and ep[METRIC_FROM_DATASET]
                                != METRIC_FROM_DATASET_BOTH
                            )
                            else ""
                        )
                        _metric = {
                            "name": name_prefix + metric.name,
                            "value": metric.value,
                        }
                        met_metric_exists = True
                        # Add to already existing metric dictionary
                        if ep[METRIC_TYPE] == METRIC_TYPE_DATASET:
                            if METRIC_DATASET_METRICS not in output_metric:
                                output_metric[METRIC_DATASET_METRICS] = []
                            output_metric[METRIC_DATASET_METRICS].append(
                                _metric
                            )

                        elif ep[METRIC_TYPE] == METRIC_TYPE_COLUMN:
                            column_in_metrics = False
                            if METRIC_COLUMN_METRICS not in output_metric:
                                output_metric[METRIC_COLUMN_METRICS] = []
                            for c_metric in output_metric[
                                METRIC_COLUMN_METRICS
                            ]:
                                if ep[COLUMN_NAME] in c_metric:
                                    column_in_metrics = True
                                    column = c_metric[ep[COLUMN_NAME]]
                                    column.append(_metric)
                            if not column_in_metrics:
                                # Create column dict in column_metrics
                                column_dict = {ep[COLUMN_NAME]: [_metric]}
                                output_metric[METRIC_COLUMN_METRICS].append(
                                    column_dict
                                )
                if not met_metric_exists:
                    # Add new metric in metrics list
                    metric_dict = _create_metric_dict(drift_type, metric)
                    m["metrics"].append(metric_dict)

        if create_new_component:
            # Add metrics service dict
            metrics_list = []
            metric_dict = _create_metric_dict(drift_type, metric)
            metrics_list.append(metric_dict)

            metrics_component = {KEY_NAME_Drift_TYPE: drift_type}
            metrics_component["metrics"] = metrics_list
            output_metrics.append(metrics_component)
    return output_metrics


def _show(
    datadriftdetector,
    start_time=datetime.min,
    end_time=datetime.max,
    logger=None,
):
    """Show data drift trend in given time range.

    :param start_time:  Optional, start of presenting data time window in UTC, default is 0001-01-01 00:00:00
    :type start_time: datetime.datetime
    :param end_time: Optional, end of presenting data time window in UTC, default is 9999-12-31 23:59:59.999999
    :type end_time: datetime.datetime
    :param logger: Optional, active logger in API caller.
    :type with_details: activity logger ogject
    :return: diction of all figures. Key is service_name
    :rtype: dict()
    """
    start_time_valid = ParameterValidator.validate_datetime(start_time)
    end_time_valid = ParameterValidator.validate_datetime(end_time)

    # workspace = datadriftdetector.workspace
    workspace_name = datadriftdetector.workspace.name
    datadrift_name = datadriftdetector.name
    drift_type = datadriftdetector.drift_type
    # baseline_dataset_id = datadriftdetector._baseline_dataset_id
    # target_dataset_id = datadriftdetector._target_dataset_id

    metrics = _get_metrics(
        datadriftdetector,
        start_time,
        end_time,
        run_id=None,
        daily_latest_only=True,
        with_adhoc=True,
        logger=logger,
    )

    if len(metrics) == 0:
        raise FileNotFoundError("DataDrift results are unavailable.")

    metrics_services = set()
    if drift_type == DATADRIFT_TYPE_DATASET:
        metrics_services.add(DUMMY_SERVICE)

    # build metrics for graph showing of each service in valid date range
    # the graph is per service; In side each graph, there might be sub plots for different measurements.
    # considering the order in dict is not guaranteed, organize all contents with date key and sort by date
    # general data drift will be stored per day
    # detailed measurement will be sorted per measurement per column per day
    contents = {}
    found_nothing = True
    for s in metrics_services:
        if s not in contents:
            contents[s] = {}
        for m in metrics:
            metric_ep = _properties(m.get_extended_properties())
            # use target start date as index for dataset result
            if drift_type == DATADRIFT_TYPE_DATASET:
                stdate = metric_ep[TGT_DST_START_DATE]
                eddate = metric_ep[TGT_DST_END_DATE]
            # filtering rules:
            # For dataset based result, check if 'overlap' between target start/end date and input start/end time.
            if (
                start_time_valid <= stdate <= end_time_valid
                or start_time_valid <= eddate <= end_time_valid
            ):
                found_nothing = False
                if stdate not in contents[s]:
                    # insert also end date incase needed in future.
                    contents[s][stdate] = {"eddate": eddate}
                if metric_ep[METRIC_TYPE] == METRIC_TYPE_DATASET:
                    contents[s][stdate][m.name] = m.value
                if metric_ep[METRIC_TYPE] == METRIC_TYPE_COLUMN:
                    if m.name not in contents[s][stdate]:
                        contents[s][stdate][m.name] = {}
                    contents[s][stdate][m.name][
                        metric_ep[COLUMN_NAME]
                    ] = m.value

    if found_nothing is True:
        logger.error(
            "No available drift outputs found. (from {} to {}).".format(
                start_time, end_time
            )
        )
        raise ValueError(
            "No available drift outputs found. (from {} to {}).".format(
                start_time, end_time
            )
        )

    # produce figures
    figures = {}
    for c, c_metrics in contents.items():
        # sort by stdate to ensure correct order in graph
        ordered_content = OrderedDict(
            sorted(c_metrics.items(), key=lambda t: t[0])
        )

        fig_key = ""
        # environment information (alignment refined with extra spaces
        environment = "{}\n".format(workspace_name)
        if drift_type == DATADRIFT_TYPE_DATASET:
            fig_key += datadrift_name
            environment += "DataDrift Monitor : {}".format(datadrift_name)
            # temporarily comment out these codes untill dataset fix their bug of get_by_id
            # baseline_dataset_info = Dataset.get_by_id(workspace, baseline_dataset_id).name
            # target_dataset_info = Dataset.get_by_id(workspace, target_dataset_id).name
            # environment += "{} Baseline Dataset : {}\n{} Target Dataset : {}\n".\
            #     format("Unregistered" if not baseline_dataset_info else "Registered",
            #            baseline_dataset_id if not baseline_dataset_info else baseline_dataset_info,
            #            "Unregistered" if not target_dataset_info else "Registered",
            #            target_dataset_id if not target_dataset_info else target_dataset_info)

        figure = _generate_plot_figure(
            environment, ordered_content, drift_type
        )

        figures[fig_key] = figure

    return figures


def _generate_plot_figure(environment, ordered_content, drift_type):
    """Show trends for a metrics.

    :param environment: information of workspace, and baseline/target dataset id.
    :type environment: str
    :param ordered_content: all contents to present presorted by start date (dataset based)
    :type ordered_content: nested dict
    :param with_details: flag of show all or not
    :type with_details: bool
    :return: matplotlib.figure.Figure
    """
    stdates = list(ordered_content.keys())
    eddates = [v["eddate"] for v in list(ordered_content.values())]
    drifts_train = []
    drift_contribution = {}
    distance_energy = {}
    distnace_wasserstein = {}
    columns = []
    summary_contribute = {}
    bottoms_contribute = []
    columns_distance_e = []
    columns_distance_w = []

    for d in stdates:
        drifts_train.append(
            ordered_content[d][OUTPUT_METRIC_DRIFT_COEFFICIENT]
        )
        summary_contribute[d] = 0
        bottoms_contribute.append(0)
        for c in ordered_content[d][OUTPUT_METRIC_DRIFT_CONTRIBUTION].keys():
            if c not in columns:
                columns.append(c)

    for d in stdates:
        for c in columns:
            if OUTPUT_METRIC_DRIFT_CONTRIBUTION in ordered_content[d]:
                if (
                    c
                    not in ordered_content[d][OUTPUT_METRIC_DRIFT_CONTRIBUTION]
                ):
                    warnings.warn(
                        "Drift Contribution of column {} is unavailable.".format(
                            columns.index(c)
                        )
                    )
                    if c not in drift_contribution:
                        drift_contribution[c] = {}
                    if (
                        c
                        in ordered_content[d][OUTPUT_METRIC_DRIFT_CONTRIBUTION]
                    ):
                        drift_contribution[c][d] = ordered_content[d][
                            OUTPUT_METRIC_DRIFT_CONTRIBUTION
                        ][c]
                        summary_contribute[d] += ordered_content[d][
                            OUTPUT_METRIC_DRIFT_CONTRIBUTION
                        ][c]
                    else:
                        # if drift coefficient is missing on that day, set its ratio to 0.
                        drift_contribution[c][d] = 0

    # sum up daily coefficient
    daily_summary_contribution = list(summary_contribute.values())
    columns_contribution = list(drift_contribution.keys())

    # remove columns if distance is unavailable for all days.
    distance_energy = {
        k: v for k, v in distance_energy.items() if k in columns_distance_e
    }
    distnace_wasserstein = {
        k: v
        for k, v in distnace_wasserstein.items()
        if k in columns_distance_w
    }

    # show data drift
    width = 10
    height = 8
    # by default will show both drift coefficient and drift contribution so wide always divides into 2
    wdivid = 2
    hdivid = 1

    # when time range is wide and stdates are actually uncontinuous, just show stdates with drifts.
    xrange = pd.date_range(stdates[0], stdates[-1], freq="D")
    xrange = [x for x in xrange if x.date() in [y.date() for y in stdates]]

    # reduce x ticks if there are too many, ideally always keep 10 or 11 labels.
    tick_gap = int(len(xrange) / 10)
    if tick_gap > 1:
        zoom = int(len(xrange) / tick_gap)
        if zoom > 1:
            xrange = xrange[::zoom]

    xlabel = "Day"
    for d in range(len(stdates)):
        gap = eddates[d] - stdates[d]
        if gap == timedelta(7):
            xlabel = "Week of"
        elif gap > timedelta(1):
            xlabel = "{} days from".format(gap.days)

    # create canvas
    font_size_adjuster = 0
    figure = plt.figure(figsize=(width * wdivid, height * hdivid))
    plt.suptitle(environment, fontsize=(20 - font_size_adjuster))
    plt.subplots_adjust(bottom=0.1, top=0.75, hspace=0.5)
    plt.tight_layout()
    ax1 = plt.subplot(hdivid, wdivid, 1)

    # draw drift coefficient
    plt.sca(ax1)
    ax1.set_ylim(ymin=0, ymax=1.1)
    plt.plot_date(
        stdates, drifts_train, "-g", marker=".", linewidth=0.5, markersize=5
    )
    yvals = ax1.get_yticks()
    ax1.set_yticklabels(["{:,.2%}".format(v) for v in yvals])
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(Y_LABEL_PERCENTAGE, fontsize=16)
    plt.xticks(xrange, rotation=15)
    plt.title(DRIFT_MAGNITUDE_TITLE, fontsize=(20 - font_size_adjuster))

    # draw drift contribution
    color_indexes = []
    total_columns = len(columns)
    # in total 256 colors are supported.
    step = int(255 / total_columns)
    for i in range(total_columns):
        # Assign color index from 4 different sections in turn to avoid similar color among adjacent columns.
        color_index = (i % 4) * 64 + (int(i / 4)) * step
        color_indexes.append(color_index)
    colors = plt.cm.get_cmap("gist_ncar_r")

    ax2 = plt.subplot(hdivid, wdivid, 2)
    plt.sca(ax2)
    yvals = ax2.get_yticks()
    ax2.set_yticklabels(["{:,.2%}".format(v) for v in yvals])
    # ax2.xaxis.set_major_formatter(myFmt)
    for c in columns_contribution:
        # draw bar graph
        contribution = list(drift_contribution[c].values())
        bar_ratio = [
            x / y for x, y in zip(contribution, daily_summary_contribution)
        ]
        ax2.bar(
            stdates,
            height=bar_ratio,
            bottom=bottoms_contribute,
            color=colors(color_indexes)[columns.index(c)],
        )
        bottoms_contribute = [
            x + y for x, y in zip(bottoms_contribute, bar_ratio)
        ]

    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(Y_LABEL_PERCENTAGE, fontsize=16)
    plt.xticks(xrange, rotation=15)
    plt.title(DRIFT_CONTRIBUTION_TITLE, fontsize=(20 - font_size_adjuster))
    plt.legend(
        columns_contribution, bbox_to_anchor=(1.24, 1), loc=1, prop={"size": 7}
    )

    return figure
