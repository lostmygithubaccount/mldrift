# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Parameter Validator."""
import numbers
import re
from datetime import datetime

from azureml.core import Workspace, ComputeTarget
from azureml.data import TabularDataset
from azureml.datadrift import alert_configuration
from azureml.datadrift._utils.constants import FREQUENCY_MONTH, FREQUENCY_WEEK, FREQUENCY_DAY, BACKFILL_BATCH_LIMIT

from .._logging._telemetry_logger import _TelemetryLogger
from .._schedule_state import ScheduleState

module_logger = _TelemetryLogger.get_telemetry_logger(__name__)
EMAIL_REGEX = re.compile(r"(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)")
NAME_REGEX = re.compile(r'^[a-zA-Z0-9][\w\-]{0,35}$')
NAME_ERROR_MSG = "Invalid name. Name must be up to 36 characters long and can only contain alphanumeric " \
                 "characters, dash and underscore"
FEATURE_NAME_REGEX = re.compile(r'(^([a-zA-Z0-9._][a-zA-Z0-9 ._-]{0,62})[a-zA-Z0-9._-]$|^[a-zA-Z0-9._]$)')
MAX_LATENCY = 1000


class ParameterValidator:
    @staticmethod
    def validate_workspace(input):
        if not isinstance(input, Workspace):
            raise TypeError("workspace must be a Workspace type")
        return input

    @staticmethod
    def validate_name(input):
        try:
            input = ParameterValidator._validate_str(input)
        except TypeError:
            raise TypeError("name must be a string")
        match = re.match(NAME_REGEX, input)
        if not match:
            raise ValueError(NAME_ERROR_MSG)
        return input

    @staticmethod
    def validate_state(state):
        if state == ScheduleState.Deleted.name:
            raise ValueError("DataDriftDetector object has been deleted.")

    @staticmethod
    def validate_dataset(input, none_ok=False):
        if not ((none_ok and input is None) or isinstance(input, TabularDataset)):
            raise TypeError("dataset must be of type TabularDataset")
        return input

    @staticmethod
    def validate_timeseries_dataset(input, none_ok=True):
        input = ParameterValidator.validate_dataset(input, none_ok)
        # TODO: Check trait of Dataset once they've implemented it
        fine, _ = input.timestamp_columns
        if not fine:
            raise TypeError("dataset must be time series, please use with_timestamp_columns()")
        return input

    @staticmethod
    def validate_compute_target(input, workspace, none_ok=True, not_exist_ok=False):
        def get_compute_name(compute_input):
            if isinstance(compute_input, str):
                return compute_input
            if isinstance(compute_input, ComputeTarget):
                return compute_input.name

        if not ((none_ok and input is None) or isinstance(input, str) or isinstance(input, ComputeTarget)):
            raise TypeError("compute_target must be a string (as its name) or a azureml.core.ComputeTarget object")
        if input is None:
            return input

        ct_name = get_compute_name(input)
        if not not_exist_ok and ct_name not in workspace.compute_targets:
            raise KeyError("compute_target does not exist in the supplied workspace.")
        if not not_exist_ok and workspace.compute_targets[ct_name]._compute_type != "AmlCompute":
            raise TypeError("Compute target must be an AmlCompute type")
        if not not_exist_ok and \
                workspace.compute_targets[ct_name].provisioning_state.capitalize() != "Succeeded":
            raise TypeError("Compute target must be in active state")

        return ct_name

    @staticmethod
    def validate_frequency(input, dataset_based=False, none_ok=True):
        frequencies = [FREQUENCY_DAY]
        if dataset_based:
            frequencies.extend([FREQUENCY_WEEK, FREQUENCY_MONTH])
        if not ((none_ok and input is None) or input in frequencies):
            raise TypeError("frequency must one of {}".format(frequencies))
        return input if input else frequencies[0]

    @staticmethod
    def validate_interval(input, none_ok=True, instance_logger=module_logger):
        if not ((none_ok and input is None) or (isinstance(input, int) and input > 0)):
            raise TypeError("interval must be a positive integer")
        if not (input == 1 or input is None):
            instance_logger.warning("Only interval=1 is supported for now, updating..")
        return 1  # TODO: 1 if input is None else input

    @staticmethod
    def validate_datetime(input, name="Input", none_ok=True):
        if not ((none_ok and input is None) or isinstance(input, datetime)):
            raise TypeError("{} must be a datetime".format(name))
        return input

    @staticmethod
    def validate_feature_list(input, none_ok=True):
        if not (none_ok and not input):
            if not isinstance(input, list):
                raise TypeError("feature_list must be a list")
            if len(input) > 200:
                raise ValueError("feature_list exceeds max length of 200")
            for i in input:
                if not isinstance(i, str) or "," in i:
                    raise ValueError("A feature must be a string and cannot contain any commas(,)")
                if not re.match(FEATURE_NAME_REGEX, i):
                    raise ValueError("Feature name should only contain characters, numbers, dashes, underscores, dots "
                                     "and whitespaces.")
        return input

    @staticmethod
    def validate_alert_configuration(input, none_ok=True):
        if not (none_ok and input is None):
            if not isinstance(input, alert_configuration.AlertConfiguration):
                raise TypeError("alert_config must be of type {}".format(
                    alert_configuration.AlertConfiguration.__module__
                    + alert_configuration.AlertConfiguration.__name__))
            input.email_addresses = ParameterValidator.validate_email_addresses(input.email_addresses)
        return input

    @staticmethod
    def validate_run_id(input):
        try:
            return ParameterValidator._validate_str(input)
        except TypeError:
            raise TypeError("run_id must be a string")

    @staticmethod
    def is_guid(test_str):
        pattern = r'^([a-zA-Z0-9]{8}-([a-zA-Z0-9]{4}-){3}[a-zA-Z0-9]{12})$'
        matched = re.match(pattern, test_str)
        return (matched is not None)

    @staticmethod
    def validate_drift_threshold(input, none_ok=True):
        msg = "threshold must be a positive float value between 0 and 1"
        if not (none_ok and input is None):
            if not (isinstance(input, numbers.Number)):
                raise TypeError(msg)
            if not (0 <= input <= 1):
                raise ValueError(msg)
        return input

    @staticmethod
    def validate_email_addresses(input):
        if not (isinstance(input, list)):
            raise TypeError("email_addresses must be of type list(str)")
        invalid_indexes = []
        for m in input:
            match = re.match(EMAIL_REGEX, m)
            if not match:
                invalid_indexes.append(str(input.index(m)))
        if invalid_indexes:
            raise ValueError("Invalid email address format found. Indexes are {}".format(", ".join(invalid_indexes)))
        return input

    @staticmethod
    def validate_latency(input, none_ok=True):
        if not ((none_ok and input is None) or (isinstance(input, int))):
            raise TypeError("latency must be of type int")
        if input and (input < 0 or input > MAX_LATENCY):
            raise ValueError("latency must be greater than or equal to 0 and less than or equal to 1000")
        return input

    @staticmethod
    def _validate_str(input):
        if not isinstance(input, str):
            raise TypeError("input must be a string")
        return input

    @staticmethod
    def validate_start_date_end_date(start_date, end_date, frequency):
        start_date = ParameterValidator.validate_datetime(start_date)
        end_date = ParameterValidator.validate_datetime(end_date)

        if end_date < start_date:
            raise ValueError("Parameter end_date needs to be later than start_date")

        delta = end_date - start_date
        span = 0
        if frequency == FREQUENCY_DAY:
            span = delta.days
        elif frequency == FREQUENCY_WEEK:
            span = delta.days / 7
        elif frequency == FREQUENCY_MONTH:
            _ = (end_date.year - start_date.year) * 12
            span = _ + end_date.month - start_date.month

        if span > BACKFILL_BATCH_LIMIT:
            raise ValueError("Current backfill batch size is {} which exceeds maximum backfill batch size {} {}s. "
                             "Reduce window between start_date and end_date"
                             .format(span, BACKFILL_BATCH_LIMIT, frequency))
        return start_date, end_date
