# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Contains core functionality for detecting data drift between two datasets in Azure Machine Learning.

Data drift is measured through datasets or deployments, and relies on the :class:`azureml.core.dataset.Dataset` API.
"""
import time
import warnings
from datetime import timezone, datetime, timedelta

from azureml.core import Dataset, Run
from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.core.workspace import Workspace

from azureml.datadrift import alert_configuration
from azureml.datadrift._logging._telemetry_logger import _TelemetryLogger
from azureml.datadrift._logging._telemetry_logger_context_adapter import \
    _build_general_log_context, _TelemetryLoggerContextAdapter as get_logger
from azureml.datadrift._restclient import DataDriftClient
from azureml.datadrift._restclient.models import (AlertConfiguration, CreateDataDriftDto, CreateDataDriftRunDto,
                                                  UpdateDataDriftDto)
from azureml.datadrift._result_handler import _all_outputs, _show
from azureml.datadrift._schedule_state import ScheduleState
from azureml.datadrift._utils.constants import (
    COMPUTE_TARGET_TYPE_AML, RUN_TYPE_ADHOC, RUN_TYPE_BACKFILL, DATADRIFT_TYPE_DATASET,
    DATADRIFT_CONSTRUCTOR, DATADRIFT_DELETE, DATADRIFT_GET_BY_ID,
    DATADRIFT_LIST, DATADRIFT_RUN, DATADRIFT_ENABLE_SCHEDULE, DATADRIFT_DISABLE_SCHEDULE,
    DATADRIFT_UPDATE, DATADRIFT_BACKFILL, DATADRIFT_GET_OUTPUT, DATADRIFT_SHOW,
    DATADRIFT_IN_PROGRESS_MSG, DEFAULT_LOOKBACK_CYCLES, LOG_DRIFT_ID, LOG_TOTAL_FEATURES,
    LOG_RUN_TYPE_, LOG_RUN_ID, LOG_PARENT_RUN_ID, LOG_INPUT_STARTTIME, LOG_INPUT_ENDTIME,
    LOG_ENV_SDK, COMPUTE_TARGET_SUCCEEDED, COMPUTE_TARGET_PROVISIONING, COMPUTE_TARGET_FAILED,
    COMPUTE_TARGET_CANCELED, WAIT_FOR_COMPLETION_TIMEOUT)
from azureml.exceptions import ComputeTargetException
from dateutil.relativedelta import relativedelta
from msrest.exceptions import HttpOperationError

from ._utils.parameter_validator import ParameterValidator

module_logger = _TelemetryLogger.get_telemetry_logger(__name__)

DEFAULT_COMPUTE_TARGET_NAME = "datadrift-server"
DEFAULT_VM_SIZE = "STANDARD_D2_V2"
DEFAULT_VM_MAX_NODES = 4
DEFAULT_DRIFT_THRESHOLD = 0.2


class DataDriftDetector:
    """Defines a data drift monitor that can be used to run data drift jobs in Azure Machine Learning.

    The DataDriftDetector class enables you to identify drift between a given baseline and target dataset.
    A DataDriftDetector object is created in a workspace by either specifying the baseline and target
    datasets directly.
    For more information, see https://aka.ms/datadrift.

    .. remarks::

        A DataDriftDetector object represents a data drift job definition that can be used to run three job
        run types:

        * an `adhoc` run for analyzing a specific day's worth of data;
          see the :meth:`azureml.datadrift.DataDriftDetector.run` method.
        * a scheduled run in a pipeline; see the :meth:`azureml.datadrift.DataDriftDetector.enable_schedule` method.
        * a backfill run to see how data changes over time; see the
          :meth:`azureml.datadrift.DataDriftDetector.backfill` method.

        The typical pattern for creating a DataDriftDetector is:

        * To create a dataset-based DataDriftDetector object, use
          :meth:`azureml.datadrift.DataDriftDetector.create_from_datasets`

        The following example shows how to create a dataset-based DataDriftDetector object.

        .. code-block:: inject notebooks/how-to-use-azureml/work-with-data/datadrift-tutorial
            /datadrift-tutorial.ipynb#datadrift-remarks-sample

        The DataDriftDetector constructor retrieves an existing data drift object associated with the workspace.

    :param workspace: The workspace in which to create the DataDriftDetector object.
    :type workspace: azureml.core.workspace.Workspace
    :param name: A unique name for the DataDriftDetector object.
    :type name: str
    :param baseline_dataset: Dataset to compare the target dataset against.
    :type baseline_dataset: azureml.data.tabular_dataset.TabularDataset
    :param target_dataset: Dataset to run either `adhoc` or scheduled DataDrift jobs for. Must be a time series.
    :type target_dataset: azureml.data.tabular_dataset.TabularDataset
    :param compute_target: Optional Azure Machine Learning ComputeTarget or ComputeTarget name.
        DataDriftDetector will create a compute target if one is not specified.
    :type compute_target: azureml.core.ComputeTarget or str
    :param frequency: Optional frequency indicating how often the pipeline is run.
        Supports "Day", "Week", or "Month".
    :type frequency: str
    :param feature_list: Optional whitelisted features to run the datadrift detection on. DataDriftDetector jobs
        will run on all features if ``feature_list`` is not specified. The feature list can contain characters,
        numbers, dashes, and whitespaces. The length of the list must be less than 200.
    :type feature_list: builtin.list[str]
    :param alert_config: Optional configuration object for DataDriftDetector alerts.
    :type alert_config: azureml.datadrift.AlertConfiguration
    :param drift_threshold: Optional threshold to enable DataDriftDetector alerts on. The value must be between 0
        and 1. A value of 0.2 is used when None is specified (the default).
    :type drift_threshold: float
    :param latency: Delay in hours for data to appear in dataset.
    :type latency: int
    :return: A DataDriftDetector object.
    :rtype: azureml.datadrift.DataDriftDetector
    :raises: KeyError, TypeError, ValueError
    """

    def __init__(self, workspace, name=None, baseline_dataset=None, target_dataset=None,
                 compute_target=None, frequency=None, feature_list=None, alert_config=None,
                 drift_threshold=None, latency=None):
        """Datadriftdetector constructor.

        The DataDriftDetector constructor is used to retrieve a cloud representation of a DataDriftDetector object
        associated with the provided workspace.

        :param workspace: The workspace in which to create the DataDriftDetector object.
        :type workspace: azureml.core.workspace.Workspace
        :param name: A unique name for the DataDriftDetector object.
        :type name: str
        :param baseline_dataset: Dataset to compare the target dataset against.
        :type baseline_dataset: azureml.data.tabular_dataset.TabularDataset
        :param target_dataset: Dataset to run either `adhoc` or scheduled DataDrift jobs for. Must be a time series.
        :type target_dataset: azureml.data.tabular_dataset.TabularDataset
        :param compute_target: Optional Azure Machine Learning ComputeTarget or ComputeTarget name.
            DataDriftDetector will create a compute target if one is not specified.
        :type compute_target: azureml.core.ComputeTarget or str
        :param frequency: Optional frequency indicating how often the pipeline is run.
            Supports "Day", "Week", or "Month".
        :type frequency: str
        :param feature_list: Optional whitelisted features to run the datadrift detection on. DataDriftDetector jobs
            will run on all features if ``feature_list`` is not specified. The feature list can contain characters,
            numbers, dashes, and whitespaces. The length of the list must be less than 200.
        :type feature_list: builtin.list[str]
        :param alert_config: Optional configuration object for DataDriftDetector alerts.
        :type alert_config: azureml.datadrift.AlertConfiguration
        :param drift_threshold: Optional threshold to enable DataDriftDetector alerts on. The value must be between 0
            and 1. A value of 0.2 is used when None is specified (the default).
        :type drift_threshold: float
        :param latency: Delay in hours for data to appear in dataset.
        :type latency: int
        :return: A DataDriftDetector object.
        :rtype: azureml.datadrift.DataDriftDetector
        :raises: KeyError, TypeError, ValueError
        :return: A DataDriftDetector object.
        :rtype: DataDriftDetector
        """
        if workspace is ...:
            # Instantiate an empty DataDriftDetector object. Will be initialized by DataDriftDetector.get()
            return

        workspace = ParameterValidator.validate_workspace(workspace)
        name = ParameterValidator.validate_name(name)
        baseline_dataset = ParameterValidator.validate_dataset(baseline_dataset)
        target_dataset = ParameterValidator.validate_timeseries_dataset(target_dataset)
        frequency = ParameterValidator.validate_frequency(frequency, dataset_based=True)
        feature_list = ParameterValidator.validate_feature_list(feature_list)
        drift_threshold = ParameterValidator.validate_drift_threshold(drift_threshold)
        compute_target = ParameterValidator.validate_compute_target(compute_target, workspace)
        alert_config = ParameterValidator.validate_alert_configuration(alert_config)
        latency = ParameterValidator.validate_latency(latency)

        log_context = _build_general_log_context(ws=workspace, img=None, type=DATADRIFT_TYPE_DATASET,
                                                 freq=frequency, latency=latency,
                                                 tft=len(feature_list) if feature_list else 0,
                                                 bsdid=baseline_dataset.id, tgdid=target_dataset.id,
                                                 log_env=LOG_ENV_SDK)
        self._logger = get_logger(__name__, log_context)
        with _TelemetryLogger.log_activity(self._logger, activity_name="constructor", custom_dimensions=log_context,
                                           telemetry_event_name=DATADRIFT_CONSTRUCTOR) as logger:
            dd_client = DataDriftClient(workspace.service_context)

            try:
                temp_dd = DataDriftDetector._get_datadrift_by_name(workspace, name)
                if temp_dd:
                    msg = "DataDriftDetector already exists. Name is {}, please use get_by_name() to " \
                          "retrieve it.".format(name)
                    logger.error(msg)
                    raise KeyError(msg)
            except HttpOperationError as e:
                if e.response.status_code != 404:
                    logger.error(e)
                pass

            if not compute_target:
                # Set to default workspace compute if it exists
                compute_target = DataDriftDetector._get_default_compute_target(workspace)

            if isinstance(compute_target, ComputeTarget):
                compute_target = compute_target.name

            if not drift_threshold:
                drift_threshold = DEFAULT_DRIFT_THRESHOLD

            # Ensure that datasets are saved to the workspace
            baseline_dataset._ensure_saved(workspace)
            target_dataset._ensure_saved(workspace)

            # Write object to service
            dto = CreateDataDriftDto(frequency=frequency,
                                     schedule_start_time=None,
                                     interval=1,
                                     alert_configuration=AlertConfiguration(
                                         email_addresses=alert_config.email_addresses)
                                     if alert_config else None,
                                     type=DATADRIFT_TYPE_DATASET,
                                     compute_target_name=compute_target,
                                     drift_threshold=drift_threshold,
                                     base_dataset_id=baseline_dataset.id,
                                     target_dataset_id=target_dataset.id,
                                     features=feature_list,
                                     job_latency=latency,
                                     name=name)
            client_dto = dd_client.create(dto, logger)
            DataDriftDetector._validate_client_dto(client_dto, logger)
            self._initialize(workspace, client_dto)

    def _initialize(self, workspace, client_dto):
        r"""Class DataDriftDetector Constructor helper.

        :param workspace: Object that points to workspace
        :type workspace: azureml.core.workspace.Workspace
        :param client_dto: DataDrift Client DTO object from service call
        :type client_dto: azureml.datadrift._restclient.models.DataDriftDto
        :return: A DataDriftDetector object
        :rtype: DataDriftDetector
        """
        self._workspace = workspace
        self._frequency = client_dto.frequency
        self._schedule_start = client_dto.schedule_start_time
        self._schedule_id = client_dto.schedule_id
        self._interval = client_dto.interval
        self._state = client_dto.state
        self._alert_config = client_dto.alert_configuration
        self._type = client_dto.type
        self._id = client_dto.id
        self._compute_target_name = client_dto.compute_target_name
        self._drift_threshold = client_dto.drift_threshold
        self._baseline_dataset_id = client_dto.base_dataset_id
        self._target_dataset_id = client_dto.target_dataset_id
        self._feature_list = client_dto.features if client_dto.features else []
        self._latency = client_dto.job_latency
        self._name = client_dto.name
        self._latest_run_time = client_dto.latest_run_time if hasattr(client_dto, 'latest_run_time') else None

        # Set alert configuration
        self._alert_config = alert_configuration.AlertConfiguration(client_dto.alert_configuration.email_addresses) \
            if client_dto.alert_configuration and client_dto.alert_configuration.email_addresses else None

        # Instantiate service client
        self._client = DataDriftClient(self.workspace.service_context)

        if not hasattr(self, '_logger') or LOG_DRIFT_ID not in self._logger.context:
            self._logger = get_logger(__name__, _build_general_log_context(ws=workspace, img=None, id=self._id,
                                                                           type=self._type, freq=self._frequency,
                                                                           interval=self._interval,
                                                                           state=self._state,
                                                                           latency=self._latency,
                                                                           tft=len(self._feature_list),
                                                                           bsdid=self._baseline_dataset_id,
                                                                           tgdid=self._target_dataset_id,
                                                                           log_env=LOG_ENV_SDK))

        self._updt_compute_target_and_train_dataset_in_general_log_context(self._logger)

    def __repr__(self):
        """Return the string representation of a DataDriftDetector object.

        :return: DataDriftDetector object string
        :rtype: str
        """
        return str(self.__dict__)

    @property
    def workspace(self):
        """Get the workspace of the DataDriftDetector object.

        :return: The workspace the DataDriftDetector object was created in.
        :rtype: azureml.core.workspace.Workspace
        """
        return self._workspace

    @property
    def name(self):
        """Get the name of the DataDriftDetector object.

        :return: The DataDriftDetector name.
        :rtype: str
        """
        return self._name

    @property
    def compute_target(self):
        """Get the compute target attached to the DataDriftDetector object.

        :return: The compute target.
        :rtype: azureml.core.ComputeTarget
        """
        return self._get_compute_target(self._compute_target_name)

    @property
    def frequency(self):
        """Get the frequency of the DataDriftDetector schedule.

        :return: A string of either "Day", "Week", or "Month"
        :rtype: str
        """
        return self._frequency

    @property
    def interval(self):
        """Get the interval of the DataDriftDetector schedule.

        :return: An integer value of time unit.
        :rtype: int
        """
        return self._interval

    @property
    def feature_list(self):
        """Get the list of whitelisted features for the DataDriftDetector object.

        :return: A list of feature names.
        :rtype: builtin.list[str]
        """
        return self._feature_list

    @property
    def drift_threshold(self):
        """Get the drift threshold for the DataDriftDetector object.

        :return: The drift threshold.
        :rtype: float
        """
        return self._drift_threshold

    @property
    def baseline_dataset(self):
        """Get the baseline dataset associated with the DataDriftDetector object.

        :return: Dataset type of the baseline dataset.
        :rtype: azureml.data.tabular_dataset.TabularDataset
        """
        return Dataset.get_by_id(self.workspace, self._baseline_dataset_id)

    @property
    def target_dataset(self):
        """Get the target dataset associated with the DataDriftDetector object.

        :return: The dataset type of the baseline dataset.
        :rtype: azureml.data.tabular_dataset.TabularDataset
        """
        return Dataset.get_by_id(self.workspace, self._target_dataset_id)

    @property
    def schedule_start(self):
        """Get the start time of the schedule.

        :return: A datetime object of schedule start time in UTC.
        :rtype: datetime.datetime
        """
        return self._schedule_start

    @property
    def alert_config(self):
        """Get the alert configuration for the DataDriftDetector object.

        :return: An AlertConfiguration object.
        :rtype: azureml.datadrift.AlertConfiguration
        """
        return self._alert_config

    @property
    def state(self):
        """Denotes the state of the DataDriftDetector schedule.

        :return: One of 'Disabled', 'Enabled', 'Deleted', 'Disabling', 'Enabling', 'Deleting', 'Failed',
            'DisableFailed', 'EnableFailed', 'DeleteFailed'.
        :rtype: str
        """
        return self._state

    @property
    def enabled(self):
        """Get the boolean value indicating whether the DataDriftDetector object is enabled.

        :return: A boolean value; True for enabled.
        :rtype: bool
        """
        return self._state == ScheduleState.Enabled.name

    @property
    def latency(self):
        """Get the latency of the DataDriftDetector schedule jobs (in hours).

        :return: The number of hours representing the latency.
        :rtype: int
        """
        return self._latency

    @property
    def drift_type(self):
        """Get the type of the DataDriftDetector, 'DatasetBased' is the only value supported for now.

        :return: The type of DataDriftDetector object.
        :rtype: str
        """
        return self._type

    @staticmethod
    def create_from_datasets(workspace, name, baseline_dataset, target_dataset, compute_target=None,
                             frequency=None, feature_list=None, alert_config=None, drift_threshold=None, latency=None):
        """Create a new DataDriftDetector object from a baseline tabular dataset and a target time series dataset.

        .. remarks::

            Dataset-based DataDriftDetectors enable you to calculate data drift between a baseline dataset, which
            must be a :class:`azureml.data.tabular_dataset.TabularDataset`, and a target dataset, which must be a
            time series dataset. A time series dataset is simply a :class:`azureml.data.tabular_dataset.TabularDataset`
            with the fine_grain_timestamp property. The DataDriftDetector can then run `adhoc` or scheduled jobs to
            determine if the target dataset has drifted from the baseline dataset.

            .. code-block:: python

                from azureml.core import Workspace, Dataset
                from azureml.datadrift import DataDriftDetector

                ws = Workspace.from_config()
                baseline = Dataset.get_by_name(ws, 'my_baseline_dataset')
                target = Dataset.get_by_name(ws, 'my_target_dataset')

                detector = DataDriftDetector.create_from_datasets(workspace=ws,
                                                                  name="my_unique_detector_name",
                                                                  baseline_dataset=baseline,
                                                                  target_dataset=target,
                                                                  compute_target_name='my_compute_target',
                                                                  frequency="Day",
                                                                  feature_list=['my_feature_1', 'my_feature_2'],
                                                                  alert_config=AlertConfiguration(email_addresses=['user@contoso.com']),
                                                                  drift_threshold=0.3,
                                                                  latency=1)

        :param workspace: The workspace to create the DataDriftDetector in.
        :type workspace: azureml.core.workspace.Workspace
        :param name: A unique name for the DataDriftDetector object.
        :type name: str
        :param baseline_dataset: Dataset to compare the target dataset against.
        :type baseline_dataset: azureml.data.tabular_dataset.TabularDataset
        :param target_dataset: Dataset to run either `adhoc` or scheduled DataDrift jobs for. Must be a time series.
        :type target_dataset: azureml.data.tabular_dataset.TabularDataset
        :param compute_target: Optional Azure Machine Learning ComputeTarget or ComputeTarget name.
            DataDriftDetector will create a compute target if one is not specified.
        :type compute_target: azureml.core.ComputeTarget or str
        :param frequency: Optional frequency indicating how often the pipeline is run.
            Supports "Day", "Week", or "Month".
        :type frequency: str
        :param feature_list: Optional whitelisted features to run the datadrift detection on. DataDriftDetector jobs
            will run on all features if ``feature_list`` is not specified. The feature list can contain characters,
            numbers, dashes, and whitespaces. The length of the list must be less than 200.
        :type feature_list: builtin.list[str]
        :param alert_config: Optional configuration object for DataDriftDetector alerts.
        :type alert_config: azureml.datadrift.AlertConfiguration
        :param drift_threshold: Optional threshold to enable DataDriftDetector alerts on. The value must be between 0
            and 1. A value of 0.2 is used when None is specified (the default).
        :type drift_threshold: float
        :param latency: Delay in hours for data to appear in dataset.
        :type latency: int
        :return: A DataDriftDetector object.
        :rtype: azureml.datadrift.DataDriftDetector
        :raises: KeyError, TypeError, ValueError
        """
        workspace = ParameterValidator.validate_workspace(workspace)
        name = ParameterValidator.validate_name(name)
        baseline_dataset = ParameterValidator.validate_dataset(baseline_dataset)
        target_dataset = ParameterValidator.validate_timeseries_dataset(target_dataset)
        frequency = ParameterValidator.validate_frequency(frequency, dataset_based=True)
        feature_list = ParameterValidator.validate_feature_list(feature_list)
        drift_threshold = ParameterValidator.validate_drift_threshold(drift_threshold)
        compute_target = ParameterValidator.validate_compute_target(compute_target, workspace)
        alert_config = ParameterValidator.validate_alert_configuration(alert_config)
        latency = ParameterValidator.validate_latency(latency)

        return DataDriftDetector(workspace, name=name, baseline_dataset=baseline_dataset,
                                 target_dataset=target_dataset, compute_target=compute_target,
                                 frequency=frequency, feature_list=feature_list,
                                 alert_config=alert_config, drift_threshold=drift_threshold,
                                 latency=latency)

    @staticmethod
    def get_by_name(workspace, name):
        """Retrieve a unique DataDriftDetector object for a given workspace and name.

        :param workspace: The workspace where the DataDriftDetector was created.
        :type workspace: azureml.core.workspace.Workspace
        :param name: The name of the DataDriftDetector object to return.
        :type name: str
        :return: A DataDriftDetector object.
        :rtype: azureml.datadrift.DataDriftDetector
        """
        workspace = ParameterValidator.validate_workspace(workspace)
        name = ParameterValidator.validate_name(name)

        log_context = _build_general_log_context(ws=workspace, type=DATADRIFT_TYPE_DATASET, log_env=LOG_ENV_SDK)
        logger = get_logger('datadriftdetector.get_by_name', log_context)
        with _TelemetryLogger.log_activity(logger, activity_name="get_by_name",
                                           custom_dimensions=log_context) as actlogger:
            actlogger.info("Getting DataDriftDetector object by name. Workspace: {}".format(workspace.name))
            try:
                client_dto = DataDriftDetector._get_datadrift_by_name(workspace, name, logger)
            except HttpOperationError:
                error_msg = "Could not find DataDriftDetector with given name. Workspace: {}".format(workspace.name)
                actlogger.error(error_msg)
                raise KeyError(error_msg)
            dd = DataDriftDetector(...)
            dd._initialize(workspace, client_dto)
            dd._updt_compute_target_and_train_dataset_in_general_log_context(actlogger.logger)
            actlogger.logger._update_general_contexts(id=dd._id, freq=dd._frequency, interval=dd._interval,
                                                      state=dd._state, latency=dd._latency, tft=len(dd._feature_list),
                                                      bsdid=dd._baseline_dataset_id, tgdid=dd._target_dataset_id)
            return dd

    @staticmethod
    def list(workspace, baseline_dataset=None, target_dataset=None):
        """Get a list of DataDriftDetector objects for the specified workspace and optional dataset.

        **NOTE:** Passing in only the ``workspace`` parameter will return all DataDriftDetector objects,
        defined in the workspace.

        :param workspace: The workspace where the DataDriftDetector objects were created.
        :type workspace: azureml.core.workspace.Workspace
        :param baseline_dataset: Baseline dataset to filter the return list.
        :type baseline_dataset: azureml.data.tabular_dataset.TabularDataset
        :param target_dataset: Target dataset to filter the return list.
        :type target_dataset: azureml.data.tabular_dataset.TabularDataset
        :return: A list of DataDriftDetector objects.
        :rtype: builtin.list[azureml.datadrift.DataDriftDetector]
        """
        workspace = ParameterValidator.validate_workspace(workspace)
        baseline_dataset = ParameterValidator.validate_dataset(baseline_dataset, none_ok=True)
        target_dataset = ParameterValidator.validate_dataset(target_dataset, none_ok=True)

        drift_type = DATADRIFT_TYPE_DATASET
        log_context = _build_general_log_context(ws=workspace, type=drift_type,
                                                 bsdid=baseline_dataset.id if baseline_dataset else None,
                                                 tgdid=target_dataset.id if target_dataset else None,
                                                 log_env=LOG_ENV_SDK)
        logger = get_logger('datadriftdetector.list', log_context)

        # if baseline_dataset:
        #     log_context['baseline_dataset'] = baseline_dataset.id
        # if target_dataset:
        #     log_context['target_dataset'] = target_dataset.id

        with _TelemetryLogger.log_activity(logger, activity_name="list", custom_dimensions=log_context,
                                           telemetry_event_name=DATADRIFT_LIST) as actlogger:
            try:
                dto_list = DataDriftDetector._get_datadrift_list(workspace,
                                                                 baseline_dataset=baseline_dataset,
                                                                 target_dataset=target_dataset, logger=actlogger)
            except HttpOperationError as e:
                _TelemetryLogger.log_exception(actlogger, e.message)
                raise
            dd_list = []
            for client_dto in dto_list:
                dd = DataDriftDetector(...)
                dd._initialize(workspace, client_dto)
                dd_list.append(dd)
            return dd_list

    @staticmethod
    def _get_default_compute_target(workspace):
        """Create the default computer target on demand and return the default compute target name.

        :param workspace: Object that points to the workspace
        :type workspace: azureml.core.workspace.Workspace
        :return: Compute target name
        :rtype: str
        """
        DataDriftDetector._create_aml_compute(workspace, DEFAULT_COMPUTE_TARGET_NAME)
        return DEFAULT_COMPUTE_TARGET_NAME

    @staticmethod
    def _get_datadrift_list(workspace, baseline_dataset=None, target_dataset=None,
                            logger=None, client=None):
        """Get list of DataDriftDetector objects from service.

        :param workspace: Object that points to the workspace
        :type workspace: azureml.core.workspace.Workspace
        :param baseline_dataset: Dataset to compare the target dataset against
        :type baseline_dataset: azureml.data.tabular_dataset.TabularDataset
        :param target_dataset: Dataset to run either adhoc or scheduled DataDrift jobs for. Must be Time Series
        :type target_dataset: azureml.data.tabular_dataset.TabularDataset
        :param logger: Activity logger for service call
        :type logger: datetime.datetime
        :param client: DataDriftDetector service client
        :type client: azureml.datadrift._restclient.DataDriftClient
        :return: List of DataDrift data transfer objects
        :rtype: list(azureml.datadrift._restclient.models.DataDriftDto)
        """
        dd_client = client if client else DataDriftClient(workspace.service_context)

        baseline_dataset_id = baseline_dataset.id if baseline_dataset else None
        target_dataset_id = target_dataset.id if target_dataset else None

        return list(dd_client.list(base_dataset_id=baseline_dataset_id,
                                   target_dataset_id=target_dataset_id, logger=logger))

    @staticmethod
    def _get_datadrift_by_name(workspace, name, logger=None, client=None):
        """Get list of DataDriftDetector objects from service.

        :param workspace: Object that points to the workspace
        :type workspace: azureml.core.workspace.Workspace
        :param name: Unique name of the DataDriftDetector object
        :param logger: Activity logger for service call
        :type logger: datetime.datetime
        :param client: DataDriftDetector service client
        :type client: azureml.datadrift._restclient.DataDriftClient
        :return: DataDrift data transfer object
        :rtype: azureml.datadrift._restclient.models.DataDriftDto
        """
        dd_client = client if client else DataDriftClient(workspace.service_context)

        return dd_client.get_by_name(name=name, logger=logger)

    @staticmethod
    def _get_training_dataset(workspace, drift_type):
        """Return None for dataset based drift.

        :param workspace:
        :return:
        """
        if drift_type == DATADRIFT_TYPE_DATASET:
            return None

        return None

    def _updt_compute_target_and_train_dataset_in_general_log_context(self, logger):
        # try to get train dataset for drift
        ds_training = DataDriftDetector._get_training_dataset(self._workspace, self._type)

        logger._update_general_contexts(trdid=ds_training.id if ds_training else None)

    @staticmethod
    def _create_aml_compute(workspace, name):
        """Create an AMLCompute using the specified name.

        :param workspace: The workspace to create the compute target in.
        :type workspace: azureml.core.workspace.Workspace
        :param name: The name of the AMLCompute target.
        :type name: str
        :return: An AMLCompute target.
        :rtype: azureml.core.compute.compute.AmlCompute
        """
        log_context = _build_general_log_context(ws=workspace, log_env=LOG_ENV_SDK)
        # TODO: De-provision compute if it's not run
        module_logger.info("Creating new compute target", extra={'properties': log_context})

        provisioning_config = Workspace.DEFAULT_CPU_CLUSTER_CONFIGURATION
        if name != Workspace.DEFAULT_CPU_CLUSTER_NAME:
            provisioning_config = AmlCompute.provisioning_configuration(vm_size=DEFAULT_VM_SIZE,
                                                                        max_nodes=DEFAULT_VM_MAX_NODES)

        try:
            aml_compute = AmlCompute.create(workspace, name, provisioning_config)
            aml_compute.wait_for_completion(show_output=True)
        except ComputeTargetException as e:
            err_msg = "Creating new compute target failed because {}".format(e.message)
            module_logger.info(err_msg, extra={'properties': log_context})
            raise ComputeTargetException(err_msg) from e

        return aml_compute

    def run(self, target_date, compute_target=None, create_compute_target=False, feature_list=None,
            drift_threshold=None):
        """Run a single point in time data drift analysis.

        :param target_date:  The target date of scoring data in UTC.
        :type target_date: datetime.datetime
        :param compute_target: Optional Azure Machine Learning ComputeTarget or ComputeTarget name.
            If not specified, a compute target is created automatically.
        :type compute_target: azureml.core.ComputeTarget or str
        :param create_compute_target: Indicates whether an Azure Machine Learning compute target is created
            automatically.
        :type create_compute_target: bool
        :param feature_list: Optional whitelisted features to run the datadrift detection on.
        :type feature_list: builtin.list[str]
        :param drift_threshold: Optional threshold to enable DataDriftDetector alerts on.
        :type drift_threshold: float
        :return: A DataDriftDetector run.
        :rtype: azureml.core.run.Run
        """
        ParameterValidator.validate_state(self.state)
        target_date = ParameterValidator.validate_datetime(target_date)
        compute_target = ParameterValidator.validate_compute_target(compute_target, self.workspace,
                                                                    not_exist_ok=create_compute_target)
        feature_list = ParameterValidator.validate_feature_list(feature_list)
        drift_threshold = ParameterValidator.validate_drift_threshold(drift_threshold)

        # remove extra information sepcifically for this run to avoid mis-leading in logs.
        self._logger._reset_to_general_context()
        self._logger._setkv(LOG_RUN_TYPE_, RUN_TYPE_ADHOC)
        with _TelemetryLogger.log_activity(self._logger, activity_name="run", custom_dimensions=self._logger.context,
                                           telemetry_event_name=DATADRIFT_RUN) as actlogger:
            if not compute_target:
                # Fallback to object's compute target
                compute_target = self.compute_target

            if not drift_threshold:
                # Fallback to object's drift threshold
                drift_threshold = self.drift_threshold

            if not feature_list:
                # Fallback to object's feature list
                feature_list = self.feature_list
            else:
                self._logger._setkv(LOG_TOTAL_FEATURES, len(feature_list) if feature_list else 0)

            if isinstance(compute_target, str):
                compute_target = self._get_compute_target(compute_target_name=compute_target,
                                                          create_compute_target=create_compute_target,
                                                          logger=actlogger)

            if isinstance(compute_target, ComputeTarget):
                compute_target = compute_target.name

            self._updt_compute_target_and_train_dataset_in_general_log_context(actlogger.logger)

            dto = CreateDataDriftRunDto(compute_target_name=compute_target,
                                        start_time=target_date.replace(tzinfo=timezone.utc).isoformat(),
                                        features=feature_list,
                                        drift_threshold=drift_threshold,
                                        run_type=RUN_TYPE_ADHOC)
            try:
                run_dto = self._client.run(self._id, dto, actlogger)
            except HttpOperationError as e:
                _TelemetryLogger.log_exception(actlogger, e.message)
                raise

            expname = run_dto.experiment_name \
                if hasattr(run_dto, 'experiment_name') and run_dto.experiment_name \
                else run_dto.data_drift_id
            exp = self.workspace.experiments[expname]
            run = Run(experiment=exp, run_id=run_dto.execution_run_id)

            self._logger._setkv(LOG_PARENT_RUN_ID, run.id)
            children = run.get_children()
            child_ids = ''
            if(children):
                for child in list(run.get_children()):
                    if child:
                        child_ids += ((', ' if len(child_ids) > 0 else '') + (child.id))
                self._logger._setkv(LOG_RUN_ID, child_ids)

            return run

    def backfill(self, start_date, end_date, compute_target=None, create_compute_target=False):
        """Run a backfill job over a given specified start and end date.

        See https://aka.ms/datadrift for details on data drift backfill runs.

        **NOTE:** Backfill is only supported on dataset-based DataDriftDetector objects.

        :param start_date: The start date of the backfill job.
        :type start_date: datetime.datetime
        :param end_date: The end date of the backfill job, inclusive.
        :type end_date: datetime.datetime
        :param compute_target: Optional Azure Machine Learning ComputeTarget or ComputeTarget name.
            DataDriftDetector will create a compute target if none is specified.
        :type compute_target: azureml.core.ComputeTarget or str
        :param create_compute_target: Indicates whether an Azure Machine Learning compute target is
            automatically created.
        :type create_compute_target: bool
        :return: A DataDriftDetector run.
        :rtype: azureml.core.run.Run
        """
        ParameterValidator.validate_state(self.state)
        start_date, end_date = ParameterValidator.validate_start_date_end_date(start_date, end_date, self.frequency)

        compute_target = ParameterValidator.validate_compute_target(compute_target, self.workspace,
                                                                    not_exist_ok=create_compute_target)
        # remove extra information sepcifically for this run to avoid mis-leading in logs.
        self._logger._reset_to_general_context()
        self._logger._setkv(LOG_RUN_TYPE_, RUN_TYPE_BACKFILL)
        with _TelemetryLogger.log_activity(self._logger, activity_name="backfill",
                                           custom_dimensions=self._logger.context,
                                           telemetry_event_name=DATADRIFT_BACKFILL) as actlogger:
            if not compute_target:
                # Fallback to object's compute target
                compute_target = self.compute_target

            if isinstance(compute_target, str):
                compute_target = self._get_compute_target(compute_target_name=compute_target,
                                                          create_compute_target=create_compute_target,
                                                          logger=actlogger)

            if isinstance(compute_target, ComputeTarget):
                compute_target = compute_target.name

            self._updt_compute_target_and_train_dataset_in_general_log_context(actlogger.logger)

            dto = CreateDataDriftRunDto(compute_target_name=compute_target,
                                        start_time=start_date.replace(tzinfo=timezone.utc).isoformat(),
                                        end_time=end_date.replace(tzinfo=timezone.utc).isoformat(),
                                        run_type=RUN_TYPE_BACKFILL)

            try:
                run_dto = self._client.run(self._id, dto, actlogger)
            except HttpOperationError as e:
                _TelemetryLogger.log_exception(actlogger, e.message)
                raise

            expname = run_dto.experiment_name \
                if hasattr(run_dto, 'experiment_name') and run_dto.experiment_name \
                else run_dto.data_drift_id
            exp = self.workspace.experiments[expname]
            run = Run(experiment=exp, run_id=run_dto.execution_run_id)

            return run

    def enable_schedule(self, create_compute_target=False, wait_for_completion=True):
        """Create a schedule to run dataset-based DataDriftDetector job.

        :param create_compute_target: Indicates whether an Azure Machine Learning compute target is created
            automatically.
        :type create_compute_target: bool
        :param wait_for_completion: Whether to wait for the enable operation to complete.
        :type wait_for_completion: bool
        """
        # remove extra information sepcifically for this run to avoid mis-leading in logs.
        ParameterValidator.validate_state(self.state)
        self._logger._reset_to_general_context()
        with _TelemetryLogger.log_activity(self._logger, activity_name="enable_schedule",
                                           custom_dimensions=self._logger.context,
                                           telemetry_event_name=DATADRIFT_ENABLE_SCHEDULE) as actlogger:
            # TODO: Add check for baseline dataset property being set
            if create_compute_target is True:
                compute_target_name = self._compute_target_name if self._compute_target_name \
                    else DEFAULT_COMPUTE_TARGET_NAME
            else:
                compute_target_name = self.compute_target.name

            compute_target = self._get_compute_target(compute_target_name, create_compute_target, actlogger)
            compute_target_type = compute_target.type
            if compute_target_type != COMPUTE_TARGET_TYPE_AML:
                raise AttributeError("Compute target type error. It must be of type {} while it is {}".format(
                    COMPUTE_TARGET_TYPE_AML, compute_target_type))

            try:
                self._state = ScheduleState.Enabled.name
                dto = self._update_schedule(actlogger, wait_for_completion)
                self._schedule_id = dto.schedule_id
                self._schedule_start = dto.schedule_start_time if dto.schedule_start_time else None
                self._state = dto.state
            except HttpOperationError or SystemError as e:
                latest_state = self._get_monitor_state(actlogger)
                self._state = ScheduleState.EnableFailed.name if latest_state is None else latest_state
                if DATADRIFT_IN_PROGRESS_MSG not in e.message:
                    actlogger.error("Unable to enable the schedule. "
                                    "Workspace: {}, drift id: {}, current scheduling state: {}, error message: {}.".
                                    format(self.workspace.name, self._id, self._state, e.message))
                    raise
                actlogger.info("Enable task is still running. Please check the status by calling .get() and .state")

    def disable_schedule(self, wait_for_completion=True):
        """Disable the schedule for the DataDriftDetector object.

        :param wait_for_completion: Whether to wait for the disable operation to complete.
        :type wait_for_completion: bool
        """
        # remove extra information specifically for this run to avoid mis-leading in logs.
        ParameterValidator.validate_state(self.state)
        self._logger._reset_to_general_context()
        with _TelemetryLogger.log_activity(self._logger, activity_name="disable_schedule",
                                           custom_dimensions=self._logger.context,
                                           telemetry_event_name=DATADRIFT_DISABLE_SCHEDULE) as actlogger:
            try:
                self._state = ScheduleState.Disabled.name
                dto = self._update_schedule(actlogger, wait_for_completion)
                self._state = dto.state
            except HttpOperationError or SystemError as e:
                latest_state = self._get_monitor_state(actlogger)
                self._state = ScheduleState.DisableFailed.name if latest_state is None else latest_state
                if DATADRIFT_IN_PROGRESS_MSG not in e.message:
                    actlogger.error("Unable to disable the schedule. "
                                    "Workspace: {}, drift id: {}, current scheduling state: {}, error message: {}.".
                                    format(self.workspace.name, self._schedule_id, self._state, e.message))
                    raise
                actlogger.info("Disable task is still running. Please check the status by calling .get() and .state")

    def delete(self, wait_for_completion=True):
        """Delete the schedule for the DataDriftDetector object.

        :param wait_for_completion: Whether to wait for the delete operation to complete.
        :type wait_for_completion: bool
        """
        # remove extra information specifically for this run to avoid mis-leading in logs.
        ParameterValidator.validate_state(self.state)
        console = DataDriftDetector._get_progress_logger()
        self._logger._reset_to_general_context()
        with _TelemetryLogger.log_activity(self._logger, activity_name="delete",
                                           custom_dimensions=self._logger.context,
                                           telemetry_event_name=DATADRIFT_DELETE) as actlogger:
            try:
                self._state = ScheduleState.Deleted.name
                dto = self._delete_remote(actlogger, wait_for_completion)

                if dto.state is ScheduleState.Deleted.name:
                    console("Delete succeeded!")
                self._state = dto.state
                self._name = dto.name
            except HttpOperationError or SystemError as e:
                latest_state = self._get_monitor_state(actlogger)
                self._state = ScheduleState.DeleteFailed.name if latest_state is None else latest_state
                if DATADRIFT_IN_PROGRESS_MSG not in e.message:
                    actlogger.error("Unable to delete monitor. "
                                    "Workspace: {}, drift id: {}, current scheduling state: {}, error message: {}.".
                                    format(self.workspace.name, self._schedule_id, self._state, e.message))
                    raise
                actlogger.info("Delete task is still running. Please check the status by calling .get() and .state")

    def get_output(self, start_time=None, end_time=None, run_id=None):
        """Get a tuple of the drift results and metrics for a specific DataDriftDetector over a given time window.

        .. remarks::

            This method returns a tuple of drift results and metrics for either a time window or run ID
            based on the type of run: an `adhoc` run, a scheduled run, and a backfill run.

            * To retrieve `adhoc` run results, there is only one way: ``run_id`` should be a valid GUID.

            * To retrieve scheduled runs and backfill run results, there are two different ways: either assign a
              valid GUID to ``run_id`` or assign a specific ``start_time`` and/or ``end_time`` (inclusive) while
              keeping ``run_id`` as None.

            * If ``run_id``, ``start_time``, and ``end_time`` are not None in the same method call, a parameter
              validation exception is raised.

            **NOTE:** Specify either ``start_time`` and ``end_time`` parameters or the ``run_id`` parameter, but
            not both.

            It's possible that there are multiple results for the same target date (target date means target dataset
            start date for dataset-based drift). Therefore, it's necessary to identify and handle duplicate results.
            For dataset-based drift, if results are for the same target date, then they are duplicated results.
            The ``get_output`` method will dedup any duplicated results by one rule:
            always pick up the latest generated results.

            The ``get_output`` method can be used to retrieve all outputs or partial outputs of scheduled runs in a
            specific time range between ``start_time`` and ``end_time`` (boundary included). You can also limit the
            results of an individual `adhoc` by specifying the ``run_id``.

            Use the following guidelines to help interpret results returned from the ``get_output`` method:

            * Principle for filtering is "overlapping": as long as there is an overlap between the actual result time
              (dataset-based: target dataset [start date, end date]) and the given [``start_time``, ``end_time``],
              then the result will be picked up.

            * If there are multiple outputs for one target date because the drift calculation was executed
              several times against that day, then only the latest output will be picked by default.

            * Given there are multiple types of a data drift instance, the result contents could be various.

            For dataset-based results, the output will look like:

            .. code-block:: python

                results : [{'drift_type': 'DatasetBased',
                            'result':[{'has_drift': True, 'drift_threshold': 0.3,
                                       'start_date': '2019-04-03', 'end_date': '2019-04-04',
                                       'base_dataset_id': '4ac144ef-c86d-4c81-b7e5-ea6bbcd2dc7d',
                                       'target_dataset_id': '13445141-aaaa-bbbb-cccc-ea23542bcaf9'}]}]
                metrics : [{'drift_type': 'DatasetBased',
                            'metrics': [{'schema_version': '0.1',
                                         'start_date': '2019-04-03', 'end_date': '2019-04-04',
                                         'baseline_dataset_id': '4ac144ef-c86d-4c81-b7e5-ea6bbcd2dc7d',
                                         'target_dataset_id': '13445141-aaaa-bbbb-cccc-ea23542bcaf9'
                                         'dataset_metrics': [{'name': 'datadrift_coefficient', 'value': 0.53459}],
                                         'column_metrics': [{'feature1': [{'name': 'datadrift_contribution',
                                                                           'value': 288.0},
                                                                          {'name': 'wasserstein_distance',
                                                                           'value': 4.858040000000001},
                                                                          {'name': 'energy_distance',
                                                                           'value': 2.7204799576545313}]}]}]}]

        :param start_time: The start time of the results window in UTC. If None (the default) is specified, then
            the most recent 10th cycle's results are used as the start time. For example, if frequency of the
            data drift schedule is day, then ``start_time`` is 10 days. If frequency is week, then ``start_time``
            is 10 weeks.
        :type start_time: datetime.datetime, optional
        :param end_time: The end time of the results window in UTC. If None (the default) is specified, then
            the current day UTC is used as the end time.
        :type end_time: datetime.datetime, optional
        :param run_id: A specific run ID.
        :type run_id: int, optional
        :return: A tuple of a list of drift results and a list of individual dataset and columnar metrics.
        :rtype: tuple(builtin.list, builtin.list)
        """
        ParameterValidator.validate_state(self.state)
        if run_id and (start_time or end_time):
            raise ValueError("Either run_id or start_time/end_time should be None.")

        start_time, end_time = self._get_valid_time_range(start_time=start_time, end_time=end_time)

        run_id = ParameterValidator.validate_run_id(run_id) if run_id is not None else run_id

        # remove extra information specifically for this run to avoid mis-leading in logs.
        self._logger._reset_to_general_context()
        if run_id:
            self._logger._setkv(LOG_RUN_ID, run_id)
        else:
            self._logger._setkv(LOG_INPUT_STARTTIME, str(start_time))
            self._logger._setkv(LOG_INPUT_ENDTIME, str(end_time))
        with _TelemetryLogger.log_activity(self._logger, activity_name="get_output",
                                           custom_dimensions=self._logger.context,
                                           telemetry_event_name=DATADRIFT_GET_OUTPUT) as actlogger:

            try:
                dto = self._get_by_id(self.workspace, self._id)
            except HttpOperationError as e:
                _TelemetryLogger.log_exception(actlogger, e.message)
                raise
            if hasattr(dto, 'latest_run_time'):
                self._latest_run_time = dto.latest_run_time

        return _all_outputs(self, start_time, end_time, run_id, actlogger)

    def update(self, compute_target=..., feature_list=..., schedule_start=..., alert_config=...,
               drift_threshold=..., wait_for_completion=True):
        r"""Update the schedule associated with the DataDriftDetector object.

        Optional parameter values can be set to ``None``, otherwise they default to their existing values.

        :param compute_target: Optional Azure Machine Learning ComputeTarget or ComputeTarget name.
            DataDriftDetector will create a compute target if this parameter is not specified.
        :type compute_target: azureml.core.ComputeTarget or str
        :param feature_list: Whitelisted features to run the datadrift detection on.
        :type feature_list: builtin.list[str]
        :param schedule_start: The start time of the data drift schedule in UTC.
        :type schedule_start: datetime.datetime
        :param alert_config: Optional configuration object for DataDriftDetector alerts.
        :type alert_config: azureml.datadrift.AlertConfiguration
        :param drift_threshold: The threshold to enable DataDriftDetector alerts on.
        :type drift_threshold: float
        :param wait_for_completion: Whether to wait for the enable/disable/delete operations to complete.
        :type wait_for_completion: bool
        :return: self
        :rtype: azureml.datadrift.DataDriftDetector
        """
        # remove extra information sepcifically for this run to avoid mis-leading in logs.
        ParameterValidator.validate_state(self.state)
        self._logger._reset_to_general_context()
        with _TelemetryLogger.log_activity(self._logger, activity_name="update",
                                           custom_dimensions=self._logger.context,
                                           telemetry_event_name=DATADRIFT_UPDATE) as actlogger:
            if compute_target is not ...:
                compute_target = ParameterValidator.validate_compute_target(compute_target, self.workspace)
                if not compute_target:
                    self._compute_target_name = self._get_default_compute_target(self.workspace)
                else:
                    self._compute_target_name = compute_target
            if feature_list is not ...:
                self._logger._setkv("current_total_features",
                                    len(self._feature_list) if self._feature_list else 0)
                self._logger._setkv("new_total_features", len(feature_list) if feature_list else 0)
                self._feature_list = ParameterValidator.validate_feature_list(feature_list)
            if schedule_start is not ...:
                self._logger._setkv("current_schedule_start",
                                    str(self._schedule_start) if self._schedule_start else 'N/A')
                self._logger._setkv("new_schedule_start", str(schedule_start))
                self._schedule_start = ParameterValidator.validate_datetime(schedule_start)
            if alert_config is not ...:
                self._alert_config = ParameterValidator.validate_alert_configuration(alert_config)
            if drift_threshold is not ...:
                self._drift_threshold = ParameterValidator.validate_drift_threshold(drift_threshold)
                if not self._drift_threshold:
                    self._drift_threshold = DEFAULT_DRIFT_THRESHOLD

            dto = self._update_remote(actlogger, wait_for_completion)
            self._state = dto.state

            # refresh log context and remove temp items
            self._logger._update_general_contexts(tft=len(self._feature_list) if self._feature_list else 0)
            self._logger._reset_to_general_context()

            return self

    def _update_remote(self, logger, wait_for_completion=True):
        """Update the DataDriftDetector entry in the service database.

        :param logger: Activity logger for service call
        :type logger: datetime.datetime
        :param wait_for_completion: Whether to wait for the enable/disable/delete operations to complete.
        :type wait_for_completion: bool
        :return: DataDrift data transfer object
        :rtype: azureml.datadrift._restclient.models.DataDriftDto
        """
        dto = UpdateDataDriftDto(compute_target_name=self._compute_target_name,
                                 alert_configuration=AlertConfiguration(
                                     email_addresses=self.alert_config.email_addresses)
                                 if self.alert_config else None,
                                 features=self._feature_list,
                                 drift_threshold=self._drift_threshold,
                                 state=self._state)
        client_dto = self._client.update(self._id, dto, logger)
        if wait_for_completion:
            self._wait_for_completion()
            client_dto = DataDriftDetector._get_by_id(self.workspace, self._id, False)
        DataDriftDetector._validate_client_dto(client_dto, logger)
        return client_dto

    def _update_schedule(self, logger, wait_for_completion=True):
        """Update the DataDriftDetector entry in the service database with schedule information.

        :param logger: Activity logger for service call
        :type logger: datetime.datetime
        :param wait_for_completion: Whether to wait for the enable/disable/delete operations to complete.
        :type wait_for_completion: bool
        :return: DataDrift data transfer object
        :rtype: azureml.datadrift._restclient.models.DataDriftDto
        """
        dto = UpdateDataDriftDto(compute_target_name=self._compute_target_name,
                                 features=self._feature_list,
                                 drift_threshold=self._drift_threshold,
                                 state=self._state)
        client_dto = self._client.update(self._id, dto, logger)
        if wait_for_completion:
            self._wait_for_completion()
            client_dto = DataDriftDetector._get_by_id(self.workspace, self._id, False)
        return client_dto

    def _delete_remote(self, logger, wait_for_completion=True):
        """Mark the DataDriftDetector entry deleted in the service database.

        :param logger: Activity logger for service call
        :type logger: datetime.datetime
        :param wait_for_completion: Whether to wait for the delete operation to complete.
        :type wait_for_completion: bool
        :return: DataDrift data transfer object
        :rtype: azureml.datadrift._restclient.models.DataDriftDto
        """
        client_dto = self._client.delete(self._id, logger)
        if wait_for_completion:
            self._wait_for_completion()
            client_dto = DataDriftDetector._get_by_id(self.workspace, self._id, False)
        return client_dto

    def _wait_for_completion(self, show_progress=True):
        """Wait for the DataDriftDetector enable/disable/delete operation to complete.

        This method ensures that the DataDriftDetector object has reached a terminal state (i.e. enabled, disabled,
        failed, deleted) before continuing execution of the current process.

        :param show_progress: Indicates whether to show more verbose output.
        :type show_progress: bool
        """
        console = DataDriftDetector._get_progress_logger(show_progress)
        in_progress_states = [ScheduleState.Enabling.name, ScheduleState.Disabling.name, ScheduleState.Deleting.name]
        failed_states = [ScheduleState.Failed.name, ScheduleState.EnableFailed.name, ScheduleState.DisableFailed.name,
                         ScheduleState.DeleteFailed.name]
        stable_states = [ScheduleState.Enabled.name, ScheduleState.Disabled.name, ScheduleState.Deleted.name]
        datadrift_state = DataDriftDetector._get_by_id(self.workspace, self._id, False).state
        time_run = 0
        sleep_period = 5
        console('Current state: {}'.format(datadrift_state))

        while datadrift_state in in_progress_states:
            if time_run + sleep_period > WAIT_FOR_COMPLETION_TIMEOUT:
                console('Timed out of waiting, status:%s.' % datadrift_state)
                break
            time_run += sleep_period
            time.sleep(sleep_period)
            temp_dto = DataDriftDetector._get_by_id(self.workspace, self._id, False)
            datadrift_state = temp_dto.state
            self._state = datadrift_state
            if datadrift_state in failed_states:
                console('State transitioning failed, please try again. status: {}'.format(datadrift_state))
            elif datadrift_state in stable_states:
                console('State transitioning completed, status: {}'.format(datadrift_state))
            else:
                console(".", cr=False)

    @staticmethod
    def _get_by_id(workspace, id, filter_deleted=True, client=None):
        """Retrieve a unique DataDriftDetector object for a given workspace and drift id.

        :param workspace: The workspace where the DataDriftDetector was created.
        :type workspace: azureml.core.workspace.Workspace
        :param id: The id the DataDriftDetector.
        :type id: str
        :param filter_deleted: Indicates whether to retrieve deleted monitor.
        :type filter_deleted: bool
        :param client: DataDriftDetector service client
        :type client: azureml.datadrift._restclient.DataDriftClient
        :return: DataDrift data transfer object
        :rtype: azureml.datadrift._restclient.models.DataDriftDto
        """
        workspace = ParameterValidator.validate_workspace(workspace)
        if not ParameterValidator.is_guid(id):
            raise TypeError("id is not a valid guid string.")

        log_context = _build_general_log_context(ws=workspace, type=DATADRIFT_TYPE_DATASET, log_env=LOG_ENV_SDK)
        logger = get_logger('datadriftdetector.get_by_id', log_context)
        with _TelemetryLogger.log_activity(logger, activity_name="get_by_id", custom_dimensions=log_context,
                                           telemetry_event_name=DATADRIFT_GET_BY_ID) as actlogger:
            actlogger.info("Getting DataDriftDetector object by id. Workspace: {}".format(workspace.name))
            try:
                dd_client = client if client else DataDriftClient(workspace.service_context)
                dto = dd_client.get(id=id, filter_deleted=filter_deleted, logger=logger)
            except HttpOperationError as e:
                error_msg = "Could not find DataDriftDetector with given id: {}. Workspace: {}, error message: {}"\
                    .format(id, workspace.name, e.message)
                actlogger.error(error_msg)
                raise
            return dto

    def _get_compute_target(self, compute_target_name=None, create_compute_target=False, logger=None):
        """Get compute target.

        :type compute_target_name: str
        :param create_compute_target: if or not or create a aml compute target if not existing
        :type create_compute_target: boolean
        :param compute_target_name: Optional, AzureML ComputeTarget name, creates one if none is specified
        :type compute_target_name: str
        :param logger: Optional, Datadrift logger
        :type logger: logging.LoggerAdapter
        :return: ComputeTarget
        :rtype: azureml.core.compute.ComputeTarget
        """
        # If any of the params below are None, this is a schedule run so default to the DataDriftDetector object's args
        if not compute_target_name:
            compute_target_name = self._compute_target_name
        if not logger:
            logger = self._logger

        try:
            compute_target = ComputeTarget(self.workspace, compute_target_name)
            if hasattr(compute_target, "provisioning_state"):
                prov_state = compute_target.provisioning_state
                if prov_state == COMPUTE_TARGET_SUCCEEDED:
                    return compute_target
                elif prov_state == COMPUTE_TARGET_PROVISIONING:
                    compute_target.wait_for_completion(True)
                    return compute_target
                elif prov_state == COMPUTE_TARGET_FAILED or prov_state == COMPUTE_TARGET_CANCELED:
                    # If the compute is in canceled or failed state, remove it and recreate
                    compute_target.delete()
                    return self._get_compute_target(compute_target_name, create_compute_target, logger)
            return compute_target

        except ComputeTargetException as e:
            # get the aml compute target or create if it is not available and create_compute_target is True
            if create_compute_target:
                return self._create_aml_compute(self.workspace, compute_target_name)
            else:
                error_message = "Compute target is not available. " \
                                "Use create_compute_target=True to allow creation of a new compute target."
                logger.error(error_message)
                raise ComputeTargetException(error_message) from e

    def _get_monitor_state(self, logger=None):
        """Update DataDriftMonitor state.

        :param logger: Optional, Datadrift logger
        :type logger: logging.LoggerAdapter
        :return: state of the DataDriftDetector schedule
        :rtype: str or None
        """
        try:
            client_dto = DataDriftDetector._get_by_id(self.workspace, self._id, False)
            return client_dto.state
        except HttpOperationError as e:
            error_msg = "Could get DataDriftDetector state with given id: {}. Workspace: {}, error message: {}" \
                .format(id, self.workspace.name, e.message)
            logger.error(error_msg)
            return None

    def _get_date_for_x_cycles(self, base_date, cycles, go_back):
        """To retrieve start or end date of given cycles.

        By default get_output and show will always return x cycles results if start_time or end_time is not inputted.
        Currently x is defined as 10, which means:
        If both start time and end time are not provided, then end time is today, start time is end time- 10 cycles;
        If only start time provided, then the end time should be start + 10 cycles;
        If only end time provided, then the start time should be end - 10 cycles;

        :param base_date: base date to calculate + or - x cycles.
        :param go_back: if to calculate start time, then go_back should be set to true to do minus x cycles.
        :return: calculated x cycles start or end date
        """
        base_date = base_date if base_date else datetime.utcnow().date()
        step = (0 - cycles) if go_back is True else cycles
        if self.frequency == 'Day':
            target_date = base_date + timedelta(days=step)
        elif self.frequency == 'Week':
            target_date = base_date + timedelta(weeks=step)
        elif self.frequency == 'Month':
            target_date = base_date + relativedelta(months=step)
        else:
            raise ValueError("Property frequency is invalid or not defined. Freq: {}".format(self.frequency))
        return datetime(target_date.year, target_date.month, target_date.day)

    def _get_valid_time_range(self, start_time=None, end_time=None):
        """Calculate time range for result retrieving or graph showing.

        :param start_time: start time input to get_output or show API.
        :param end_time: end time input to get_output or show API.
        :return: calculated time range
        """
        now = datetime.utcnow()
        end_time = datetime(now.year, now.month, now.day) if not start_time and not end_time else end_time

        start_time_valid = ParameterValidator.validate_datetime(start_time) if start_time \
            else self._get_date_for_x_cycles(base_date=end_time, cycles=DEFAULT_LOOKBACK_CYCLES, go_back=True)

        end_time_valid = ParameterValidator.validate_datetime(end_time) if end_time \
            else self._get_date_for_x_cycles(base_date=start_time_valid, cycles=DEFAULT_LOOKBACK_CYCLES, go_back=False)

        return start_time_valid, end_time_valid

    def show(self, start_time=None, end_time=None):
        """Show data drift trend in given time range.

        By default, this method shows the most recent 10 cycles. For example, if frequency is day, then it will be
        the most recent 10 days. If frequency is week, then it will be the most recent 10 weeks.

        :param start_time: The start of the presentation time window in UTC. The default None means to pick up
            the most recent 10th cycle's results.
        :type start_time: datetime.datetime, optional
        :param end_time: The end of the presentation data time window in UTC. The default None means the current day.
        :type end_time: datetime.datetime, optional
        :return: A dictionary of all figures. The key is service_name.
        :rtype: dict()
        """
        ParameterValidator.validate_state(self.state)
        start_time_valid, end_time_valid = self._get_valid_time_range(start_time=start_time, end_time=end_time)

        # remove extra information sepcifically for this run to avoid mis-leading in logs.
        self._logger._reset_to_general_context()
        self._logger._setkv(LOG_INPUT_STARTTIME, str(start_time))
        self._logger._setkv(LOG_INPUT_ENDTIME, str(end_time))

        with _TelemetryLogger.log_activity(self._logger, activity_name="show", custom_dimensions=self._logger.context,
                                           telemetry_event_name=DATADRIFT_SHOW) as actlogger:
            return _show(self, start_time_valid, end_time_valid, logger=actlogger)

    @staticmethod
    def _validate_client_dto(datadriftdto, logger):
        if datadriftdto and datadriftdto.alert_configuration and datadriftdto.alert_configuration.email_addresses \
                and len(datadriftdto.alert_configuration.email_addresses) > 0:
            # this means alert config is supposed to be set.
            if not datadriftdto.alert_id:
                error_msg = "Alert has not been setup. Datadriftdetector with id: {}.\n" \
                            "This may be because you do not have access to the AppInsights associated " \
                            "with this AzureML Workspace".format(datadriftdto.id)
                warnings.warn(error_msg)
                if logger:
                    logger.warning(error_msg)

    @staticmethod
    def _get_progress_logger(show_progress=True):
        import sys
        console = sys.stdout

        def log(message, cr=True):
            if show_progress:
                console.write("{}\n".format(message)) if cr else console.write("{}".format(message))

        return log
