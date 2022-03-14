# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

DATADRIFT_KEY = "_datadrift"
DATADRIFT_SERVICES = "_services"
DATADRIFT_VERSION = "_version"
DATADRIFT_VERSION_VALUE = 1.0
DATADRIFT_SCHEDULES = "_schedules"
DATADRIFT_SCHEDULE_ID = "_schedule_id"
DATADRIFT_ENABLE = "_enabled"
DATADRIFT_FREQUENCY = "_frequency"
DATADRIFT_DRIFT_THRESHOLD = "_drift_threshold"
DATADRIFT_INTERVAL = "_interval"
DATADRIFT_ALERT_CONFIG = "_alert_config"
DATADRIFT_SCHEDULE_START = "_schedule_start"

DEFAULT_LOOKBACK_CYCLES = 10

METRIC_SCHEMA_VERSION_KEY = "schema_version"
METRIC_SCHEMA_VERSION_DFT = "0.1"
METRIC_SCHEMA_VERSION_CUR = "1.0"

# old key names, keep them for compatibility
METRIC_DATASHIFT_MCC_TEST = "ds_mcc_test"
METRIC_DATASHIFT_MCC_TRAIN = "ds_mcc_train"
METRIC_DATASHIFT_MCC_ALL = "ds_mcc_all"
# new key names with debug postfix
DEBUG_DATASHIFT_MCC_TEST = "mcc_test_for_debug"
DEBUG_DATASHIFT_MCC_TRAIN = "mcc_train_for_debug"
DEBUG_DATASHIFT_MCC_ALL = "mcc_all_for_debug"

METRIC_DATASHIFT_FEATURE_IMPORTANCE = "ds_feature_importance"
METRIC_STATISTICAL_DISTANCE_WASSERSTEIN = "wasserstein_distance"
METRIC_STATISTICAL_DISTANCE_ENERGY = "energy_distance"
METRIC_DATASET_METRICS = "dataset_metrics"
METRIC_COLUMN_METRICS = "column_metrics"
METRIC_TYPE = "metric_type"
METRIC_TYPE_DATASET = "dataset"
METRIC_TYPE_COLUMN = "column"
METRIC_FROM_DATASET = "from_dataset"
METRIC_FROM_DATASET_BOTH = "both"

LOG_FLUSH_LATENCY = 60
EMAIL_ADDRESSES = "email_addresses"

KEY_NAME_Drift_TYPE = "drift_type"
KEY_NAME_SERVICE = "service"
KEY_NAME_BASE_DATASET_ID = "baseline_dataset_id"
KEY_NAME_TARGET_DATASET_ID = "target_dataset_id"

COLUMN_NAME = "column_name"
SERVICE_NAME = "service_name"
SERVICE = "service"
DUMMY_SERVICE = "dummy service"
DRIFT_THRESHOLD = "drift_threshold"
DATETIME = "datetime"
HAS_DRIFT = "has_drift"
RUN_ID = "runid"
SCORING_DATE = "scoring_date"
TGT_DST_START_DATE = "start_date"
TGT_DST_END_DATE = "end_date"
PIPELINE_START_TIME = "pipeline_starttime"
WORKSPACE_ID = "workspace_id"
WORKSPACE_NAME = "workspace_name"
WORKSPACE_LOCATION = "workspace_location"
SUBSCRIPTION_ID = "subscription_id"
PIPELINE_NAME = "pipeline_name"
PIPELINE_VERSION = "pipeline_version"
ROOT_CORRELATION_ID = "root_correlation_id"
RUN_TYPE = "run_type"
DATADRIFT_ID = "datadrift_id"

RUN_TYPE_KEY = "run_type"
RUN_TYPE_ADHOC = "Adhoc"
RUN_TYPE_SCHEDULE = "Scheduler"
RUN_TYPE_BACKFILL = "BackFill"

BACKFILL_BATCH_LIMIT = 30

DATADRIFT_TYPE_MODEL = "ModelBased"
DATADRIFT_TYPE_DATASET = "DatasetBased"

PIPELINE_PARAMETER_ADHOC_RUN = "pipeline_arg_adhoc_run"
DRIFT_MAGNITUDE_TITLE = "Data drift magnitude"
DRIFT_CONTRIBUTION_TITLE = "Drift contribution by feature"
Y_LABEL_PERCENTAGE = "Percentage"

CURATION_COL_NAME_CORRELATIONID = "$CorrelationId"
CURATION_COL_NAME_TIMESTAMP = "$Timestamp"
CURATION_COL_NAME_TIMESTAMP_CURATION = "$Timestamp_Curation"
CURATION_COL_NAME_TIMESTAMP_INPUTS = "$Timestamp_Inputs"
CURATION_COL_NAME_TIMESTAMP_PREDICTIONS = "$Timestamp_Predictions"
CURATION_COL_NAME_REQUESTID = "$RequestId"
CURATION_COL_NAME_REQUESTID_INPUTS = "$RequestId_Inputs"
CURATION_COL_NAME_REQUESTID_PREDICTIONS = "$RequestId_Predictions"
CURATION_COL_NAME_FEATURES = "$Features"
CURATION_COL_NAME_PREDCTION_RESULT = "$Prediction_Result"
CURATION_COL_NAME_MODELNAME = "$ModelName"
CURATION_COL_NAME_MODELVERSION = "$ModelVersion"
CURATION_COL_NAME_SERVICENAME = "$ServiceName"
CURATION_COL_NAME_LABELDATAISAVAILABLE = "$LabelDataIsAvailable"
CURATION_COL_NAME_SIGNALDATAISAVAILABLE = "$SignalDataIsAvailable"

SCRIPT_DATABRICKS_INIT = "/databricks/init_datadrift/install.sh"
COMPUTE_TARGET_TYPE_AML = "AmlCompute"
COMPUTE_TARGET_TYPE_DATABRICKS = "Databricks"

OUTPUT_METRIC_DRIFT_COEFFICIENT = "datadrift_coefficient"
OUTPUT_METRIC_DRIFT_CONTRIBUTION = "datadrift_contribution"

FREQUENCY_DAY = "Day"
FREQUENCY_WEEK = "Week"
FREQUENCY_MONTH = "Month"

WAIT_FOR_COMPLETION_TIMEOUT = 3600

# event logger constants
DATADRIFT_CONSTRUCTOR = "DataDrift.DataDriftDetector.Constructor"
DATADRIFT_CREATE_FROM_DATASET = (
    "DataDrift.DataDriftDetector.Create_From_Dataset"
)
DATADRIFT_GET = "DataDrift.DataDriftDetector.Get"
DATADRIFT_GET_BY_NAME = "DataDrift.DataDriftDetector.Get_By_Name"
DATADRIFT_GET_BY_ID = "DataDrift.DataDriftDetector.Get_By_Id"
DATADRIFT_LIST = "DataDrift.DataDriftDetector.List"
DATADRIFT_RUN = "DataDrift.DataDriftDetector.Run"
DATADRIFT_ENABLE_SCHEDULE = "DataDrift.DataDriftDetector.Enable_Schedule"
DATADRIFT_DISABLE_SCHEDULE = "DataDrift.DataDriftDetector.Disable_Schedule"
DATADRIFT_UPDATE = "DataDrift.DataDriftDetector.Update"
DATADRIFT_DELETE = "DataDrift.DataDriftDetector.Delete"
DATADRIFT_BACKFILL = "DataDrift.DataDriftDetector.Back_Fill"
DATADRIFT_GET_OUTPUT = "DataDrift.DataDriftDetector.Get_Output"
DATADRIFT_RUN_INVOKER = "DataDrift.Run.Run_Invoker"
DATADRIFT_SHOW = "DataDrift.Script.Show"

# Misc.
DATADRIFT_IN_PROGRESS_MSG = "operation is still running"

# Compute
COMPUTE_TARGET_SUCCEEDED = "Succeeded"
COMPUTE_TARGET_FAILED = "Failed"
COMPUTE_TARGET_CANCELED = "Canceled"
COMPUTE_TARGET_PROVISIONING = "Provisioning"


# Logging fields
LOG_TENANT_ID = "tenant_id"
LOG_SUBSCRIPTION_ID = "subscription_id"
LOG_RESGROUP = "resource_group"
LOG_WS_ID = "workspace_id"
LOG_WS_LOCATION = "workspace_location"
LOG_COMPUTE_TYPE = "compute_type"
LOG_COMPUTE_SIZE = "compute_size"
LOG_COMPUTE_NODES_MIN = "compute_nodes_min"
LOG_COMPUTE_NODES_MAX = "compute_nodes_max"
LOG_IMAGE_ID = "image_id"
LOG_DRIFT_ID = "dd_id"
LOG_DRIFT_TYPE = "dd_type"
LOG_FREQUENCY = "freq"
LOG_INTERVAL = "interval"
LOG_SCHEDULE_STATUS = "scheduling"
LOG_THRESHOLD = "threshold"
LOG_LATENCY = "latency"
LOG_TOTAL_FEATURES = "total_features"
LOG_SERVICES = "services"
LOG_TRAINING_DS_ID = "train_dataset_id"
LOG_BASELINE_DS_ID = "baseline_dataset_id"
LOG_TARGET_DS_ID = "target_dataset_id"
# Extra fields
LOG_TELEMETRY_EVENT_ID = "telemetry_event_id"
LOG_ACTIVITY_ID = "activity_id"
LOG_ACTIVITY_NAME = "activity_name"
LOG_ACTIVITY_TYPE = "activity_type"
LOG_CATEGORICAL_FEATURES = "categorical_features"
LOG_NUMERICAL_FEATURES = "numerical_features"
LOG_RUN_TYPE_ = "run_type"
LOG_RUN_ID = "run_id"
LOG_PARENT_RUN_ID = "parent_run_id"
LOG_INPUT_STARTTIME = "start_time"
LOG_INPUT_ENDTIME = "end_time"
# For Telemetry logging
LOG_COMPONENT = "data-drift"
LOG_ENV = "log_env"
LOG_ENV_SDK = "sdk client"
LOG_ENV_PIPELINE = "pipeline job"

DEFAULT_DOMAIN_NAME = "https://ml.azure.com"
MOONCAKE_DOMAIN_NAME = "https://studio.ml.azure.cn"
MOONCAKE_REGION = "chinaeast2"
