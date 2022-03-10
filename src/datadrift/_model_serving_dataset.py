# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""Model Serving Dataset."""
import os
from ._utils.parameter_validator import ParameterValidator
from azureml.core import Dataset
from azureml.datadrift._utils.constants import CURATION_COL_NAME_TIMESTAMP_INPUTS

EXPORT_FOLDER = "export"
EXPORT_FILENAME = "msd.csv"
EXPORT_PATH = os.path.join(EXPORT_FOLDER, EXPORT_FILENAME)

DATASET_PREFIX = "ModelServingDataset-"

TIMESTAMP_COLUMN = "$Timestamp_Inputs"


class ModelServingDataset:
    """Represents a dataset used internally when a model-based DataDriftDetector object is created.

    A model-based DataDriftDetector enables you to calculate data drift between a model's training dataset and its
    scoring dataset. To create a model-based DataDriftDetector, use the
    :meth:`azureml.datadrift.DataDriftDetector.create_from_model` method.

    :param workspace: The workspace of the model-serving dataset.
    :type workspace: azureml.core.Workspace
    """

    def __init__(self, workspace):
        """Constructor.

        :param workspace: The workspace of the model serving dataset.
        :type workspace: Workspace
        """
        self.workspace = workspace

    def __repr__(self):
        """Return the string representation of a ModelServingDataset object.

        :return: ModelServingDataset object string
        :rtype: str
        """
        return str(self.__dict__)

    def export_to_csv(self, start_time, end_time):
        """Export a model-serving dataset to a local CSV file.

        :param start_time: The start time in UTC at which to start the export.
        :type start_time: datetime.datetime
        :param end_time: The end time in UTC at which to end the export.
        :type end_time: datetime.datetime
        :return: The relative path of the exported CSV file.
        :rtype: str
        """
        start_time = ParameterValidator.validate_datetime(start_time)
        end_time = ParameterValidator.validate_datetime(end_time)

        dataset = Dataset.get_by_name(self.workspace, DATASET_PREFIX + self.workspace._workspace_id.split('-')[0])
        ds = dataset.with_timestamp_columns(fine_grain_timestamp=CURATION_COL_NAME_TIMESTAMP_INPUTS)
        ds = ds.time_between(start_time=start_time, end_time=end_time)
        df = ds.to_pandas_dataframe()
        os.makedirs(EXPORT_FOLDER, exist_ok=True)
        df.to_csv(EXPORT_PATH, index=False)

        return EXPORT_PATH
