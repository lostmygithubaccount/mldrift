# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Contains functionality for configuring data drift alerts in Azure Machine Learning."""
from ._utils.parameter_validator import ParameterValidator


class AlertConfiguration:
    """Represents alert configuration for data drift jobs.

    The AlertConfiguration class allows for setting configurable alerts (such as email) on
    :class:`azureml.datadrift.DataDriftDetector` jobs. Alert configuration can be specified
    when using one of the create methods of the DataDriftDetector class.

    :param email_addresses: A list of email addresses to send DataDriftDetector alerts to.
    :type email_addresses: builtin.list[str]
    """

    def __init__(self, email_addresses):
        """Constructor.

        Allows for setting configurable alerts (such as email) on DataDriftDetector jobs.

        :param email_addresses: List of email addresses to send DataDriftDetector alerts to.
        :type email_addresses: builtin.list[str]
        """
        email_addresses = ParameterValidator.validate_email_addresses(email_addresses)
        self.email_addresses = email_addresses

    def __repr__(self):
        """Return the string representation of an AlertConfiguration object.

        :return: AlertConfiguration object string
        :rtype: str
        """
        return str(self.__dict__)
