# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""A file for telemetry logger context filter classes."""

import logging


class _TelemetryLoggerContextFilter(logging.Filter):
    """An filter for loggers to keep contextual information in logging output."""

    def __init__(self, context=None):
        """Initialize the instance of the class.

        :param context: extended context of the log record, defaults to None
        :param context: dict
        """

        self._context = context

    def filter(self, record):
        """ Add extended context to the log record

        :param record: log record
        :type record: logging.LogRecord
        :return: The flags indicates if the specified record is to be logged.
        :rtype: bool
        """
        if self._context:
            if hasattr(record, 'properties'):
                record.properties.update(self._context)
            else:
                record.properties = self._context
        return True
