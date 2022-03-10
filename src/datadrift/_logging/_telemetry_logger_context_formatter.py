# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""A file for telemetry logger context formatter classes."""

import logging


class _TelemetryLoggerContextFormatter(logging.Formatter):
    """An formatter for loggers to log context info."""

    def format(self, record):
        """Format the specified record with the extended context as text.

        :param record: log record
        :type record: logging.LogRecord
        :return: The formatted string.
        :rtype: str
        """

        msg = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s').format(record)
        if hasattr(record, 'properties'):
            context_str = ' '.join([
                '{}:{}'.format(k, record.properties[k]) for k in record.properties.keys()
            ])
            return '{msg} - {context}'.format(msg=msg, context=context_str)
        return msg
