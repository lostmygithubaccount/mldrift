# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Telemetry logger helper class"""

import copy
import logging
import platform
import sys
import traceback

try:
    from azureml.telemetry import get_telemetry_log_handler
    from azureml.telemetry.activity import log_activity as _log_activity, ActivityType
    from azureml.telemetry.logging_handler import AppInsightsLoggingHandler
    from azureml.telemetry.contracts import (RequiredFields, StandardFields, ExtensionFields, Event)
    from azureml import telemetry

    telemetry_imported = True
    DEFAULT_ACTIVITY_TYPE = ActivityType.INTERNALCALL
except ImportError:
    telemetry_imported = False
    DEFAULT_ACTIVITY_TYPE = "InternalCall"

from ._telemetry_logger_context_formatter import _TelemetryLoggerContextFormatter
from ._telemetry_logger_context_filter import _TelemetryLoggerContextFilter
from azureml._base_sdk_common import __version__
from azureml.datadrift._utils.constants import WORKSPACE_ID, SUBSCRIPTION_ID, WORKSPACE_LOCATION, LOG_COMPONENT, \
    LOG_TELEMETRY_EVENT_ID, LOG_RUN_ID, LOG_PARENT_RUN_ID

# Follow the same pattern from azureml-telemetry
# The instrumentation key is needed in code to send instrumentation to service level monitoring
# This key is used for writing instrumentation only, so there is no log leaking, but cannot prevent spoofing logs
INSTRUMENTATION_KEY = '436f6faa-027f-4eb0-b08a-af3ce696ba01'
INSTRUMENTATION_KEY_TEST = 'b79fa9a1-005c-4168-854a-8e7a78897bb8'
TELEMETRY_COMPONENT_NAME = "azureml.datadrift"


class _NullContextManager(object):
    """A class for null context manager"""

    def __init__(self, dummy_resource=None):
        self.dummy_resource = dummy_resource

    def __enter__(self):
        return self.dummy_resource

    def __exit__(self, *args):
        pass


class _TelemetryLogger:
    """A class for telemetry logger"""

    @staticmethod
    def get_telemetry_logger(name, component_name=TELEMETRY_COMPONENT_NAME, stream_handler_verbosity=logging.WARNING):
        """
        gets the telemetry logger
        :param name: name of the logger
        :type name: str
        :param component_name: telemetry component name
        :type component_name: str
        :param stream_handler_verbosity: verbosity for stream handler
        :type stream_handler_verbosity: int
        :return: logger
        :rtype: logging.Logger
        """
        logger = logging.getLogger(__name__).getChild(name)
        logger.propagate = False
        logger.setLevel(logging.DEBUG)

        if telemetry_imported:
            if not _TelemetryLogger._found_handler(logger, AppInsightsLoggingHandler):
                logger.addHandler(get_telemetry_log_handler(component_name=component_name))

        if not _TelemetryLogger._found_handler(logger, logging.StreamHandler):
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setFormatter(_TelemetryLoggerContextFormatter())
            stream_handler.setLevel(stream_handler_verbosity)
            logger.addHandler(stream_handler)

        context = {'sdk_version': __version__, 'telemetry_component_name': component_name}
        logger.addFilter(_TelemetryLoggerContextFilter(context))

        return logger

    @staticmethod
    def log_activity(logger, activity_name, activity_type=DEFAULT_ACTIVITY_TYPE,
                     custom_dimensions=None, telemetry_event_name=None):
        """
        the wrapper of log_activity from azureml-telemetry
        :param logger: the logger object
        :type logger: logging.Logger
        :param activity_name: the name of the activity which should be unique per the wrapped logical code block
        :type activity_name: str
        :param activity_type: the type of the activity
        :type activity_type: str
        :param custom_dimensions: custom properties of the activity
        :type custom_dimensions: dict
        :param telemetry_event_name: name for telemetry events in required format
        :type custom_dimensions: str
        """
        if telemetry_imported:
            # integrate telemetry event logging if telemetry event name is provided.
            if telemetry_event_name:
                # log activity as telemetry event
                event_id = _TelemetryLogger.log_telemetry_event(telemetry_event_name, **custom_dimensions)
                # event id is generated by telemetry, store it in activity in case join is needed in future.
                custom_dimensions[LOG_TELEMETRY_EVENT_ID] = event_id

            # log activity
            return _log_activity(logger, activity_name, activity_type, custom_dimensions)
        else:
            return _NullContextManager(dummy_resource=logger)

    @staticmethod
    def log_exception(logger, default=None):
        message = default or "Unable to get traceback information"
        try:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            if exc_type and exc_value and exc_traceback:
                message = repr(traceback.format_exception(exc_type, exc_value, exc_traceback, limit=1))
        except Exception:
            pass

        logger.error(message)

    @staticmethod
    def log_telemetry_event(event_name, **kwargs):
        """
        :param event_name: name of the event
        :type event_name: str
        :param kwargs: a list of the key/value pairs which will be stored in event
        :type kwargs: dict
        """
        req = RequiredFields()
        std = StandardFields()
        dct = copy.deepcopy(kwargs)

        req.client_type = 'SDK'
        req.client_version = __version__
        req.component_name = LOG_COMPONENT

        std.client_os = platform.system()

        if WORKSPACE_ID in kwargs:
            req.workspace_id = kwargs[WORKSPACE_ID]
            dct.pop(WORKSPACE_ID)
        if SUBSCRIPTION_ID in kwargs:
            req.subscription_id = kwargs[SUBSCRIPTION_ID]
            dct.pop(SUBSCRIPTION_ID)
        if WORKSPACE_LOCATION in kwargs:
            std.workspace_region = kwargs[WORKSPACE_LOCATION]
            dct.pop(WORKSPACE_LOCATION)
        if LOG_PARENT_RUN_ID in kwargs:
            std.parent_run_id = kwargs[LOG_PARENT_RUN_ID]
            dct.pop(LOG_PARENT_RUN_ID)
        if LOG_RUN_ID in kwargs:
            std.run_id = kwargs[LOG_RUN_ID]
            dct.pop(LOG_RUN_ID)

        ext = ExtensionFields(**dct)
        event = Event(name=event_name, required_fields=req, standard_fields=std, extension_fields=ext)
        logger = telemetry.get_event_logger()
        logger.log_event(event)

        return event.required_fields.event_id

    @staticmethod
    def _found_handler(logger, handle_name):
        """
        checks logger for the given handler and returns the found status
        :param logger: Logger
        :param handle_name: handler name
        :return: boolean: True if found else False
        """
        for log_handler in logger.handlers:
            if isinstance(log_handler, handle_name):
                return True

        return False
