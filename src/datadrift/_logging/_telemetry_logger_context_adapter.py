# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""A file for telemetry logger context adapter classes."""

import logging
from azureml.datadrift._logging._telemetry_logger import _TelemetryLogger
from azureml.datadrift._utils.constants import \
    LOG_TENANT_ID, LOG_SUBSCRIPTION_ID, LOG_RESGROUP, LOG_WS_ID, LOG_WS_LOCATION, \
    LOG_COMPUTE_TYPE, LOG_COMPUTE_SIZE, LOG_COMPUTE_NODES_MIN, LOG_COMPUTE_NODES_MAX, LOG_IMAGE_ID, \
    LOG_DRIFT_ID, LOG_DRIFT_TYPE, LOG_FREQUENCY, LOG_INTERVAL, LOG_SCHEDULE_STATUS, \
    LOG_THRESHOLD, LOG_LATENCY, LOG_TOTAL_FEATURES, LOG_SERVICES, LOG_TRAINING_DS_ID, \
    LOG_BASELINE_DS_ID, LOG_TARGET_DS_ID, LOG_ENV


class _TelemetryLoggerContextAdapter(logging.LoggerAdapter):
    """An adapter for loggers to keep contextual information in logging output."""

    def __init__(self, logger_name, context):
        """
        Initialize a new instance of the class.

        :param logger:
        :param context:
        """
        self._context = context
        logger = _TelemetryLogger.get_telemetry_logger(logger_name)
        super(_TelemetryLoggerContextAdapter, self).__init__(logger, None)

    @property
    def context(self):
        """Return current context info."""
        return self._context

    @property
    def manager(self):
        return self.logger.manager

    @manager.setter
    def manager(self, value):
        self.logger.manager = value

    @property
    def name(self):
        return self.logger.name

    def process(self, msg, kwargs):
        """
        Process the log message.

        :param msg: The log message.
        :type msg: str
        :param kwargs: The arguments with properties.
        :type kwargs: dict
        """
        if 'extra' not in kwargs:
            kwargs["extra"] = {}

        if self._context:
            if "properties" not in kwargs["extra"]:
                kwargs["extra"]["properties"] = {}
            kwargs["extra"]["properties"].update(self._context)
        else:
            raise AttributeError("Log context is not available.")

        return msg, kwargs

    def _log(self, level, msg, args, exc_info=None, extra=None, stack_info=False):
        """
        Low-level log implementation, proxied to allow nested logger adapters.
        """
        return self.logger._log(
            level,
            msg,
            args,
            exc_info=exc_info,
            extra=extra,
            stack_info=stack_info,
        )

    def _setkv(self, context_key, context_value):
        """Update context item

        :param context_key:
        :param context_value:
        :return:
        """
        if self._context:
            if context_value:
                self._context[context_key] = context_value
        else:
            raise AttributeError("Log context is not available.")

    def _update_general_contexts(self, compute=None, img=None,
                                 id=None, type=None, freq=None, interval=None, state=None, thredshold=None,
                                 latency=None, tft=0, mn=None, mv=None, svc=None, trdid=None, bsdid=None, tgdid=None,
                                 log_env=None):
        """Update one or multiple general log context fields.

        :param compute:
        :param img:
        :param id:
        :param name:
        :param type:
        :param freq:
        :param state:
        :param thredshold:
        :param tft:
        :param mn:
        :param mv:
        :param svc:
        :param trdid:
        :param bsdid:
        :param tgdid:
        :return:
        """
        min_nodes = compute.scale_settings.minimum_node_count if compute and compute.scale_settings else None
        max_modes = compute.scale_settings.maximum_node_count if compute and compute.scale_settings else None

        self._setkv(LOG_COMPUTE_TYPE, compute.type if compute else None)
        self._setkv(LOG_COMPUTE_SIZE, compute.vm_size if compute else None)
        self._setkv(LOG_COMPUTE_NODES_MIN, min_nodes)
        self._setkv(LOG_COMPUTE_NODES_MAX, max_modes)
        self._setkv(LOG_IMAGE_ID, img)
        self._setkv(LOG_DRIFT_ID, id)
        self._setkv(LOG_DRIFT_TYPE, type)
        self._setkv(LOG_FREQUENCY, freq)
        self._setkv(LOG_INTERVAL, interval)
        self._setkv(LOG_SCHEDULE_STATUS, state)
        self._setkv(LOG_THRESHOLD, thredshold)
        self._setkv(LOG_LATENCY, latency)
        self._setkv(LOG_TOTAL_FEATURES, tft)
        self._setkv(LOG_SERVICES, str(svc))
        self._setkv(LOG_TRAINING_DS_ID, trdid)
        self._setkv(LOG_BASELINE_DS_ID, bsdid)
        self._setkv(LOG_TARGET_DS_ID, tgdid)
        self._setkv(LOG_ENV, log_env)

    def _reset_to_general_context(self):
        general_context = [LOG_TENANT_ID, LOG_SUBSCRIPTION_ID, LOG_RESGROUP, LOG_WS_ID, LOG_WS_LOCATION,
                           LOG_COMPUTE_TYPE, LOG_COMPUTE_SIZE, LOG_COMPUTE_NODES_MIN, LOG_COMPUTE_NODES_MAX,
                           LOG_IMAGE_ID, LOG_DRIFT_ID, LOG_DRIFT_TYPE, LOG_FREQUENCY, LOG_INTERVAL,
                           LOG_SCHEDULE_STATUS, LOG_THRESHOLD, LOG_LATENCY, LOG_TOTAL_FEATURES, LOG_SERVICES,
                           LOG_TRAINING_DS_ID, LOG_BASELINE_DS_ID, LOG_TARGET_DS_ID, LOG_ENV]
        log_context = {k: v for k, v in self._context.items() if k in general_context}
        self._context = log_context


def _build_general_log_context(ws, compute=None, img=None,
                               id=None, type=None, freq=None, interval=None, state=None, thredshold=None,
                               latency=None, tft=0, mn=None, mv=None, svc=None, trdid=None, bsdid=None, tgdid=None,
                               log_env=None):
    """create/reset general log context for all SDK APIs

    :param ws:
    :param compute:
    :param img:
    :param id:
    :param name:
    :param type:
    :param freq:
    :param state:
    :param thredshold:
    :param latency:
    :param tft:
    :param mn:
    :param mv:
    :param svc:
    :param trdid:
    :param bsdid:
    :param tgdid:
    :return:
    """
    min_nodes = compute.scale_settings.minimum_node_count if compute and compute.scale_settings else None
    max_modes = compute.scale_settings.maximum_node_count if compute and compute.scale_settings else None

    log_context = {LOG_TENANT_ID: ws._auth._tenant_id if ws and ws._auth and hasattr(ws._auth, '_tenant_id') else None,
                   LOG_SUBSCRIPTION_ID: ws.subscription_id,
                   LOG_RESGROUP: ws.resource_group,
                   LOG_WS_ID: ws._workspace_id,
                   LOG_WS_LOCATION: ws.location,
                   LOG_COMPUTE_TYPE: compute.type if compute else None,
                   LOG_COMPUTE_SIZE: compute.vm_size if compute else None,
                   LOG_COMPUTE_NODES_MIN: min_nodes,
                   LOG_COMPUTE_NODES_MAX: max_modes,
                   LOG_IMAGE_ID: img,
                   LOG_DRIFT_ID: id,
                   LOG_DRIFT_TYPE: type,
                   LOG_FREQUENCY: freq,
                   LOG_INTERVAL: interval,
                   LOG_SCHEDULE_STATUS: state,
                   LOG_THRESHOLD: thredshold,
                   LOG_LATENCY: latency,
                   LOG_TOTAL_FEATURES: tft,
                   LOG_SERVICES: svc,
                   LOG_TRAINING_DS_ID: trdid,
                   LOG_BASELINE_DS_ID: bsdid,
                   LOG_TARGET_DS_ID: tgdid,
                   LOG_ENV: log_env}
    return log_context
