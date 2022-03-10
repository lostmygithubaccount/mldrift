# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for license information.
# ---------------------------------------------------------
import logging
import uuid

from azureml._base_sdk_common import _ClientSessionId
from azureml._base_sdk_common.user_agent import get_user_agent
from azureml._restclient.workspace_client import WorkspaceClient
from azureml._restclient.service_context import ServiceContext
from azureml.datadrift._restclient.models.data_drift_query_dto import DataDriftQueryDto
from azureml.datadrift._restclient.api_versions import GAPREVIEW
from . import RestClient


def _get_data_drift_restclient(self, host=None, user_agent=None):
    if not hasattr(self, 'datadrift_restclient'):
        host = host if host is not None else self._get_run_history_url()
        self.datadrift_restclient = RestClient(self.get_auth(), base_url=host)
        self._add_user_agent(self.datadrift_restclient, user_agent=user_agent)

    return self.datadrift_restclient


def _get_datadrift_scope(self):
    return self._get_workspace_scope() + '/datadrifts'


ServiceContext._get_data_drift_restclient = _get_data_drift_restclient
ServiceContext._get_datadrift_scope = _get_datadrift_scope


class DataDriftClient(WorkspaceClient):
    def __init__(self, service_context, host=None, **kwargs):
        """
        Constructor of the class.
        """
        super(DataDriftClient, self).__init__(service_context, **kwargs)

    @property
    def auth(self):
        return self._service_context.get_auth()

    def get_rest_client(self, user_agent=None):
        """get service rest client"""
        return self._service_context._get_data_drift_restclient(
            host=self._override_host, user_agent=user_agent)

    def get_cluster_url(self):
        """get service url"""
        return self._host

    def get_datadrift_uri_path(self):
        return self._service_context._get_datadrift_scope()

    def create(self, dto, logger=None, api_version=GAPREVIEW):
        return self._execute_with_datadrift_arguments(self._client.create, dto=dto, logger=logger,
                                                      api_version=api_version)

    def get(self, id, filter_deleted=True, logger=None, api_version=GAPREVIEW):
        return self._execute_with_datadrift_arguments(self._client.get, id=id, filter_deleted=filter_deleted,
                                                      logger=logger, api_version=api_version)

    def get_by_name(self, name, logger=None, api_version=GAPREVIEW):
        return self._execute_with_datadrift_arguments(self._client.get_by_name, name=name, logger=logger,
                                                      api_version=api_version)

    def list(self, model_name=None, model_version=0, services=None, type=None, base_dataset_id=None,
             target_dataset_id=None, logger=None, api_version=GAPREVIEW):
        queryDto = DataDriftQueryDto()
        if model_name:
            queryDto.model_name = model_name
        if model_version and model_version > 0:
            queryDto.model_version = model_version
        if services:
            queryDto.services = services
        if type:
            queryDto.type = type
        if base_dataset_id:
            queryDto.base_dataset_id = base_dataset_id
        if target_dataset_id:
            queryDto.target_dataset_id = target_dataset_id

        return self._execute_with_datadrift_arguments(self._client.list, query=queryDto, logger=logger,
                                                      is_paginated=True, api_version=api_version)

    def delete(self, id, logger=None):
        return self._execute_with_datadrift_arguments(self._client.delete, id=id, logger=logger)

    def update(self, id, dto, logger=None, api_version=GAPREVIEW):
        return self._execute_with_datadrift_arguments(self._client.update, id=id, dto=dto, logger=logger,
                                                      api_version=api_version)

    def run(self, id, dto, logger=None, api_version=GAPREVIEW):
        return self._execute_with_datadrift_arguments(self._client.run, id=id, dto=dto, logger=logger,
                                                      api_version=api_version)

    def schedule_run(self, id, logger=None, api_version=GAPREVIEW):
        return self._execute_with_datadrift_arguments(self._client.schedule_run, id=id, logger=logger,
                                                      api_version=api_version)

    def get_runs(self, id, logger=None):
        return self._execute_with_datadrift_arguments(self._client.get_runs, id=id, logger=logger, is_paginated=True)

    def get_metrics(self, id, start_date, end_date, service_name=None, run_id=None, logger=None):
        return self._execute_with_datadrift_arguments(self._client.get_metrics, id=id,
                                                      start_date_string=start_date.isoformat(),
                                                      end_date_string=end_date.isoformat(),
                                                      service_name=service_name, run_id=run_id, logger=logger)

    def _execute_with_datadrift_arguments(self, func, *args, **kwargs):
        activity_id = None
        request_id = str(uuid.uuid4())
        if 'logger' in kwargs and (
                isinstance(kwargs["logger"], logging.Logger) or isinstance(kwargs["logger"], logging.LoggerAdapter)):

            kwargs["logger"].info('Invoking: ' + func.__name__ + ',request_id: ' + request_id)

            if hasattr(kwargs["logger"], 'activity_info') and 'activity_id' in kwargs["logger"].activity_info:
                activity_id = kwargs["logger"].activity_info['activity_id']
        if 'logger' in kwargs:
            kwargs.pop('logger')

        common_header = {
            "User-Agent": get_user_agent(),
            "x-ms-client-session-id": _ClientSessionId,
            "x-ms-client-request-id": request_id
        }

        if activity_id:
            common_header['x-ms-request-root-id'] = activity_id
            common_header['Request-Id'] = "|{}.{}".format(activity_id, request_id)
        else:
            common_header['Request-Id'] = "|{}".format(request_id)

        kwargs["custom_headers"] = common_header

        return self._execute_with_workspace_arguments(func, *args, **kwargs)
