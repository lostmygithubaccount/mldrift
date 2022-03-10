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
from azureml.datadrift._restclient.api_versions import PUBLICPREVIEW
from . import RestClient


def _get_model_serving_restclient(self, host=None, user_agent=None):
    if not hasattr(self, 'modelserving_restclient'):
        host = host if host is not None else self._get_run_history_url()
        self.modelserving_restclient = RestClient(self.get_auth(), base_url=host)
        self._add_user_agent(self.modelserving_restclient, user_agent=user_agent)

    return self.modelserving_restclient


def _get_msd_scope(self):
    return self._get_workspace_scope() + '/modelservingdataset'


ServiceContext._get_model_serving_restclient = _get_model_serving_restclient
ServiceContext._get_msd_scope = _get_msd_scope


class ModelServingClient(WorkspaceClient):
    def __init__(self, service_context, host=None, **kwargs):
        """
        Constructor of the class.
        """
        super(ModelServingClient, self).__init__(service_context, **kwargs)

    @property
    def auth(self):
        return self._service_context.get_auth()

    def get_rest_client(self, user_agent=None):
        """get service rest client"""
        return self._service_context._get_model_serving_restclient(
            host=self._override_host, user_agent=user_agent)

    def get_cluster_url(self):
        """get service url"""
        return self._host

    def get_modelserving_uri_path(self):
        return self._service_context._get_msd_scope()

    def schedule_run(self, logger=None, api_version=PUBLICPREVIEW):
        return self._execute_with_modelserving_arguments(self._client.schedule_msd_run, logger=logger,
                                                         api_version=api_version)

    def _execute_with_modelserving_arguments(self, func, *args, **kwargs):
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
