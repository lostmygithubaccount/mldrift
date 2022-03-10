# coding=utf-8
# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
#
# Code generated by Microsoft (R) AutoRest Code Generator 1.0.0.0
# Changes may cause incorrect behavior and will be lost if the code is
# regenerated.
# --------------------------------------------------------------------------

from msrest.serialization import Model


class PaginatedResultDataDriftDto(Model):
    """PaginatedResultDataDriftDto.

    :param value:
    :type value: list[~_restclient.models.DataDriftDto]
    :param continuation_token:
    :type continuation_token: str
    :param next_link:
    :type next_link: str
    """

    _attribute_map = {
        'value': {'key': 'value', 'type': '[DataDriftDto]'},
        'continuation_token': {'key': 'continuationToken', 'type': 'str'},
        'next_link': {'key': 'nextLink', 'type': 'str'},
    }

    def __init__(self, *, value=None, continuation_token: str=None, next_link: str=None, **kwargs) -> None:
        super(PaginatedResultDataDriftDto, self).__init__(**kwargs)
        self.value = value
        self.continuation_token = continuation_token
        self.next_link = next_link
