# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Represents the DataDrift schedule state."""

from enum import Enum


class ScheduleState(Enum):
    """Represents the DataDrift schedule state."""

    Disabled = 1,
    Enabled = 2,
    Disabling = 3,
    Enabling = 4,
    Deleted = 5,
    Deleting = 6,
    Failed = 7,
    DeleteFailed = 8,
    EnableFailed = 9,
    DisableFailed = 10
