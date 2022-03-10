# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import IPython


def main():
    commands = ["#!/bin/bash",
                "/databricks/python/bin/pip install numpy==1.14.0",
                "/databricks/python/bin/pip install pandas==0.23.4",
                "/databricks/python/bin/pip install -U azureml-defaults",
                "/databricks/python/bin/pip install -U azureml-datadrift"]

    # have to explicitly get dbutils to avoid flake8 check error F821
    dbutils = IPython.get_ipython().user_ns["dbutils"]
    dbutils.fs.put("/databricks/init_datadrift/install.sh", "\n".join(commands), True)


if __name__ == '__main__':
    main()
