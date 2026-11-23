"""AWS EMR Serverless connector plugin for Flyte.

The plugin submits existing Spark and Hive scripts to an EMR Serverless
application through the `emr_serverless` connector task type.
"""

from flyteplugins.emr_serverless.connector import EMRServerlessConnector, EMRServerlessMetadata
from flyteplugins.emr_serverless.task import EMRServerless, EMRServerlessFunctionTask

__all__ = [
    "EMRServerless",
    "EMRServerlessConnector",
    "EMRServerlessFunctionTask",
    "EMRServerlessMetadata",
]
