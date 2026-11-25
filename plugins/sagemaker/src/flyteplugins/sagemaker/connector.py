"""Backward-compatible connector imports.

New code may import from the package root. This module remains available for
the initial Flyte 2 draft's import paths.
"""

from flyteplugins.sagemaker.connectors import (
    BotoCallMetadata,
    BotoConnector,
    PythonicJobMetadata,
    PythonicSageMakerJobConnector,
    SageMakerEndpointConnector,
    SageMakerEndpointMetadata,
    SageMakerHyperParameterTuningJobConnector,
    SageMakerHyperParameterTuningJobMetadata,
    SageMakerInferenceRecommenderJobConnector,
    SageMakerInferenceRecommenderJobMetadata,
    SageMakerJobMetadata,
    SageMakerProcessingConnector,
    SageMakerProcessingJobConnector,
    SageMakerProcessingJobMetadata,
    SageMakerProcessingTaskConnector,
    SageMakerTrainingConnector,
    SageMakerTrainingJobConnector,
    SageMakerTrainingJobMetadata,
    SageMakerTrainingTaskConnector,
    SageMakerTransformConnector,
    SageMakerTransformJobConnector,
    SageMakerTransformJobMetadata,
)

__all__ = [
    "BotoCallMetadata",
    "BotoConnector",
    "PythonicJobMetadata",
    "PythonicSageMakerJobConnector",
    "SageMakerEndpointConnector",
    "SageMakerEndpointMetadata",
    "SageMakerHyperParameterTuningJobConnector",
    "SageMakerHyperParameterTuningJobMetadata",
    "SageMakerInferenceRecommenderJobConnector",
    "SageMakerInferenceRecommenderJobMetadata",
    "SageMakerJobMetadata",
    "SageMakerProcessingConnector",
    "SageMakerProcessingJobConnector",
    "SageMakerProcessingJobMetadata",
    "SageMakerProcessingTaskConnector",
    "SageMakerTrainingConnector",
    "SageMakerTrainingJobConnector",
    "SageMakerTrainingJobMetadata",
    "SageMakerTrainingTaskConnector",
    "SageMakerTransformConnector",
    "SageMakerTransformJobConnector",
    "SageMakerTransformJobMetadata",
]
