import pytest

from flyteplugins.sagemaker import (
    create_sagemaker_deployment,
    delete_sagemaker_deployment,
)


def test_sagemaker_deployment_workflow():
    deployment = create_sagemaker_deployment(
        "demo",
        {"PrimaryContainer": {"Image": "{images.primary}"}},
        {"ProductionVariants": [{}]},
        {},
        images={"primary": "repo/image"},
        region="us-west-2",
    )
    model = deployment.model_task.custom_config(None)
    endpoint_config = deployment.endpoint_config_task.custom_config(None)
    endpoint = deployment.endpoint_task.custom_config(None)
    assert model["config"]["ModelName"] == "demo-{idempotence_token}"
    assert endpoint_config["config"]["ProductionVariants"][0]["ModelName"] == "demo-{inputs.idempotence_token}"
    assert endpoint["config"]["EndpointConfigName"] == "demo-{inputs.idempotence_token}"


def test_sagemaker_deployment_workflow_with_region_at_runtime():
    deployment = create_sagemaker_deployment("demo", {}, {}, {}, region_at_runtime=True)
    assert "region" in deployment.model_task.interface.inputs
    assert "region" in deployment.endpoint_config_task.interface.inputs
    assert "region" in deployment.endpoint_task.interface.inputs


@pytest.mark.parametrize("factory", [create_sagemaker_deployment, delete_sagemaker_deployment])
def test_deployment_requires_region(factory):
    with pytest.raises(ValueError, match="Region parameter is required"):
        if factory is create_sagemaker_deployment:
            factory("demo", {}, {}, {})
        else:
            factory("demo")


def test_sagemaker_deployment_deletion_workflow():
    deletion = delete_sagemaker_deployment("demo", region="us-west-2")
    assert deletion.endpoint_task.custom_config(None)["method"] == "delete_endpoint"
    assert deletion.endpoint_config_task.custom_config(None)["method"] == "delete_endpoint_config"
    assert deletion.model_task.custom_config(None)["method"] == "delete_model"


def test_deployment_does_not_mutate_configs():
    model = {}
    endpoint_config = {"ProductionVariants": [{}]}
    endpoint = {}
    create_sagemaker_deployment(
        "demo",
        model,
        endpoint_config,
        endpoint,
        region="us-west-2",
    )
    assert model == {}
    assert endpoint_config == {"ProductionVariants": [{}]}
    assert endpoint == {}
