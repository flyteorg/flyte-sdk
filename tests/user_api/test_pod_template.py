import flyte
from flyte._pod import PodTemplate


def test_pod_template_defaults():
    pt = PodTemplate()
    assert pt.pod_spec is None
    assert pt.primary_container_name == "primary"
    assert pt.labels is None
    assert pt.annotations is None


def test_pod_template_with_labels_and_annotations():
    pt = PodTemplate(
        labels={"team": "ml", "env": "prod"},
        annotations={"note": "testing"},
    )
    assert pt.labels == {"team": "ml", "env": "prod"}
    assert pt.annotations == {"note": "testing"}


def test_pod_template_custom_primary_container():
    pt = PodTemplate(primary_container_name="worker")
    assert pt.primary_container_name == "worker"


def test_pod_template_equality():
    pt1 = PodTemplate(labels={"a": "b"})
    pt2 = PodTemplate(labels={"a": "b"})
    assert pt1 == pt2


def test_pod_template_inequality():
    pt1 = PodTemplate(labels={"a": "b"})
    pt2 = PodTemplate(labels={"c": "d"})
    assert pt1 != pt2


def test_pod_template_to_k8s_pod():
    from kubernetes.client import V1Container, V1PodSpec

    pt = PodTemplate(
        pod_spec=V1PodSpec(containers=[V1Container(name="primary")]),
        labels={"team": "data"},
        annotations={"version": "v1"},
    )
    k8s_pod = pt.to_k8s_pod()
    assert k8s_pod.metadata.labels == {"team": "data"}
    assert k8s_pod.metadata.annotations == {"version": "v1"}
    assert k8s_pod.primary_container_name == "primary"


def test_pod_template_to_k8s_pod_with_empty_container():
    from kubernetes.client import V1Container, V1PodSpec

    pt = PodTemplate(pod_spec=V1PodSpec(containers=[V1Container(name="primary")]))
    k8s_pod = pt.to_k8s_pod()
    assert k8s_pod is not None


def test_pod_template_importable():
    assert flyte.PodTemplate is PodTemplate
