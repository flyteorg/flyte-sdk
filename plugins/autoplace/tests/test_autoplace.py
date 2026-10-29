import asyncio
from datetime import timedelta
from pathlib import Path
from typing import List

import flyte
import pytest
from flyte._internal.runtime.resources_serde import _get_gpu_extended_resource_entry
from flyte._timeout import Timeout
from kubernetes.client import V1Container, V1PodSpec, V1Toleration

from flyteplugins.autoplace import (
    AutoPlaceEnvironment,
    AutoPlaceSettings,
    AutoPlaceTaskTemplate,
    NoCapacityError,
    PodSet,
    Reservation,
    podset_from_resources,
    register_provider,
    resolve_provider,
)
from flyteplugins.autoplace.gke import (
    CLASS_ANNOTATION,
    CONSUME_ANNOTATION,
    GKEFlexStart,
    consume_pod_template,
    probe_env,
)
from flyteplugins.autoplace.gke import _provreq as pr


def make_env(**kwargs) -> AutoPlaceEnvironment:
    defaults = {
        "name": "train",
        "image": flyte.Image.from_debian_base(),
        "resources": flyte.Resources(gpu="H100:8", cpu=(4, 8), memory="64Gi"),
        "clusters": ["eu-west4", "us-central1"],
    }
    defaults.update(kwargs)
    return AutoPlaceEnvironment(**defaults)


# --- shape (cloud-neutral) ---------------------------------------------------------------------------


def test_podset_from_resources_keeps_sdk_gpu_name_and_takes_requests():
    ps = podset_from_resources(flyte.Resources(gpu="H100:8", cpu=(4, 8), memory=("32Gi", "64Gi")))
    assert ps == PodSet(count=1, cpu="4", memory="32Gi", gpu_count=8, gpu_type="H100", gpu_class="GPU")


def test_podset_float_cpu_is_millis_and_cpu_only_has_no_gpu():
    ps = podset_from_resources(flyte.Resources(gpu="A10G:1", cpu=0.5, memory="1Gi"), count=4)
    assert (ps.cpu, ps.count, ps.gpu_count, ps.gpu_type) == ("500m", 4, 1, "A10G")
    assert podset_from_resources(flyte.Resources(cpu=2, memory="2Gi")).gpu_type is None


# --- GKE provider ------------------------------------------------------------------------------------


@pytest.mark.parametrize("gpu", ["L4:1", "T4:1", "A100:4", "A100_80G:8", "H100:8", "RTX_PRO_6000:1"])
def test_gke_accelerator_label_matches_serialized_device(gpu):
    # The request must select the same label value the worker pod will: the device the SDK serializes.
    resources = flyte.Resources(gpu=gpu)
    serialized = _get_gpu_extended_resource_entry(resources).device
    assert pr.accelerator_label(podset_from_resources(resources)) == serialized


def test_gke_accelerator_label_override_and_unmapped():
    gpu = PodSet(count=1, cpu="1", memory="1Gi", gpu_count=8, gpu_type="H100", gpu_class="GPU")
    assert pr.accelerator_label(gpu, "nvidia-h100-mega-80gb") == "nvidia-h100-mega-80gb"
    # Not in the SDK map: the backend normalizes it from config the SDK cannot see, so no label.
    assert (
        pr.accelerator_label(PodSet(count=1, cpu="1", memory="1Gi", gpu_count=1, gpu_type="H200", gpu_class="GPU"))
        is None
    )
    assert pr.accelerator_label(PodSet(count=1, cpu="1", memory="1Gi"), "nvidia-l4") is None  # CPU-only: no label


def test_gke_provider_validates_max_run_and_gpu_class():
    with pytest.raises(ValueError, match="7 days"):
        GKEFlexStart(max_run=timedelta(days=8))
    with pytest.raises(ValueError, match="NVIDIA GPUs only"):
        GKEFlexStart().validate([PodSet(count=1, cpu="1", memory="1Gi", gpu_count=1, gpu_type="V5P", gpu_class="TPU")])
    GKEFlexStart().validate([PodSet(count=1, cpu="1", memory="1Gi")])  # CPU-only is fine


def test_gke_provider_is_the_registered_default():
    assert isinstance(resolve_provider("gke"), GKEFlexStart)
    assert GKEFlexStart.capacity_hold == timedelta(minutes=10)
    assert list(GKEFlexStart().environments) == [probe_env]
    with pytest.raises(ValueError, match="Unknown capacity provider 'aws'"):
        resolve_provider("aws")


def test_manifests_follow_gke_contract():
    owner = pr.owner_reference({"metadata": {"name": "probe-pod", "uid": "abc"}})
    ps = PodSet(count=2, cpu="4", memory="32Gi", gpu_count=8, gpu_type="H100", gpu_class="GPU")
    tmpl = pr.pod_template_manifest("req-0", "ns", ps, "nvidia-h100-80gb", owner)
    assert tmpl["metadata"]["labels"] == {pr.WARDEN_LABEL: "true"}
    assert tmpl["metadata"]["ownerReferences"] == [owner]
    spec = tmpl["template"]["spec"]
    assert spec["nodeSelector"] == {pr.FLEX_START_LABEL: "true", pr.ACCELERATOR_LABEL: "nvidia-h100-80gb"}
    assert spec["containers"][0]["resources"]["requests"] == {"cpu": "4", "memory": "32Gi", "nvidia.com/gpu": "8"}
    assert spec["tolerations"] == [{"operator": "Exists"}]

    req = pr.provisioning_request_manifest("req", "ns", [("req-0", 2)], 86400, owner)
    assert req["apiVersion"] == "autoscaling.x-k8s.io/v1"
    assert req["spec"]["provisioningClassName"] == "queued-provisioning.gke.io"
    assert req["spec"]["parameters"] == {"maxRunDurationSeconds": "86400"}
    assert req["spec"]["podSets"] == [{"count": 2, "podTemplateRef": {"name": "req-0"}}]
    assert req["metadata"]["ownerReferences"] == [owner]


def test_consume_pod_template_from_nothing():
    pt = GKEFlexStart().consume(None, Reservation(queue="q", ok=True, name="req"))
    assert pt.annotations == {CONSUME_ANNOTATION: "req", CLASS_ANNOTATION: "queued-provisioning.gke.io"}
    assert pt.pod_spec.node_selector == {pr.FLEX_START_LABEL: "true"}
    assert pt.pod_spec.containers == []  # task_serde clones in the primary container
    assert [t.key for t in pt.pod_spec.tolerations] == [pr.QUEUED_TAINT]


def test_consume_pod_template_merges_and_does_not_mutate_base():
    base = flyte.PodTemplate(
        labels={"team": "ml"},
        annotations={"keep": "me"},
        pod_spec=V1PodSpec(
            containers=[V1Container(name="primary")],
            node_selector={"disk": "ssd"},
            tolerations=[V1Toleration(key=pr.QUEUED_TAINT, operator="Exists", effect="NoSchedule")],
        ),
    )
    pt = consume_pod_template(base, "req")
    assert pt.labels == {"team": "ml"}
    assert pt.annotations["keep"] == "me" and pt.annotations[CONSUME_ANNOTATION] == "req"
    assert pt.pod_spec.node_selector == {"disk": "ssd", pr.FLEX_START_LABEL: "true"}
    assert len(pt.pod_spec.tolerations) == 1  # already tolerated, not duplicated
    assert pt.pod_spec.containers[0].name == "primary"
    assert base.annotations == {"keep": "me"}
    assert base.pod_spec.node_selector == {"disk": "ssd"}


# --- environment ----------------------------------------------------------------------------------------


def test_env_resolves_provider_derives_podset_and_depends_on_probe_env():
    env = make_env()
    assert isinstance(env.provider, GKEFlexStart)
    assert env.podset == PodSet(count=1, cpu="4", memory="64Gi", gpu_count=8, gpu_type="H100", gpu_class="GPU")
    assert env.clusters == ("eu-west4", "us-central1")
    assert env.booking_window == timedelta(minutes=8)  # GKE hold (10m) minus the 2m margin
    assert probe_env in env.depends_on
    assert probe_env.cache == "disable"
    assert probe_env.service_account == "autoplace"


def test_env_accepts_a_provider_instance_with_its_own_settings():
    env = make_env(provider=GKEFlexStart(max_run=timedelta(days=2), accelerator="nvidia-h100-mega-80gb"))
    assert env.provider.max_run == timedelta(days=2)
    assert env.provider.accelerator == "nvidia-h100-mega-80gb"


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"clusters": []}, "non-empty"),
        ({"clusters": ["a", "a"]}, "duplicates"),
        ({"queue": "x"}, "use `clusters`"),
        ({"pod_template": "named"}, "named pod template"),
        ({"resources": None}, "needs `resources`"),
        ({"booking_window": timedelta(minutes=11)}, "holds reserved capacity for 0:10:00"),
        ({"reusable": flyte.ReusePolicy(replicas=1)}, "reusable"),
        ({"provider": "aws"}, "Unknown capacity provider"),
        ({"resources": flyte.Resources(gpu=flyte.TPU(device="V5P", partition="2x2x1"))}, "NVIDIA GPUs only"),
    ],
)
def test_env_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        make_env(**kwargs)


def test_task_is_autoplace_template_and_serializes_as_plain_python_task():
    from flyte._internal.runtime.task_serde import translate_task_to_wire
    from flyte.models import SerializationContext

    env = make_env()

    @env.task
    async def train(x: int) -> int:
        return x

    @env.task(short_name="t2")
    async def train2(x: int) -> int:
        return x

    assert isinstance(train, AutoPlaceTaskTemplate) and isinstance(train2, AutoPlaceTaskTemplate)
    assert train.autoplace.clusters == ("eu-west4", "us-central1")
    assert train.autoplace.podsets == (env.podset,)
    assert train.autoplace.provider is env.provider
    assert env.tasks["train.train"] is train
    assert train.placed is False
    assert train.queue is None

    spec = translate_task_to_wire(
        train, SerializationContext(version="v1", root_dir=Path(__file__).resolve().parents[3])
    )
    assert spec.task_template.type == "python"


def test_override_queue_pins_and_skips_search():
    env = make_env()

    @env.task
    async def train(x: int) -> int:
        return x

    pinned = train.override(queue="us-central1")
    assert pinned.placed is True and pinned.queue == "us-central1"
    assert train.override(retries=2).placed is False


@pytest.mark.asyncio
async def test_local_call_runs_the_function():
    env = make_env()

    @env.task
    async def train(x: int) -> int:
        return x + 1

    assert await train(1) == 2


# --- placement logic (provider-agnostic) ---------------------------------------------------------------------


class FakeProvider:
    """A CapacityProvider whose `reserve` answers from a table; `consume` reuses the GKE merge."""

    name = "fake"
    capacity_hold = timedelta(minutes=10)
    environments = ()

    def __init__(self, outcomes):
        self.outcomes = outcomes  # queue -> (delay, Reservation | Exception)
        self.cancelled = []

    def validate(self, podsets):
        pass

    def reserve(self, queue, podsets, wait):
        async def run():
            delay, outcome = self.outcomes[queue]
            try:
                await asyncio.sleep(delay)
            except asyncio.CancelledError:
                self.cancelled.append(queue)
                raise
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        return run()

    def consume(self, pod_template, reservation):
        return consume_pod_template(pod_template, reservation.name)


def template_with(provider, **settings) -> AutoPlaceTaskTemplate:
    env = make_env()

    @env.task
    async def train(x: int) -> int:
        return x

    defaults = {
        "clusters": ("a", "b", "c"),
        "podsets": (env.podset,),
        "search_timeout": timedelta(minutes=5),
        "booking_window": timedelta(minutes=8),
        "provider": provider,
    }
    defaults.update(settings)
    train.autoplace = AutoPlaceSettings(**defaults)
    return train


def test_registering_a_provider_makes_it_resolvable_by_name_and_wires_its_environments():
    fake_probe_env = flyte.TaskEnvironment(name="fake-probe", image=flyte.Image.from_debian_base())
    provider = FakeProvider({})
    provider.environments = (fake_probe_env,)
    register_provider("fake", lambda: provider)
    env = make_env(provider="fake", booking_window=timedelta(minutes=3))
    assert env.provider is provider
    assert fake_probe_env in env.depends_on and probe_env not in env.depends_on
    assert env.booking_window == timedelta(minutes=3)


@pytest.mark.asyncio
async def test_reserve_first_ok_wins_and_cancels_the_rest():
    provider = FakeProvider(
        {
            "a": (0.05, Reservation(queue="a", ok=False, reason="ResourcePoolExhausted")),
            "b": (0.1, Reservation(queue="b", ok=True, name="req-b", namespace="ns")),
            "c": (10, Reservation(queue="c", ok=True, name="req-c", namespace="ns")),
        }
    )
    r = await template_with(provider)._reserve()
    assert r.queue == "b" and r.name == "req-b"
    assert provider.cancelled == ["c"]


@pytest.mark.asyncio
async def test_reserve_raises_with_every_reason_when_all_fail():
    provider = FakeProvider(
        {
            "a": (0.01, Reservation(queue="a", ok=False, reason="QuotaExceeded")),
            "b": (0.02, RuntimeError("boom")),
            "c": (0.03, Reservation(queue="c", ok=False, reason="")),
        }
    )
    with pytest.raises(NoCapacityError) as ei:
        await template_with(provider)._reserve()
    msg = str(ei.value)
    assert "a: QuotaExceeded" in msg and "b: probe failed: RuntimeError('boom')" in msg and "c: no capacity" in msg
    assert provider.cancelled == []


@pytest.mark.asyncio
async def test_autoplace_submit_places_with_consume_template_and_booking_timeout(monkeypatch):
    provider = FakeProvider(
        {"a": (0.01, Reservation(queue="a", ok=True, name="req-a", namespace="ns")), "b": (5, None), "c": (5, None)}
    )
    train = template_with(provider)
    train.timeout = Timeout(
        max_runtime=timedelta(hours=2), max_queued_time=timedelta(minutes=30), deadline=timedelta(hours=3)
    )

    seen = []

    async def fake_aio(self, *args, **kwargs):
        seen.append(self)
        return 42

    monkeypatch.setattr("flyte._task.AsyncFunctionTaskTemplate.aio", fake_aio)
    assert await train._autoplace_submit(7) == 42

    placed = seen[0]
    assert placed.placed is True and placed.queue == "a"
    assert placed.pod_template.annotations[CONSUME_ANNOTATION] == "req-a"
    assert placed.timeout == Timeout(
        max_runtime=timedelta(hours=2), max_queued_time=timedelta(minutes=8), deadline=timedelta(hours=3)
    )


@pytest.mark.asyncio
async def test_autoplace_submit_reprobes_on_booking_miss_then_gives_up(monkeypatch):
    provider = FakeProvider({"a": (0.01, Reservation(queue="a", ok=True, name="req-a", namespace="ns"))})
    train = template_with(provider, clusters=("a",), max_placements=2)

    async def always_late(self, *args, **kwargs):
        raise flyte.errors.MaxQueuedTimeExceededError("queued too long")

    monkeypatch.setattr("flyte._task.AsyncFunctionTaskTemplate.aio", always_late)
    with pytest.raises(NoCapacityError, match="booking window 2 time"):
        await train._autoplace_submit(7)


def test_booking_timeout_keeps_a_tighter_user_bound():
    train = template_with(FakeProvider({}))
    train.timeout = Timeout(max_queued_time=timedelta(minutes=2))
    assert train._booking_timeout().max_queued_time == timedelta(minutes=2)
    train.timeout = 0
    assert train._booking_timeout() == Timeout(max_queued_time=timedelta(minutes=8))


# --- wire types -------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_probe_io_types_round_trip_through_type_engine():
    """What crosses the wire between driver and probe: a list of podsets in, a reservation out."""
    from flyte.types import TypeEngine

    podsets = [PodSet(count=2, cpu="4", memory="32Gi", gpu_count=8, gpu_type="H100", gpu_class="GPU")]
    lit = await TypeEngine.to_literal(podsets, List[PodSet], TypeEngine.to_literal_type(List[PodSet]))
    assert await TypeEngine.to_python_value(lit, List[PodSet]) == podsets

    reservation = Reservation(queue="us-central1", ok=True, name="autoplace-abc", namespace="ns")
    lit = await TypeEngine.to_literal(reservation, Reservation, TypeEngine.to_literal_type(Reservation))
    assert await TypeEngine.to_python_value(lit, Reservation) == reservation


def test_gke_provider_runs_the_probe_as_the_configured_service_account(monkeypatch):
    import flyteplugins.autoplace.gke as gke

    captured = {}

    def fake_override(**kwargs):
        captured.update(kwargs)

        def submit(*args):
            captured["args"] = args
            return "coro"

        return submit

    monkeypatch.setattr(gke.reserve_capacity, "override", fake_override)
    podsets = [PodSet(count=1, cpu="1", memory="1Gi")]
    assert GKEFlexStart(probe_service_account="default").reserve("q", podsets, timedelta(minutes=3)) == "coro"
    assert captured["queue"] == "q" and captured["service_account"] == "default"
    assert captured["args"] == ("q", podsets, 7 * 24 * 3600, 180, None)
    GKEFlexStart().reserve("q", podsets, timedelta(minutes=3))
    assert captured["service_account"] is None  # falls back to the probe environment's own account


def test_gke_condition_evaluation_matches_what_gke_returned_on_dogfood_gcp():
    from flyteplugins.autoplace.gke import evaluate_conditions

    # Observed on dogfood-gcp (no flex-start pool): the request is not even accepted, with a reason.
    no_pool = {
        "Accepted": {
            "status": "False",
            "reason": "NoQueuedNodepoolAvailable",
            "message": "No nodepool with QueuedProvisioning enabled is available for scale up",
        }
    }
    assert evaluate_conditions(no_pool) == (
        None,
        "NoQueuedNodepoolAvailable: No nodepool with QueuedProvisioning enabled is available for scale up",
    )
    assert evaluate_conditions({}) == (None, "")
    assert evaluate_conditions({"Accepted": {"status": "True"}}) == (None, "")
    queued = {"Accepted": {"status": "True"}, "Provisioned": {"status": "False", "reason": "NotProvisioned"}}
    assert evaluate_conditions(queued) == (None, "NotProvisioned")
    assert evaluate_conditions({"Provisioned": {"status": "True"}}) == (True, "")
    assert evaluate_conditions({"Failed": {"status": "True", "reason": "Expired"}}) == (False, "Expired")
    assert evaluate_conditions({"Failed": {"status": "True"}}) == (False, "Failed")
