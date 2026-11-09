import asyncio
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import List

import flyte
import flyte.errors
import pytest
from flyte import Timeout
from flyte._internal.runtime.resources_serde import _get_gpu_extended_resource_entry
from flyte.extend import TaskPluginRegistry
from kubernetes.client import V1Container, V1PodSpec, V1Toleration

import flyteplugins.autoplace._task as autoplace_task
from flyteplugins.autoplace import (
    AutoPlaceEnvironment,
    AutoPlaceSettings,
    AutoPlaceTaskTemplate,
    NoCapacityError,
    PodSet,
    Reservation,
    podset_from_resources,
    register_provider,
    reservation_name,
    resolve_provider,
)
from flyteplugins.autoplace.gke import (
    CLASS_ANNOTATION,
    CONSUME_ANNOTATION,
    CapacityRequestFailed,
    GKEFlexStart,
    consume_pod_template,
    probe_env,
)
from flyteplugins.autoplace.gke import _probe as probe
from flyteplugins.autoplace.gke import _provreq as pr


def make_env(**kwargs) -> AutoPlaceEnvironment:
    defaults = {
        "name": "train",
        "image": flyte.Image.from_debian_base(),
        "resources": flyte.Resources(gpu="H100:8", cpu=(4, 8), memory="64Gi"),
        "cluster_queues": ["eu-west4", "us-central1"],
        "service_account": "autoplace",
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
    assert tmpl["metadata"]["labels"] == {pr.WARDEN_LABEL: "true", pr.MANAGED_BY_LABEL: pr.MANAGED_BY}
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
    pt = GKEFlexStart().consume(None, Reservation(queue="q", name="req"))
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
    assert env.cluster_queues == ("eu-west4", "us-central1")
    assert probe_env in env.depends_on
    assert probe_env.cache == "disable"
    # The probes run as the environment's account, set per call; the probe environment has none of its own.
    assert probe_env.service_account is None
    assert env.plugin_config == AutoPlaceSettings(
        cluster_queues=("eu-west4", "us-central1"),
        podsets=(env.podset,),
        search_deadline=timedelta(hours=1),
        service_account="autoplace",
        provider=env.provider,
    )


def test_env_accepts_a_provider_instance_with_its_own_settings():
    env = make_env(provider=GKEFlexStart(max_run=timedelta(days=2), accelerator="nvidia-h100-mega-80gb"))
    assert env.provider.max_run == timedelta(days=2)
    assert env.provider.accelerator == "nvidia-h100-mega-80gb"


def test_cloned_env_rebuilds_its_settings():
    env = make_env().clone_with("train-a100", resources=flyte.Resources(gpu="A100:1", cpu=2, memory="8Gi"))
    assert env.plugin_config.podsets == (env.podset,)
    assert env.podset.gpu_type == "A100"


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"cluster_queues": []}, "non-empty"),
        ({"cluster_queues": ["a", "a"]}, "duplicates"),
        ({"service_account": None}, "needs `service_account`"),
        ({"queue": "x"}, "use `cluster_queues`"),
        ({"plugin_config": object()}, "sets `plugin_config` itself"),
        ({"pod_template": "named"}, "named pod template"),
        ({"resources": None}, "needs `resources`"),
        ({"search_deadline": timedelta(0)}, "search_deadline must be positive"),
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

    assert TaskPluginRegistry.find(AutoPlaceSettings) is AutoPlaceTaskTemplate
    assert isinstance(train, AutoPlaceTaskTemplate) and isinstance(train2, AutoPlaceTaskTemplate)
    assert train.autoplace is env.plugin_config
    assert train.service_account == "autoplace"
    assert env.tasks["train.train"] is train
    assert train.placed is False
    assert train.queue is None

    spec = translate_task_to_wire(
        train, SerializationContext(version="v1", root_dir=Path(__file__).resolve().parents[3])
    )
    assert spec.task_template.type == "python"
    assert not spec.task_template.custom  # the settings never reach the wire


def test_override_queue_pins_and_skips_search():
    env = make_env()

    @env.task
    async def train(x: int) -> int:
        return x

    pinned = train.override(queue="us-central1")
    assert pinned.placed is True and pinned.queue == "us-central1"
    assert train.override(retries=2).placed is False


def test_override_plugin_config_changes_the_queues_searched():
    env = make_env()

    @env.task
    async def train(x: int) -> int:
        return x

    other = train.override(plugin_config=replace(train.autoplace, cluster_queues=("a", "b", "c")))
    assert isinstance(other, AutoPlaceTaskTemplate) and other.placed is False
    assert other.autoplace.cluster_queues == ("a", "b", "c")
    assert train.autoplace.cluster_queues == ("eu-west4", "us-central1")


@pytest.mark.asyncio
async def test_local_call_runs_the_function():
    env = make_env()

    @env.task
    async def train(x: int) -> int:
        return x + 1

    assert await train(1) == 2


# --- placement logic (provider-agnostic) ---------------------------------------------------------------------


class FakeProvider:
    """
    A CapacityProvider whose probes answer from a table, `queue -> (delay, None | Exception)`: None
    reserves after the delay, an exception is raised after it. `consume` reuses the GKE merge.
    """

    name = "fake"
    capacity_hold = timedelta(minutes=10)
    max_run = timedelta(hours=2)
    environments = ()

    def __init__(self, outcomes, release_error=None):
        self.outcomes = outcomes
        self.release_error = release_error
        self.reserved = []
        self.cancelled = []
        self.released = []

    def validate(self, podsets):
        pass

    async def reserve(self, queue, name, podsets, deadline, service_account):
        self.reserved.append((queue, name, deadline, service_account))
        delay, outcome = self.outcomes[queue]
        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            self.cancelled.append(queue)
            raise
        if outcome is not None:
            raise outcome
        return Reservation(queue=queue, name=name)

    async def release(self, reservation, service_account):
        if self.release_error:
            raise self.release_error
        self.released.append((reservation.queue, reservation.name, service_account))

    def consume(self, pod_template, reservation):
        return consume_pod_template(pod_template, reservation.name)


def template_with(provider, queues=("a", "b", "c"), **task_kwargs) -> AutoPlaceTaskTemplate:
    env = make_env(provider=provider, cluster_queues=list(queues), search_deadline=timedelta(minutes=5))

    @env.task(**task_kwargs)
    async def train(x: int) -> int:
        return x

    return train


def record_placements(monkeypatch, result=42, error=None):
    """Stand in for the placed submit: record the template it would have run, return or raise."""
    seen = []

    async def fake_aio(self, *args, **kwargs):
        seen.append(self)
        if error is not None:
            raise error
        return result

    monkeypatch.setattr("flyte._task.AsyncFunctionTaskTemplate.aio", fake_aio)
    return seen


def names_for(train, queues, seq=0):
    return {q: reservation_name(f"/{train.name}/{seq}", q) for q in queues}


@pytest.fixture(autouse=True)
def fresh_searches(monkeypatch):
    monkeypatch.setattr(autoplace_task, "_searches", {})


def test_registering_a_provider_makes_it_resolvable_by_name_and_wires_its_environments():
    fake_probe_env = flyte.TaskEnvironment(name="fake-probe", image=flyte.Image.from_debian_base())
    provider = FakeProvider({})
    provider.environments = (fake_probe_env,)
    register_provider("fake", lambda: provider)
    env = make_env(provider="fake")
    assert env.provider is provider
    assert fake_probe_env in env.depends_on and probe_env not in env.depends_on


def test_a_provider_whose_hold_leaves_no_time_to_schedule_is_refused():
    provider = FakeProvider({})
    provider.capacity_hold = timedelta(minutes=2)
    with pytest.raises(ValueError, match="leaves no time to schedule"):
        make_env(provider=provider)


@pytest.mark.asyncio
async def test_first_reservation_wins_and_every_other_queue_is_released(monkeypatch):
    provider = FakeProvider({"a": (0.2, None), "b": (0.05, None), "c": (0.2, None)})
    train = template_with(provider)
    seen = record_placements(monkeypatch)
    names = names_for(train, "abc")

    assert await train._autoplace_submit(1) == 42
    assert [(q, n) for q, n, _, _ in provider.reserved] == [("a", names["a"]), ("b", names["b"]), ("c", names["c"])]
    assert all(d == timedelta(minutes=5) and sa == "autoplace" for _, _, d, sa in provider.reserved)
    assert sorted(provider.cancelled) == ["a", "c"]

    (placed,) = seen
    assert placed.queue == "b" and placed.placed is True
    assert placed.pod_template.annotations[CONSUME_ANNOTATION] == names["b"]
    assert placed.timeout == Timeout(max_queued_time=timedelta(minutes=8), deadline=timedelta(hours=2))
    # The losers right after the search, the winner after the task.
    assert provider.released == [
        ("a", names["a"], "autoplace"),
        ("c", names["c"], "autoplace"),
        ("b", names["b"], "autoplace"),
    ]


@pytest.mark.asyncio
async def test_a_probe_that_says_no_does_not_end_the_search(monkeypatch):
    provider = FakeProvider(
        {
            "a": (0.01, CapacityRequestFailed("GKE gave up")),
            "b": (0.02, flyte.errors.DeadlineExceededError("deadline")),
            "c": (0.1, None),
        }
    )
    train = template_with(provider)
    seen = record_placements(monkeypatch)

    assert await train._autoplace_submit(1) == 42
    assert seen[0].queue == "c"
    assert sorted(q for q, _, _ in provider.released) == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_reservations_that_land_together_keep_one_and_release_the_rest(monkeypatch):
    provider = FakeProvider({"a": (0, None), "b": (0, None), "c": (0, None)})
    train = template_with(provider)
    seen = record_placements(monkeypatch)

    await train._autoplace_submit(1)
    winner = seen[0].queue
    assert sorted(q for q, _, _ in provider.released[:2]) == sorted(set("abc") - {winner})
    assert provider.released[2][0] == winner


@pytest.mark.asyncio
async def test_no_reservation_raises_with_every_reason_and_releases_every_queue(monkeypatch):
    provider = FakeProvider(
        {
            "a": (0.01, flyte.errors.DeadlineExceededError("deadline")),
            "b": (0.02, CapacityRequestFailed("Expired")),
            "c": (0.03, RuntimeError("forbidden")),
        }
    )
    train = template_with(provider)
    seen = record_placements(monkeypatch)

    with pytest.raises(NoCapacityError) as e:
        await train._autoplace_submit(1)
    assert seen == []
    assert "a: no capacity within 0:05:00" in str(e.value)
    assert "b: probe failed: CapacityRequestFailed('Expired')" in str(e.value)
    assert "c: probe failed: RuntimeError('forbidden')" in str(e.value)
    assert sorted(q for q, _, _ in provider.released) == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_missing_the_hold_fails_the_call_and_releases_the_winner(monkeypatch):
    provider = FakeProvider({"a": (0, None), "b": (0.5, None)})
    train = template_with(provider, queues=("a", "b"))
    seen = record_placements(monkeypatch, error=flyte.errors.MaxQueuedTimeExceededError("queued 8m"))

    with pytest.raises(NoCapacityError, match="reserved in a but the task was not scheduled within 0:08:00"):
        await train._autoplace_submit(1)
    assert len(seen) == 1  # no second search
    assert [q for q, _, _ in provider.released] == ["b", "a"]


@pytest.mark.asyncio
async def test_a_failed_release_does_not_mask_the_task_result(monkeypatch):
    provider = FakeProvider({"a": (0, None), "b": (0.5, None)}, release_error=RuntimeError("boom"))
    train = template_with(provider, queues=("a", "b"))
    record_placements(monkeypatch, result=7)
    assert await train._autoplace_submit(1) == 7


@pytest.mark.asyncio
async def test_the_winner_is_released_when_placing_it_fails(monkeypatch):
    provider = FakeProvider({"a": (0, None), "b": (0.5, None)})

    def broken_consume(pod_template, reservation):
        raise ValueError("bad template")

    provider.consume = broken_consume
    train = template_with(provider, queues=("a", "b"))
    seen = record_placements(monkeypatch)

    with pytest.raises(ValueError, match="bad template"):
        await train._autoplace_submit(1)
    assert seen == []
    assert [q for q, _, _ in provider.released] == ["b", "a"]


@pytest.mark.asyncio
async def test_a_loser_that_never_answers_does_not_hold_up_placement(monkeypatch):
    provider = FakeProvider({"a": (0, None), "b": (0.5, None)})
    answered = asyncio.Event()
    release = provider.release

    async def slow_release(reservation, service_account):
        if reservation.queue == "b":
            await answered.wait()
        await release(reservation, service_account)

    provider.release = slow_release
    train = template_with(provider, queues=("a", "b"))

    async def fake_aio(self, *args, **kwargs):
        answered.set()  # only reached if placement did not wait for the loser's release
        return 42

    monkeypatch.setattr("flyte._task.AsyncFunctionTaskTemplate.aio", fake_aio)
    assert await asyncio.wait_for(train._autoplace_submit(1), 1) == 42
    assert sorted(q for q, _, _ in provider.released) == ["a", "b"]


@pytest.mark.asyncio
async def test_a_cancelled_placement_still_releases_the_losers_and_the_winner(monkeypatch):
    provider = FakeProvider({"a": (0, None), "b": (0.5, None)})
    train = template_with(provider, queues=("a", "b"))
    record_placements(monkeypatch, error=asyncio.CancelledError())

    with pytest.raises(asyncio.CancelledError):
        await train._autoplace_submit(1)
    assert sorted(q for q, _, _ in provider.released) == ["a", "b"]


@pytest.mark.asyncio
async def test_a_named_pod_template_is_refused_before_anything_is_reserved(monkeypatch):
    provider = FakeProvider({"a": (0, None), "b": (0.5, None)})
    train = template_with(provider, queues=("a", "b")).override(pod_template="named")
    record_placements(monkeypatch)

    with pytest.raises(flyte.errors.RuntimeUserError, match="named pod template 'named'"):
        await train._autoplace_submit(1)
    assert provider.reserved == []


@pytest.mark.asyncio
async def test_each_search_in_a_parent_gets_its_own_names_and_a_retry_gets_the_same_ones(monkeypatch):
    provider = FakeProvider({"a": (0, None), "b": (0.5, None)})
    train = template_with(provider, queues=("a", "b"))
    record_placements(monkeypatch)

    await train._autoplace_submit(1)
    await train._autoplace_submit(1)
    first, second = provider.reserved[0:2], provider.reserved[2:4]
    assert {n for _, n, _, _ in first}.isdisjoint({n for _, n, _, _ in second})

    monkeypatch.setattr(autoplace_task, "_searches", {})  # a retried parent starts counting again
    await train._autoplace_submit(1)
    assert provider.reserved[4:6] == first


def test_placed_timeout_keeps_a_tighter_user_bound():
    provider = FakeProvider({})
    train = template_with(
        provider, timeout=Timeout(max_runtime=timedelta(minutes=30), max_queued_time=180, deadline=timedelta(hours=1))
    )
    assert train._placed_timeout(train.autoplace) == Timeout(
        max_runtime=timedelta(minutes=30), max_queued_time=timedelta(minutes=3), deadline=timedelta(hours=1)
    )
    loose = template_with(provider, timeout=Timeout(max_queued_time=timedelta(hours=1), deadline=timedelta(days=1)))
    assert loose._placed_timeout(loose.autoplace) == Timeout(
        max_queued_time=timedelta(minutes=8), deadline=timedelta(hours=2)
    )


@pytest.mark.asyncio
async def test_probe_io_types_round_trip_through_type_engine():
    """What crosses the wire into a probe: the pod shapes."""
    from flyte.types import TypeEngine

    podsets = [PodSet(count=2, cpu="4", memory="32Gi", gpu_count=8, gpu_type="H100", gpu_class="GPU")]
    lit = await TypeEngine.to_literal(podsets, List[PodSet], TypeEngine.to_literal_type(List[PodSet]))
    assert await TypeEngine.to_python_value(lit, List[PodSet]) == podsets


def test_reservation_names_are_fixed_by_search_and_queue():
    a = reservation_name("run-1/action-1/train/0", "eu-west4")
    assert a == reservation_name("run-1/action-1/train/0", "eu-west4")
    assert a != reservation_name("run-1/action-1/train/0", "us-central1")
    assert a != reservation_name("run-1/action-1/train/1", "eu-west4")
    assert a.startswith("autoplace-") and len(a) == len("autoplace-") + 10


# --- GKE provider routing, probe and handover -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gke_reserve_and_release_run_in_the_queue_as_the_account(monkeypatch):
    import flyteplugins.autoplace.gke as gke

    calls = {}

    def fake_override(which):
        def override(**kwargs):
            calls[f"{which}_override"] = kwargs

            class T:
                async def aio(self, *args):
                    calls[which] = args

            return T()

        return override

    monkeypatch.setattr(gke.reserve_capacity, "override", fake_override("reserve"))
    monkeypatch.setattr(gke.release_capacity, "override", fake_override("release"))
    podsets = (PodSet(count=1, cpu="1", memory="1Gi"),)
    provider = GKEFlexStart(max_run=timedelta(hours=1))

    got = await provider.reserve("q1", "req-x", podsets, timedelta(minutes=3), "sa")
    assert got == Reservation(queue="q1", name="req-x")
    assert calls["reserve_override"] == {
        "queue": "q1",
        "service_account": "sa",
        "timeout": Timeout(deadline=timedelta(minutes=3)),
    }
    assert calls["reserve"] == ("req-x", list(podsets), 3600, None)

    await provider.release(Reservation(queue="q2", name="req-y"), "sa")
    assert calls["release_override"] == {
        "queue": "q2",
        "service_account": "sa",
        "retries": gke.RELEASE_RETRIES,
        "timeout": Timeout(deadline=gke.RELEASE_DEADLINE),
    }
    assert calls["release"] == ("req-y",)


class FakeCluster:
    """Stands in for the probe's Kubernetes calls; `conditions` are returned one per poll."""

    def __init__(self, monkeypatch, conditions):
        self.conditions = list(conditions)
        self.created = []
        self.deleted = []
        self.handed_over = []
        monkeypatch.setenv("HOSTNAME", "probe-pod")
        monkeypatch.setattr(probe, "POLL_SECONDS", 0)
        monkeypatch.setattr(pr, "in_cluster_namespace", lambda: "ns")
        monkeypatch.setattr(pr, "load_clients", lambda: ("core", "custom"))
        monkeypatch.setattr(pr, "read_pod", lambda core, ns, name: {"metadata": {"name": name, "uid": "u"}})
        monkeypatch.setattr(pr, "create_pod_template", lambda core, m: self.created.append(m["metadata"]["name"]))
        monkeypatch.setattr(
            pr, "create_provisioning_request", lambda custom, m: self.created.append(m["metadata"]["name"]) or m
        )
        monkeypatch.setattr(pr, "read_conditions", lambda custom, ns, name: self.conditions.pop(0))
        monkeypatch.setattr(pr, "hand_over", lambda core, custom, req, templates: self.handed_over.append(templates))
        monkeypatch.setattr(pr, "delete_provisioning_request", lambda custom, ns, name: self.deleted.append(name))


@pytest.mark.asyncio
async def test_gke_probe_waits_for_provisioned_then_hands_the_request_over(monkeypatch):
    cluster = FakeCluster(
        monkeypatch,
        [{}, {"Provisioned": {"status": "False", "reason": "NotProvisioned"}}, {"Provisioned": {"status": "True"}}],
    )
    await probe.reserve_capacity.func("req", [PodSet(count=1, cpu="1", memory="1Gi")], 3600, None)
    assert cluster.created == ["req-0", "req"]
    assert cluster.handed_over == [["req-0"]]
    assert cluster.deleted == []


@pytest.mark.asyncio
async def test_gke_probe_raises_and_withdraws_the_request_when_gke_gives_up(monkeypatch):
    cluster = FakeCluster(monkeypatch, [{}, {"Failed": {"status": "True", "reason": "Expired"}}])
    with pytest.raises(CapacityRequestFailed, match="Expired"):
        await probe.reserve_capacity.func("req", [PodSet(count=1, cpu="1", memory="1Gi")], 3600, None)
    assert cluster.deleted == ["req"]
    assert cluster.handed_over == []


@pytest.mark.asyncio
async def test_gke_release_deletes_the_request_in_its_own_namespace(monkeypatch):
    cluster = FakeCluster(monkeypatch, [])
    await probe.release_capacity.func("req")
    assert cluster.deleted == ["req"]


def test_gke_handover_moves_ownership_off_the_probe_pod():
    request = {"metadata": {"name": "req", "namespace": "ns", "uid": "u-1", "ownerReferences": [{"name": "probe-pod"}]}}
    template_patch, request_patch = pr.handover_patches(request)
    assert template_patch == [
        {
            "op": "replace",
            "path": "/metadata/ownerReferences",
            "value": [
                {
                    "apiVersion": "autoscaling.x-k8s.io/v1",
                    "kind": "ProvisioningRequest",
                    "name": "req",
                    "uid": "u-1",
                    "blockOwnerDeletion": False,
                }
            ],
        }
    ]
    assert request_patch == [{"op": "remove", "path": "/metadata/ownerReferences"}]


class _Conflict(Exception):
    status = 409


class _Forbidden(Exception):
    status = 403


def test_gke_create_reuses_what_an_earlier_attempt_left():
    class Custom:
        def create_namespaced_custom_object(self, *args):
            raise _Conflict

        def get_namespaced_custom_object(self, group, version, namespace, plural, name):
            return {"metadata": {"name": name, "namespace": namespace, "uid": "existing"}}

    class Core:
        def __init__(self, error):
            self.error = error

        def create_namespaced_pod_template(self, namespace, manifest):
            raise self.error

    manifest = {"metadata": {"name": "req", "namespace": "ns"}}
    assert pr.create_provisioning_request(Custom(), manifest)["metadata"]["uid"] == "existing"
    pr.create_pod_template(Core(_Conflict()), manifest)  # already there: fine
    with pytest.raises(_Forbidden):
        pr.create_pod_template(Core(_Forbidden()), manifest)


def test_gke_objects_are_labelled_as_managed_by_the_plugin():
    owner = pr.owner_reference({"metadata": {"name": "probe-pod", "uid": "abc"}})
    ps = PodSet(count=1, cpu="1", memory="1Gi")
    assert (
        pr.pod_template_manifest("t", "ns", ps, None, owner)["metadata"]["labels"][pr.MANAGED_BY_LABEL] == pr.MANAGED_BY
    )
    assert pr.provisioning_request_manifest("r", "ns", [("t", 1)], 60, owner)["metadata"]["labels"] == {
        pr.MANAGED_BY_LABEL: pr.MANAGED_BY
    }


def test_gke_condition_evaluation_matches_what_gke_returns():
    from flyteplugins.autoplace.gke import evaluate_conditions

    # Observed on a GKE cluster with no flex-start pool: the request is not even accepted, with a reason.
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


@pytest.mark.asyncio
async def test_gke_probe_that_cannot_hand_over_is_not_a_win(monkeypatch):
    cluster = FakeCluster(monkeypatch, [{"Provisioned": {"status": "True"}}])

    def forbidden(*args):
        raise _Forbidden

    monkeypatch.setattr(pr, "hand_over", forbidden)
    with pytest.raises(CapacityRequestFailed, match="does the service account have `patch`"):
        await probe.reserve_capacity.func("req", [PodSet(count=1, cpu="1", memory="1Gi")], 3600, None)
    assert cluster.deleted == ["req"]


def test_gke_hand_over_of_a_request_that_has_no_owner_only_touches_the_templates():
    calls = []

    class Core:
        def patch_namespaced_pod_template(self, name, namespace, patch, _content_type):
            calls.append(name)

    class Custom:
        def patch_namespaced_custom_object(self, *args, **kwargs):
            raise AssertionError("removing an absent ownerReferences is an error")

    pr.hand_over(Core(), Custom(), {"metadata": {"name": "req", "namespace": "ns", "uid": "u-1"}}, ["req-0"])
    assert calls == ["req-0"]
