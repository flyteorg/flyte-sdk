import asyncio
import time
from datetime import datetime, timedelta, timezone
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
    Selection,
    podset_from_resources,
    queue_budget,
    race,
    register_provider,
    resolve_provider,
)
from flyteplugins.autoplace.gke import (
    CLASS_ANNOTATION,
    CONSUME_ANNOTATION,
    GKEFlexStart,
    consume_pod_template,
    probe_env,
    select_env,
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
    assert list(GKEFlexStart().environments) == [probe_env, select_env]
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
    assert probe_env in env.depends_on and select_env in env.depends_on
    assert probe_env.cache == "disable" and select_env.cache == "disable"
    assert probe_env.service_account == "autoplace"
    assert select_env.service_account is None  # the search only submits probes; it needs no cluster rights


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
    """
    A CapacityProvider whose probes answer from a table, raced with the real `race`; `consume` reuses
    the GKE merge. A table entry is (delay, Reservation | Exception), or a list of them used one per
    search (the last one repeats).
    """

    name = "fake"
    capacity_hold = timedelta(minutes=10)
    environments = ()

    def __init__(self, outcomes, release_error=None):
        self.outcomes = outcomes
        self.release_error = release_error
        self.cancelled = []
        self.released = []
        self.searches = 0

    def validate(self, podsets):
        pass

    def _probe(self, queue):
        entry = self.outcomes[queue]
        delay, outcome = (entry.pop(0) if len(entry) > 1 else entry[0]) if isinstance(entry, list) else entry

        async def run():
            try:
                await asyncio.sleep(delay)
            except asyncio.CancelledError:
                self.cancelled.append(queue)
                raise
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        return run

    def select(self, queues, podsets, wait):
        # Names each probe's reservation up front, as the GKE provider does, so cancelled probes come back
        # in `unresolved`.
        self.searches += 1
        return race({q: self._probe(q) for q in queues}, handles={q: f"req-{q}" for q in queues})

    def release(self, reservation):
        async def run():
            if self.release_error:
                raise self.release_error
            self.released.append(reservation.name)

        return run()

    def consume(self, pod_template, reservation):
        return consume_pod_template(pod_template, reservation.name)


def won(queue, name, provisioned_at=None):
    return Reservation(
        queue=queue,
        ok=True,
        name=name,
        namespace="ns",
        provisioned_at=time.time() if provisioned_at is None else provisioned_at,
    )


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
async def test_race_first_ok_wins_and_cancels_the_rest():
    provider = FakeProvider(
        {
            "a": (0.05, Reservation(queue="a", ok=False, reason="ResourcePoolExhausted")),
            "b": (0.1, won("b", "req-b")),
            "c": (10, won("c", "req-c")),
        }
    )
    sel = await provider.select(["a", "b", "c"], [], timedelta(minutes=1))
    assert sel.winner.name == "req-b" and sel.extras == []
    assert provider.cancelled == ["c"]
    assert sel.misses == {"a": "ResourcePoolExhausted", "c": "cancelled: another cluster won"}
    # c's outcome died with the cancel: it may have won, so it is handed back for release by name.
    assert [(r.queue, r.name, r.ok) for r in sel.unresolved] == [("c", "req-c", False)]


@pytest.mark.asyncio
async def test_race_collects_simultaneous_winners_as_extras():
    provider = FakeProvider({"b": (0.01, won("b", "req-b")), "a": (0.01, won("a", "req-a"))})
    sel = await provider.select(["b", "a"], [], timedelta(minutes=1))
    # Completed together: the winner does not depend on set iteration order.
    assert sel.winner.name == "req-a"
    assert [r.name for r in sel.extras] == ["req-b"]


@pytest.mark.asyncio
async def test_race_keeps_a_win_that_lands_while_the_rest_are_cancelled():
    async def stubborn():
        # Already past the point of no return when the cancel arrives: it still reports its win.
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            return won("b", "req-b")

    async def quick():
        await asyncio.sleep(0.01)
        return won("a", "req-a")

    sel = await race({"a": quick, "b": stubborn})
    assert sel.winner.name == "req-a"
    assert [r.name for r in sel.extras] == ["req-b"]


@pytest.mark.asyncio
async def test_race_without_handles_cannot_name_unresolved_probes():
    async def slow():
        await asyncio.sleep(10)
        return won("b", "req-b")

    async def quick():
        await asyncio.sleep(0.01)
        return won("a", "req-a")

    sel = await race({"a": quick, "b": slow})
    assert sel.winner.name == "req-a"
    assert sel.unresolved == [] and sel.misses == {"b": "cancelled: another cluster won"}


@pytest.mark.asyncio
async def test_race_reports_every_reason_when_all_fail():
    provider = FakeProvider(
        {
            "a": (0.01, Reservation(queue="a", ok=False, reason="QuotaExceeded")),
            "b": (0.02, RuntimeError("boom")),
            "c": (0.03, Reservation(queue="c", ok=False, reason="")),
        }
    )
    sel = await provider.select(["a", "b", "c"], [], timedelta(minutes=1))
    assert sel.winner is None and sel.extras == []
    assert sel.misses == {"a": "QuotaExceeded", "b": "probe failed: RuntimeError('boom')", "c": "no capacity"}


@pytest.mark.asyncio
async def test_submit_places_on_the_winner_and_releases_extras_and_winner(monkeypatch):
    provider = FakeProvider({"a": (0.01, won("a", "req-a")), "b": (0.01, won("b", "req-b")), "c": (5, None)})
    train = template_with(provider)
    train.timeout = Timeout(
        max_runtime=timedelta(hours=2), max_queued_time=timedelta(minutes=30), deadline=timedelta(hours=3)
    )
    seen = record_placements(monkeypatch)

    assert await train._autoplace_submit(7) == 42
    placed = seen[0]
    assert placed.placed is True and placed.queue == "a"
    assert placed.pod_template.annotations[CONSUME_ANNOTATION] == "req-a"
    assert placed.timeout == Timeout(
        max_runtime=timedelta(hours=2), max_queued_time=timedelta(minutes=8), deadline=timedelta(hours=3)
    )
    # The winner, the extra that won alongside it, and c, cancelled before it could say whether it won.
    assert sorted(provider.released) == ["req-a", "req-b", "req-c"]


@pytest.mark.asyncio
async def test_submit_caps_queue_time_to_what_is_left_of_the_hold(monkeypatch):
    # Provisioned 7 minutes ago: 10m hold - 7m - 30s safety leaves 2m30s, under the 8m booking window.
    provider = FakeProvider({"a": (0.01, won("a", "req-a", provisioned_at=time.time() - 7 * 60))})
    train = template_with(provider, clusters=("a",))
    seen = record_placements(monkeypatch)

    await train._autoplace_submit(7)
    queued = seen[0].timeout.max_queued_time
    assert timedelta(minutes=2, seconds=25) <= queued <= timedelta(minutes=2, seconds=30)


@pytest.mark.asyncio
async def test_submit_releases_a_stale_winner_and_searches_again(monkeypatch):
    provider = FakeProvider(
        {"a": [(0.01, won("a", "req-stale", provisioned_at=time.time() - 20 * 60)), (0.01, won("a", "req-fresh"))]}
    )
    train = template_with(provider, clusters=("a",))
    seen = record_placements(monkeypatch)

    assert await train._autoplace_submit(7) == 42
    assert provider.searches == 2
    assert [p.pod_template.annotations[CONSUME_ANNOTATION] for p in seen] == ["req-fresh"]
    assert provider.released == ["req-stale", "req-fresh"]


@pytest.mark.asyncio
async def test_submit_reprobes_on_booking_miss_then_gives_up(monkeypatch):
    provider = FakeProvider({"a": (0.01, won("a", "req-a"))})
    train = template_with(provider, clusters=("a",), max_placements=2)
    record_placements(monkeypatch, error=flyte.errors.MaxQueuedTimeExceededError("queued too long"))

    with pytest.raises(NoCapacityError, match="booking window 2 time"):
        await train._autoplace_submit(7)
    assert provider.searches == 2
    assert provider.released == ["req-a", "req-a"]


@pytest.mark.asyncio
async def test_submit_without_a_winner_raises_with_every_reason(monkeypatch):
    provider = FakeProvider({"a": (0.01, Reservation(queue="a", ok=False, reason="QuotaExceeded"))})
    train = template_with(provider, clusters=("a",))
    seen = record_placements(monkeypatch)

    with pytest.raises(NoCapacityError, match="a: QuotaExceeded"):
        await train._autoplace_submit(7)
    assert seen == [] and provider.released == []


@pytest.mark.asyncio
async def test_a_failed_release_does_not_mask_the_task_result(monkeypatch):
    provider = FakeProvider({"a": (0.01, won("a", "req-a"))}, release_error=RuntimeError("cluster unreachable"))
    train = template_with(provider, clusters=("a",))
    record_placements(monkeypatch)

    assert await train._autoplace_submit(7) == 42


def test_queue_budget_is_the_booking_window_cut_to_the_remaining_hold():
    hold, window, now = timedelta(minutes=10), timedelta(minutes=8), 1_000_000.0
    assert queue_budget(0.0, now, hold, window) == window  # unknown: treated as just provisioned
    assert queue_budget(now, now, hold, window) == window
    assert queue_budget(now + 30, now, hold, window) == window  # cloud clock ahead of ours
    assert queue_budget(now - 8 * 60, now, hold, window) == timedelta(minutes=1, seconds=30)
    assert queue_budget(now - 9 * 60, now, hold, window) == timedelta(0)  # 30s left: not worth placing on


def test_booking_timeout_keeps_a_tighter_user_bound():
    train = template_with(FakeProvider({}))
    train.timeout = Timeout(max_queued_time=timedelta(minutes=2))
    assert train._booking_timeout(timedelta(minutes=8)).max_queued_time == timedelta(minutes=2)
    train.timeout = 0
    assert train._booking_timeout(timedelta(minutes=8)) == Timeout(max_queued_time=timedelta(minutes=8))


# --- wire types -------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_probe_io_types_round_trip_through_type_engine():
    """What crosses the wire: podsets into the search, a reservation out of a probe, a selection out of the search."""
    from flyte.types import TypeEngine

    podsets = [PodSet(count=2, cpu="4", memory="32Gi", gpu_count=8, gpu_type="H100", gpu_class="GPU")]
    lit = await TypeEngine.to_literal(podsets, List[PodSet], TypeEngine.to_literal_type(List[PodSet]))
    assert await TypeEngine.to_python_value(lit, List[PodSet]) == podsets

    reservation = Reservation(queue="us-central1", ok=True, name="autoplace-abc", namespace="ns", provisioned_at=1.5)
    lit = await TypeEngine.to_literal(reservation, Reservation, TypeEngine.to_literal_type(Reservation))
    assert await TypeEngine.to_python_value(lit, Reservation) == reservation

    selection = Selection(winner=reservation, extras=[won("b", "req-b", 2.0)], misses={"c": "no capacity"})
    lit = await TypeEngine.to_literal(selection, Selection, TypeEngine.to_literal_type(Selection))
    assert await TypeEngine.to_python_value(lit, Selection) == selection


# --- GKE provider routing and handover --------------------------------------------------------------------------


def test_gke_select_passes_the_probe_account_and_release_targets_the_reservation_cluster(monkeypatch):
    import flyteplugins.autoplace.gke as gke

    calls = {}
    monkeypatch.setattr(gke, "select_capacity", lambda *args: calls.setdefault("select", args) and "search")

    def fake_override(**kwargs):
        calls["release_override"] = kwargs
        return lambda *args: calls.setdefault("release", args) and "release"

    monkeypatch.setattr(gke.release_capacity, "override", fake_override)
    podsets = [PodSet(count=1, cpu="1", memory="1Gi")]
    provider = GKEFlexStart(probe_service_account="default")

    assert provider.select(("q1", "q2"), podsets, timedelta(minutes=3)) == "search"
    assert calls["select"] == (["q1", "q2"], podsets, 7 * 24 * 3600, 180, None, "default")
    assert provider.release(won("q2", "req-x")) == "release"
    assert calls["release_override"] == {"queue": "q2", "service_account": "default"}
    assert calls["release"] == ("req-x", "ns")


def test_gke_handover_moves_ownership_off_the_probe_pod():
    request = {"metadata": {"name": "req", "namespace": "ns", "uid": "u-1"}}
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


def test_gke_provisioned_at_reads_the_provisioned_transition_time():
    cond = {"Provisioned": {"status": "True", "lastTransitionTime": "2026-10-29T22:11:51Z"}}
    assert pr.provisioned_at(cond) == datetime(2026, 10, 29, 22, 11, 51, tzinfo=timezone.utc).timestamp()
    assert pr.provisioned_at({"Provisioned": {"status": "False", "lastTransitionTime": "2026-10-29T22:11:51Z"}}) == 0.0
    assert pr.provisioned_at({}) == 0.0
    assert pr.provisioned_at({"Provisioned": {"status": "True", "lastTransitionTime": "not a time"}}) == 0.0


def test_gke_request_names_are_fixed_by_search_and_queue():
    a = pr.request_name("run-1/action-1", "eu-west4")
    assert a == pr.request_name("run-1/action-1", "eu-west4")
    assert a != pr.request_name("run-1/action-1", "us-central1")
    assert a != pr.request_name("run-1/action-2", "eu-west4")
    assert a.startswith("autoplace-") and len(a) == len("autoplace-") + 10


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
