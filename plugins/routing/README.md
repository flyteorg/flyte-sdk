# Routing Plugin

`flyteplugins-routing` routes Flyte runs across several control planes from the client, using the
config file's `profiles:` section to describe the targets.

It exists for teams running multiple clusters for data separation or across clouds, who want the
placement decision made locally — from what the code and its inputs say — rather than shipped to a
control plane that then re-routes.

The SDK carries no routing code at all. This plugin is built entirely on two things the SDK
provides for their own sake — named profiles (`--profile`) and `flyte.use_profile` — plus the
existing CLI plugin-hook mechanism. Which cluster, how runs are named, how they are found again:
all of it is ordinary plugin code you can fork.

## Installation

```bash
pip install flyteplugins-routing
```

Installing it is all the wiring there is; the entry points register themselves.

## Setup

Describe each cluster as a profile. Anything a profile does not set is inherited from the file's
top-level sections, so shared settings stay written once:

```yaml
admin:
  endpoint: dns:///default.example.com
task:
  project: research
  domain: development

profiles:
  us-east:
    admin:
      endpoint: dns:///us-east.example.com
  eu-west:
    admin:
      endpoint: dns:///eu-west.example.com
  gpu-pool:
    admin:
      endpoint: dns:///gpu.example.com
```

Or add them one at a time, which merges into the file rather than replacing it:

```bash
flyte create config --endpoint dns:///us-east.example.com --profile us-east
flyte get profiles
```

## Usage

From the CLI, nothing changes at the call site — runs are routed as they are submitted:

```bash
flyte run train.py train --dataset s3://bucket/x
```

From Python, swap the module name — the signatures match `flyte`'s exactly, so it is a mechanical
change:

```python
from flyteplugins import routing

routing.run(train, "s3://bucket/x")  # was flyte.run(...)
routing.with_runcontext(version="v2").run(train, x=1)  # was flyte.with_runcontext(...)
```

Positional and keyword task arguments both work, and `with_runcontext` takes every option
`flyte.with_runcontext` does. A name or labels you pass yourself survive routing.

**`flyte.run(...)` is not routed.** The plugin hooks the CLI, and a call from a script, a notebook
or an orchestrator never goes through the CLI. Those submissions go to the default profile. This
is the trade for the SDK carrying no routing seam — see *Two paths* below.

Every command that takes a run name finds it, from any machine, with no local state — whether it
reads the run or acts on it:

```bash
flyte get run 4h-9c1e02af
flyte get logs 4h-9c1e02af
flyte rerun 4h-9c1e02af --epochs 5
flyte rerun 4h-9c1e02af --recover
flyte abort run 4h-9c1e02af
```

To pin a single command to one cluster and bypass routing, name the profile explicitly:

```bash
flyte --profile eu-west run train.py train --dataset s3://bucket/x
```

## How it decides

The bundled router hashes the task name, project and domain onto the profile set using rendezvous
("highest random weight") hashing.

Every run of a task in a given project and domain lands on the same cluster. That is the point:
repeat work returns to the cluster that already holds that task's cached outputs and its data, so
the policy provides locality rather than just spreading load. Rendezvous is used rather than
`hash(key) % len(profiles)` because modulo reshuffles nearly every key when a profile is added or
removed, throwing that locality away on an unrelated config edit.

Placement does not depend on a run's arguments. `RoutingContext` carries no input values (see
below), and per-argument placement is arguably the wrong granularity anyway — a task's datasets
usually want to sit together rather than be scattered by argument value.

Every routed run is labelled with the decision, so you can ask afterwards:

```bash
flyte get run --with-label routed-to=gpu-pool
```

## Two paths

| | Routed? | How |
|---|---|---|
| `flyte run ...` | yes, transparently | CLI hook on `run` |
| `flyteplugins.routing.run(...)` | yes | same policy, called directly |
| `flyte.run(...)` | **no** | goes to the default profile |

The split is deliberate and worth naming rather than discovering. Routing where the CLI is
involved costs the SDK nothing; routing *everywhere* would need a hook inside `flyte.run` itself,
which is core surface this deliberately does not ask for. If a scheduler submits your production
runs, route them through `flyteplugins.routing.run` or pin them with `--profile` — do not assume
they inherit the CLI's behaviour.

## How a run is found again

Every run-addressed command is given a name and nothing else, so a run on a non-ambient profile
would otherwise be invisible to it. Hooks cover `get run`, `get logs`, `get action`, `get io`,
`get condition`, `abort run`, `abort action`, `signal condition` and `rerun`.

This is *resolution*, and it is a different job from routing. The policy picks where a new run
goes; resolution finds where an existing run already is. `rerun` never consults the policy — its
target is fixed by where the source run lives — but it very much needs resolving, and so do the
commands that write.

`rerun` also needs its profile settled earlier than the rest. It reads the source run's interface
while Click is still *parsing*, so it can turn `--some-input v` into a typed option; an
invoke-time hook would be too late. Its hook wraps `parse_args` instead.

Resolution is pure string work: routed runs are named `<tag>-<random>`, where the tag is derived
from the profile, so decoding needs no network and no stored state.

**There is no search.** A name this plugin did not mint — one the control plane generated, one you
chose, one from before the plugin was installed — resolves to nothing, and the command runs
against the default profile: the config file's top-level sections, or whatever `--profile`
selected. Probing every cluster to locate a run would cost a round trip per profile on every
lookup, and would make one unreachable cluster or one expired credential everybody's problem on
every command. A stale tag and a colliding tag are treated the same way, for the same reason:
falling back to the default beats picking a cluster at random and reporting it confidently.

Two constraints shape the naming, both from the control plane:

- Run names are capped at 30 characters.
- Names beginning with `u`, `r` or `l` are reserved — the platform classifies runs by the first
  character of the name.

The tag alphabet therefore excludes those three letters. That is affordable only because the tag
is derived from a hash rather than spelled from the profile's own name: `us-east` cannot be used
as a prefix, but its tag can be any character.

The run name's suffix is random, not derived from the routing key. Two runs of the same task on
the same inputs route to the same profile and would collide on a name otherwise — the second
submission coming back as `RunAlreadyExistsError`. Placement is deterministic; names are not.

## Writing your own policy

`route()` is a plain function over a plain dataclass — fork this package and change it, or import
the pieces and build your own hook:

```toml
[project.entry-points."flyte.plugins.cli.hooks"]
run = "my_package.routing:route_run"
```

```python
from flyteplugins.routing import RoutingContext, RoutingDecision


def route(ctx: RoutingContext) -> RoutingDecision | None:
    if ctx.resources and ctx.resources.gpu:
        return RoutingDecision(profile="gpu-pool")
    return None  # decline; the run goes to the default profile
```

`ctx` carries the task name, its arguments by name, the resources it requests, the current
project/domain, and the profiles available. Returning `None` declines.

`RoutingContext` and `RoutingDecision` are defined in *this package*, not in the SDK — routing is
entirely a plugin concern, so nothing about these shapes is fixed by Flyte and you can change them
freely in a fork.

All three signals teams usually route on are reachable: **data location** from `ctx.inputs`,
**capacity** from `ctx.resources` plus whatever you probe yourself, and **user identity** from the
local user or `flyte whoami`. Note that `ctx.inputs` means the policy sees whatever was passed to
the task, which can include values you would not want a third party to see — that is a
consideration when installing someone else's routing plugin, not just when writing one.

A policy that probes a cluster should cache that itself — it runs on every submission.

## Not routed

These do not consult the router. Note that "not routed" is not the same as "not resolved" —
`rerun` and `abort` are resolved from the run name, they just do not get a fresh policy decision.

- **`rerun` and `--recover`** target the control plane holding the source run, which is a fact to
  look up rather than a decision to make.
- **`deploy`** is not routed. Deploying to a chosen control plane is what `--profile` already
  does, and a deployment is usually meant to land wherever it is pointed rather than be placed by
  a policy.
- **Local runs** have no control plane to choose between.
- **`flyte.run(...)`** — see *Two paths*.

## Trade-offs

- **`flyte.run(...)` is not routed.** The largest one, and the price of the SDK carrying no
  routing code. Use `flyteplugins.routing.run` on that path.
- Taking over run naming means taking over uniqueness. Collisions surface as
  `flyte.errors.RuntimeUserError` with code `RunAlreadyExistsError`; the random suffix makes them
  rare, not impossible.
- A run without a decodable tag is not searched for — it is read against the default profile. If
  you point a command at a run that lives elsewhere and was not named by this plugin, you get
  not-found rather than the run.
- Two profiles can collide on a tag. With no search to disambiguate, such a name falls back to the
  default profile rather than resolving to a guess.
- The CLI hook reaches into `flyte`'s command objects (`RunTaskCommand.obj`, `.run_args`) and the
  shape of the `run` group. Those are internal, so a future refactor of the CLI can break this
  plugin in a way the SDK is not obliged to avoid. A policy fault degrades to the default profile
  rather than failing the run.
- Routing decides placement per run. Nothing here moves data between clusters, and a run routed to
  a cluster that cannot reach its inputs will fail there.
