FLYTE_SYS_PATH = "_F_SYS_PATH"  # The paths that will be appended to sys.path at runtime

# Reserved Inputs.context key through which a fork's client points every action of the run at its
# overrides file (a serialized flyteidl2.task.ActionOverridesData; see
# flyte._internal.controllers.remote._input_overrides). Set on the root action's inputs, it
# propagates to every descendant like any context key, but is hidden from get_custom_context(),
# cannot be set or dropped with flyte.custom_context(...), and is not carried into a rerun.
ACTION_OVERRIDES_CONTEXT_KEY = "_u_action_overrides"
