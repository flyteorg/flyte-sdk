FLYTE_SYS_PATH = "_F_SYS_PATH"  # The paths that will be appended to sys.path at runtime

# Reserved env var through which the server points a fork's task pods at its action input
# overrides file (see flyte._internal.controllers.remote._input_overrides). Never inherited.
ACTION_INPUT_OVERRIDES_ENV_VAR = "_U_ACTION_INPUT_OVERRIDES"
