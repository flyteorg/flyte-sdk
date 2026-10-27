"""Constants shared by the oMLX plugin."""

OMLX_MIN_VERSION = (0, 1, 0)
OMLX_MIN_VERSION_STR = ".".join(map(str, OMLX_MIN_VERSION))

# oMLX's own default serve port. Kept distinct from the SDK's 8080 so that
# "just construct an OMLXAppEnvironment" matches what `omlx serve` would bind
# if you ran it by hand.
DEFAULT_OMLX_PORT = 8000
