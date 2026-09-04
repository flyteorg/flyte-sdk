import json
import re
import shlex
import typing

from flyte.config._reader import ConfigEntry, YamlConfigEntry


def _as_argv(v: typing.Any) -> typing.Optional[typing.List[str]]:
    """Normalize a config value that must end up as an argv list.

    YAML already yields a list, and that is passed through untouched. An
    environment variable can only ever carry a string, so accept both a JSON
    array (``["uctl", "get-token"]``) and a plain shell-quoted command line
    (``uctl get-token --audience foo``), which is what people reach for first.
    Without this, ``FLYTE_ADMIN_COMMAND`` reached the external-command
    authenticator as a bare string and ``create_subprocess_exec(*cmd)`` spread
    it one character per argument.
    """
    if v is None or isinstance(v, list):
        return v
    s = str(v).strip()
    if not s:
        return None
    if s.startswith("["):
        try:
            parsed = json.loads(s)
        except json.JSONDecodeError:
            pass
        else:
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
    return shlex.split(s)


def _as_str_list(v: typing.Any) -> typing.Optional[typing.List[str]]:
    """Normalize a config value that must end up as a list of strings.

    Same reasoning as `_as_argv`, but for value lists (scopes) rather than an
    argv: accept a JSON array or a comma/whitespace separated string.
    """
    if v is None or isinstance(v, list):
        return v
    s = str(v).strip()
    if not s:
        return None
    if s.startswith("["):
        try:
            parsed = json.loads(s)
        except json.JSONDecodeError:
            pass
        else:
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
    return [p for p in re.split(r"[,\s]+", s) if p]


class Platform(object):
    URL = ConfigEntry(YamlConfigEntry("admin.endpoint"))
    INSECURE = ConfigEntry(YamlConfigEntry("admin.insecure", bool))
    INSECURE_SKIP_VERIFY = ConfigEntry(YamlConfigEntry("admin.insecureSkipVerify", bool))
    CONSOLE_ENDPOINT = ConfigEntry(YamlConfigEntry("console.endpoint"))
    CA_CERT_FILE_PATH = ConfigEntry(YamlConfigEntry("admin.caCertFilePath"))
    HTTP_PROXY_URL = ConfigEntry(YamlConfigEntry("admin.httpProxyURL"))
    DISABLE_KEYRING = ConfigEntry(YamlConfigEntry("admin.disableKeyring", bool))


class Credentials(object):
    SECTION = "credentials"
    COMMAND = ConfigEntry(YamlConfigEntry("admin.command", list, aliases=("FLYTE_AUTH_COMMAND",)), transform=_as_argv)
    """
    This command is executed to return a token using an external process.

    Env var: `FLYTE_AUTH_COMMAND` (the derived `FLYTE_ADMIN_COMMAND` is still accepted).
    """

    PROXY_COMMAND = ConfigEntry(
        YamlConfigEntry("admin.proxyCommand", list, aliases=("FLYTE_AUTH_PROXY_COMMAND",)), transform=_as_argv
    )
    """
    This command is executed to return a token for authorization with a proxy
     in front of Flyte using an external process.

    Env var: `FLYTE_AUTH_PROXY_COMMAND` (the derived `FLYTE_ADMIN_PROXYCOMMAND` is still accepted).
    """

    CLIENT_ID = ConfigEntry(YamlConfigEntry("admin.clientId"))
    """
    This is the public identifier for the app which handles authorization for a Flyte deployment.
    More details here: https://www.oauth.com/oauth2-servers/client-registration/client-id-secret/.
    """

    CLIENT_CREDENTIALS_SECRET_LOCATION = ConfigEntry(YamlConfigEntry("admin.clientSecretLocation"))
    """
    Used for basic auth, which is automatically called during pyflyte. This will allow the Flyte engine to read the
    password from a mounted file.
    """

    CLIENT_CREDENTIALS_SECRET_ENV_VAR = ConfigEntry(YamlConfigEntry("admin.clientSecretEnvVar"))
    """
    Used for basic auth, which is automatically called during pyflyte. This will allow the Flyte engine to read the
    password from a mounted environment variable.
    """

    SCOPES = ConfigEntry(YamlConfigEntry("admin.scopes", list), transform=_as_str_list)
    """
    This setting can be used to manually pass in scopes into authenticator flows - eg.) for Auth0 compatibility
    """

    AUTH_MODE = ConfigEntry(YamlConfigEntry("admin.authType", aliases=("FLYTE_AUTH_TYPE",)))
    """
    Env var: `FLYTE_AUTH_TYPE` (the derived `FLYTE_ADMIN_AUTHTYPE` is still accepted).

    The auth mode defines the behavior used to request and refresh credentials. The value must be one of the
    `flyte.remote._client.auth.AuthType` literals:
    - 'Pkce' (default): the pkce-enhanced authorization code flow, which opens a browser window to initiate
            credentials access.
    - 'DeviceFlow': the Device Authorization Flow, for hosts without a browser.
    - 'ClientSecret': symmetric key auth, in which a client id and a client secret are exchanged for a token.
    - 'ExternalCommand': `COMMAND` is executed and its stdout is used as the access token. Use this to plug in an
            external token minter (see the `admin.command` entry).
    - 'Passthrough': auth metadata is supplied per-call by the caller.
    """


class Task(object):
    ORG = ConfigEntry(YamlConfigEntry("task.org"))
    PROJECT = ConfigEntry(YamlConfigEntry("task.project"))
    DOMAIN = ConfigEntry(YamlConfigEntry("task.domain"))


class Local(object):
    PERSISTENCE = ConfigEntry(YamlConfigEntry("local.persistence", bool))
    # Tracked-run reporting (TrackedRunService). Section name stays `local.` — these are
    # local-execution settings and the keys are user-facing.
    TRACKED = ConfigEntry(YamlConfigEntry("local.tracked", bool))
    TRACKED_STRICT = ConfigEntry(YamlConfigEntry("local.tracked_strict", bool))


class Image(object):
    """
    Defines the configuration for the image builder.
    """

    BUILDER = ConfigEntry(YamlConfigEntry("image.builder"))
    IMAGE_REFS = ConfigEntry(YamlConfigEntry("image.image_refs"))
    REGISTRY = ConfigEntry(YamlConfigEntry("image.registry"))
    """
    The container registry to use as the base registry when building images (e.g. `ghcr.io/my-org`).
    Read from the `image.registry` config entry or the `FLYTE_IMAGE_REGISTRY` environment variable.
    When set, this overrides the built-in default base registry.
    """
