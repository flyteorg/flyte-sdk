from flyte.config._reader import ConfigEntry, YamlConfigEntry


class Platform(object):
    URL = ConfigEntry(YamlConfigEntry("admin.endpoint"))
    INSECURE = ConfigEntry(YamlConfigEntry("admin.insecure", bool))
    INSECURE_SKIP_VERIFY = ConfigEntry(YamlConfigEntry("admin.insecureSkipVerify", bool))
    CONSOLE_ENDPOINT = ConfigEntry(YamlConfigEntry("console.endpoint"))
    CA_CERT_FILE_PATH = ConfigEntry(YamlConfigEntry("admin.caCertFilePath"))
    HTTP_PROXY_URL = ConfigEntry(YamlConfigEntry("admin.httpProxyURL"))


class Credentials(object):
    SECTION = "credentials"
    COMMAND = ConfigEntry(YamlConfigEntry("admin.command", list))
    """
    This command is executed to return a token using an external process.
    """

    PROXY_COMMAND = ConfigEntry(YamlConfigEntry("admin.proxyCommand", list))
    """
    This command is executed to return a token for authorization with a proxy
     in front of Flyte using an external process.
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

    SCOPES = ConfigEntry(YamlConfigEntry("admin.scopes", list))
    """
    This setting can be used to manually pass in scopes into authenticator flows - eg.) for Auth0 compatibility
    """

    AUTH_MODE = ConfigEntry(YamlConfigEntry("admin.authType"))
    """
    The auth mode defines the behavior used to request and refresh credentials. The currently supported modes include:
    - 'standard' or 'Pkce': This uses the pkce-enhanced authorization code flow by opening a browser window to initiate
            credentials access.
    - "DeviceFlow": This uses the Device Authorization Flow
    - 'basic', 'client_credentials' or 'clientSecret': This uses symmetric key auth in which the end user enters a
            client id and a client secret and public key encryption is used to facilitate authentication.
    - None: No auth will be attempted.
    """


class Task(object):
    ORG = ConfigEntry(YamlConfigEntry("task.org"))
    PROJECT = ConfigEntry(YamlConfigEntry("task.project"))
    DOMAIN = ConfigEntry(YamlConfigEntry("task.domain"))


class Image(object):
    """
    Defines the configuration for the image builder.
    """

    BUILDER = ConfigEntry(YamlConfigEntry("image.builder"))
    IMAGE_REFS = ConfigEntry(YamlConfigEntry("image.image_refs"))
