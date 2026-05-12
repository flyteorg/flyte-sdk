from flyte.models import SerializationContext

from flyteplugins.snowflake.task import Snowflake, SnowflakeConfig

SF_CONFIG = SnowflakeConfig(
    account="test-account",
    database="test-db",
    schema="PUBLIC",
    warehouse="test-wh",
    user="test-user",
)

SCTX = SerializationContext(version="test")


def _make_task(**kwargs) -> Snowflake:
    defaults = {
        "name": "test",
        "query_template": "SELECT 1",
        "plugin_config": SF_CONFIG,
    }
    defaults.update(kwargs)
    return Snowflake(**defaults)


class TestToEnvVar:
    def test_group_and_key(self):
        task = _make_task(secret_group="snowflake", snowflake_private_key="private-key")
        assert task._to_env_var("private-key") == "SNOWFLAKE_PRIVATE_KEY"

    def test_key_only(self):
        task = _make_task(snowflake_private_key="private-key")
        assert task._to_env_var("private-key") == "PRIVATE_KEY"

    def test_uppercase_and_hyphens(self):
        task = _make_task(secret_group="my-group")
        assert task._to_env_var("my-secret-key") == "MY_GROUP_MY_SECRET_KEY"

    def test_already_uppercase(self):
        task = _make_task(secret_group="SNOWFLAKE")
        assert task._to_env_var("PRIVATE_KEY") == "SNOWFLAKE_PRIVATE_KEY"


class TestCustomConfig:
    def test_secrets_with_group(self):
        task = _make_task(
            secret_group="snowflake",
            snowflake_private_key="private-key",
            snowflake_private_key_passphrase="passphrase",
        )
        config = task.custom_config(SCTX)

        assert config["secrets"] == {
            "snowflake_private_key": "SNOWFLAKE_PRIVATE_KEY",
            "snowflake_private_key_passphrase": "SNOWFLAKE_PASSPHRASE",
        }

    def test_secrets_without_group(self):
        task = _make_task(
            snowflake_private_key="private-key",
            snowflake_private_key_passphrase="passphrase",
        )
        config = task.custom_config(SCTX)

        assert config["secrets"] == {
            "snowflake_private_key": "PRIVATE_KEY",
            "snowflake_private_key_passphrase": "PASSPHRASE",
        }

    def test_no_secrets(self):
        task = _make_task()
        config = task.custom_config(SCTX)

        assert "secrets" not in config

    def test_only_private_key(self):
        task = _make_task(
            secret_group="sf",
            snowflake_private_key="pk",
        )
        config = task.custom_config(SCTX)

        assert config["secrets"] == {
            "snowflake_private_key": "SF_PK",
        }
        assert "snowflake_private_key_passphrase" not in config["secrets"]

    def test_batch_flag(self):
        task = _make_task(batch=True)
        config = task.custom_config(SCTX)
        assert config["batch"] is True

    def test_no_batch_by_default(self):
        task = _make_task()
        config = task.custom_config(SCTX)
        assert "batch" not in config

    def test_connection_kwargs(self):
        sf_config = SnowflakeConfig(
            account="a",
            database="d",
            schema="s",
            warehouse="w",
            user="u",
            connection_kwargs={"role": "ADMIN"},
        )
        task = _make_task(plugin_config=sf_config)
        config = task.custom_config(SCTX)
        assert config["connection_kwargs"] == {"role": "ADMIN"}
