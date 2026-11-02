from flyte.app._base_path import (
    STREAMLIT_BASE_URL_PATH_ENV_VAR,
    app_base_path,
    apply_base_path_defaults,
)

BASE = "/onebox/apps/my-app"


def test_app_base_path():
    assert app_base_path({"_U_APP_BASE_PATH": BASE}) == BASE
    assert app_base_path({"_U_APP_BASE_PATH": BASE + "/"}) == BASE
    assert app_base_path({}) == ""


def test_streamlit_serves_under_the_app_base_path():
    env = {"_U_APP_BASE_PATH": BASE + "/"}
    apply_base_path_defaults(env)
    assert env[STREAMLIT_BASE_URL_PATH_ENV_VAR] == BASE


def test_an_app_setting_its_own_streamlit_base_path_keeps_it():
    env = {"_U_APP_BASE_PATH": BASE, STREAMLIT_BASE_URL_PATH_ENV_VAR: "/custom"}
    apply_base_path_defaults(env)
    assert env[STREAMLIT_BASE_URL_PATH_ENV_VAR] == "/custom"


def test_apps_at_their_host_root_are_unchanged():
    env = {}
    apply_base_path_defaults(env)
    assert env == {}
