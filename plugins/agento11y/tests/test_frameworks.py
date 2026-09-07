"""Registering agento11y's injectors as instrumentors for the agents plugin.

The framework integrations pull in heavy dependencies (langchain, google-adk, ...), so the
injector modules are stubbed here. What is under test is the wiring — that the right
injector is found, called with the right arguments, and reachable through the registry the
adapters use — not the integrations themselves.
"""

import sys
import types

import pytest
from flyteplugins.agents.core import apply_instrumentation, instrumented_frameworks

from flyteplugins.agento11y import SUPPORTED_FRAMEWORKS
from flyteplugins.agento11y._frameworks import register_all, unregister_all

# Frameworks instrumented by decorating a payload. Claude is instrumented by wrapping the
# adapter's call instead, so it never receives one and is excluded from these.
PAYLOAD_FRAMEWORKS = tuple(i for i in SUPPORTED_FRAMEWORKS if not i.call_wrapper)


@pytest.fixture
def stub_integrations(monkeypatch):
    """Install a fake module for every supported framework, recording the calls made."""
    calls = []

    for integration in SUPPORTED_FRAMEWORKS:
        module = types.ModuleType(integration.module)

        # payload is optional because agento11y_query, the call-wrapper entry point, takes
        # only keyword arguments — there is no payload to decorate in that shape.
        def injector(payload=None, _framework=integration.framework, **kwargs):
            calls.append((_framework, payload, kwargs))
            return {"instrumented": _framework, "payload": payload}

        setattr(module, integration.function, injector)
        monkeypatch.setitem(sys.modules, integration.module, module)

    yield calls
    unregister_all()


def test_every_supported_framework_gets_an_instrumentor(stub_integrations):
    registered = register_all(client="CLIENT")
    assert set(registered) == {i.framework for i in SUPPORTED_FRAMEWORKS}
    # instrumented_frameworks tracks payload instrumentors, so the call-wrapper ones are
    # registered but absent from it.
    assert {i.framework for i in PAYLOAD_FRAMEWORKS} <= set(instrumented_frameworks())


def test_the_adapter_facing_registry_reaches_the_injector(stub_integrations):
    """This is the path an adapter actually takes: core registry in, injector out."""
    register_all(client="CLIENT")
    result = apply_instrumentation("langgraph", {"callbacks": []})
    assert result == {"instrumented": "langgraph", "payload": {"callbacks": []}}


def test_the_client_is_passed_to_every_injector(stub_integrations):
    register_all(client="CLIENT")
    for integration in SUPPORTED_FRAMEWORKS:
        apply_instrumentation(integration.framework, {})
    assert {kwargs["client"] for _, _, kwargs in stub_integrations} == {"CLIENT"}


def test_async_handler_is_passed_only_where_the_injector_accepts_it(stub_integrations):
    """Passing it elsewhere is swallowed by **handler_kwargs and misconfigures the handler."""
    register_all(client="CLIENT")
    for integration in PAYLOAD_FRAMEWORKS:
        apply_instrumentation(integration.framework, {})

    passed = {framework: "async_handler" in kwargs for framework, _, kwargs in stub_integrations}
    expected = {i.framework: i.async_handler for i in PAYLOAD_FRAMEWORKS}
    assert passed == expected


def test_each_instrumentor_keeps_its_own_framework(stub_integrations):
    """A loop that leaks its variable would route every framework to the last injector."""
    register_all(client="CLIENT")
    for integration in PAYLOAD_FRAMEWORKS:
        result = apply_instrumentation(integration.framework, {})
        # Frameworks whose adapter offers the whole run payload get the injector's result
        # nested under the key it owns, rather than being the result themselves.
        if integration.payload_is_run_kwargs:
            result = result[integration.payload_is_run_kwargs]
        assert result["instrumented"] == integration.framework


def test_a_run_kwargs_payload_gets_the_conversation_id(stub_integrations, monkeypatch):
    """Frameworks that name a conversation natively should be bound to the Flyte run."""
    import flyteplugins.agento11y._frameworks as frameworks

    monkeypatch.setattr(frameworks, "_current_run_name", lambda: "run-abc")
    register_all(client="CLIENT")

    result = apply_instrumentation("pydantic_ai", {"message_history": []})
    assert result["conversation_id"] == "run-abc"
    assert result["message_history"] == [], "unrelated run kwargs must survive"
    assert result["capabilities"]["instrumented"] == "pydantic_ai"


def test_a_caller_conversation_id_is_not_overridden(stub_integrations, monkeypatch):
    import flyteplugins.agento11y._frameworks as frameworks

    monkeypatch.setattr(frameworks, "_current_run_name", lambda: "run-abc")
    register_all(client="CLIENT")

    result = apply_instrumentation("pydantic_ai", {"conversation_id": "mine"})
    assert result["conversation_id"] == "mine"


def test_frameworks_whose_package_is_missing_are_skipped():
    """Nobody installs all six; absence must be quiet rather than an error."""
    registered = register_all(client="CLIENT")
    assert registered == ()
    unregister_all()


def test_unregister_leaves_the_registry_clean(stub_integrations):
    from flyteplugins.agents.core import apply_call_wrapper

    register_all(client="CLIENT")
    unregister_all()
    for integration in SUPPORTED_FRAMEWORKS:
        payload = {"untouched": True}
        assert apply_instrumentation(integration.framework, payload) is payload

        def original():
            return "plain"

        assert apply_call_wrapper(integration.framework, original) is original


def test_claude_is_instrumented_by_wrapping_the_call(stub_integrations):
    """Its model turns arrive as stream messages, so decorating options cannot record them.

    agento11y_query drives the SDK's own query and records each message, so the adapter's
    loop over the stream is unchanged while generations finally get created.
    """
    from flyteplugins.agents.core import apply_call_wrapper

    register_all(client="CLIENT")

    def sdk_query(*, prompt, options):
        return f"stream:{prompt}"

    wrapped = apply_call_wrapper("claude", sdk_query)
    assert wrapped is not sdk_query, "the SDK's query must be replaced, not used directly"

    wrapped(prompt="hi", options="OPTS")
    framework, _, kwargs = stub_integrations[-1]
    assert framework == "claude"
    assert kwargs["client"] == "CLIENT"
    assert kwargs["_query_fn"] is sdk_query, "the real query has to be threaded through"
    assert kwargs["prompt"] == "hi"


def test_pydantic_ai_gets_the_conversation_in_metadata_too(stub_integrations, monkeypatch):
    """agento11y's capability reads RunContext.metadata, not RunContext.conversation_id.

    Setting only the native kwarg leaves the capability synthesizing its own id from ctx.run_id,
    so the run's generations are not grouped by the Flyte run.
    """
    import flyteplugins.agento11y._frameworks as frameworks

    monkeypatch.setattr(frameworks, "_current_run_name", lambda: "run-abc")
    register_all(client="CLIENT")

    result = apply_instrumentation("pydantic_ai", {})
    assert result["metadata"]["conversation_id"] == "run-abc"
    assert result["conversation_id"] == "run-abc", "pydantic-ai's own telemetry uses this"


def test_a_metadata_factory_is_left_alone(stub_integrations, monkeypatch):
    """Pydantic AI allows metadata to be a callable; dict() on one raises."""
    import flyteplugins.agento11y._frameworks as frameworks

    monkeypatch.setattr(frameworks, "_current_run_name", lambda: "run-abc")
    register_all(client="CLIENT")

    factory = lambda ctx: {"k": "v"}  # noqa: E731
    result = apply_instrumentation("pydantic_ai", {"metadata": factory})
    assert result["metadata"] is factory


def test_the_claude_call_wrapper_binds_the_conversation(stub_integrations, monkeypatch):
    """Claude's handler takes a conversation id and prefers it over the SDK session id.

    Without it the run's generations group under a session uuid that appears nowhere else.
    """
    from flyteplugins.agents.core import apply_call_wrapper

    import flyteplugins.agento11y._frameworks as frameworks

    monkeypatch.setattr(frameworks, "_current_run_name", lambda: "run-abc")
    register_all(client="CLIENT")

    def sdk_query(*, prompt, options):
        return "stream"

    apply_call_wrapper("claude", sdk_query)(prompt="hi", options="OPTS")
    _, _, kwargs = stub_integrations[-1]
    assert kwargs["conversation_id"] == "run-abc"
