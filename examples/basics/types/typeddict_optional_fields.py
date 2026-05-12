"""
Example demonstrating TypedDict with NotRequired/Optional fields in Flyte tasks.

This example specifically demonstrates the following fixes:
1. Self-referential TypedDicts (cache persistence for recursive types)
2. NotRequired fields are properly omitted from output when not provided
   (not set to None, which would break `if 'key' in result` checks)
3. NotRequired[T] type hints are properly unwrapped for Pydantic validation

These examples can be run via CLI to verify the fixes work correctly:

    flyte run examples/basics/types/typeddict_optional_fields.py process_ai_response \
        --response '{"content": "Hello!", "role": "assistant"}'

    flyte run examples/basics/types/typeddict_optional_fields.py process_ai_response \
        --response \'{"content": "Hi", "role": "assistant", "tool_calls": [{"name": "search", "args": {"q": "test"}}]}'
"""

from __future__ import annotations

from typing import List, TypedDict

from typing_extensions import NotRequired

import flyte

env = flyte.TaskEnvironment(name="typeddict_optional_fields_example")


# ============================================================================
# TypedDict with NotRequired fields
# Demonstrates that optional fields not provided are ABSENT from output,
# not set to None (which would break downstream `if 'key' in result` checks)
# ============================================================================


class ToolCall(TypedDict):
    """A tool call made by an AI assistant."""

    name: str
    args: dict


class AIResponse(TypedDict):
    """Response from an AI assistant.

    The tool_calls field is NotRequired - it should be absent from output
    when not provided, not set to None.
    """

    content: str
    role: str
    tool_calls: NotRequired[List[ToolCall]]


class AIResponseOutput(TypedDict):
    """Output containing processed AI response info."""

    message: str
    has_tool_calls: bool
    tool_call_names: List[str]


@env.task
async def create_response_without_tool_calls() -> AIResponse:
    """Create an AI response without tool_calls.

    The returned dict should NOT contain 'tool_calls' key at all.
    """
    return AIResponse(content="Hello! How can I help you?", role="assistant")


@env.task
async def create_response_with_tool_calls() -> AIResponse:
    """Create an AI response with tool_calls."""
    return AIResponse(
        content="Let me search for that.",
        role="assistant",
        tool_calls=[
            ToolCall(name="web_search", args={"query": "flyte documentation"}),
            ToolCall(name="code_search", args={"pattern": "TypedDict"}),
        ],
    )


@env.task
async def process_ai_response(response: AIResponse) -> AIResponseOutput:
    """Process an AI response, checking for tool_calls using 'in' operator.

    This demonstrates that NotRequired fields not provided are absent,
    not set to None. The `if 'tool_calls' in response` check works correctly.
    """
    # This check would fail if tool_calls was set to None instead of being absent
    has_tool_calls = "tool_calls" in response

    if has_tool_calls:
        tool_call_names = [tc["name"] for tc in response["tool_calls"]]
        message = f"Response has {len(tool_call_names)} tool calls: {', '.join(tool_call_names)}"
    else:
        tool_call_names = []
        message = f"Response from {response['role']}: {response['content'][:50]}..."

    return AIResponseOutput(
        message=message,
        has_tool_calls=has_tool_calls,
        tool_call_names=tool_call_names,
    )


@env.task
async def verify_notrequired_fields_absent(response: AIResponse) -> dict:
    """Verify that NotRequired fields are absent when not provided.

    Returns a dict with verification results.
    """
    keys_present = list(response.keys())
    tool_calls_in_response = "tool_calls" in response
    tool_calls_value = response.get("tool_calls", "NOT_PRESENT")

    return {
        "keys_present": keys_present,
        "tool_calls_in_response": tool_calls_in_response,
        "tool_calls_value": str(tool_calls_value),
        "verification_passed": not tool_calls_in_response or response["tool_calls"] is not None,
    }


# ============================================================================
# Self-referential TypedDict
# Demonstrates that the cache persistence fix works for recursive types
# ============================================================================


class TreeNode(TypedDict):
    """A tree node that can reference itself.

    This tests the cache persistence fix for self-referential TypedDicts.
    """

    value: str
    children: NotRequired[List[TreeNode]]


class TreeOutput(TypedDict):
    """Output from tree processing."""

    total_nodes: int
    max_depth: int
    leaf_values: List[str]


def count_nodes(node: TreeNode) -> int:
    """Count total nodes in a tree."""
    count = 1
    if "children" in node:
        for child in node["children"]:
            count += count_nodes(child)
    return count


def get_max_depth(node: TreeNode, current_depth: int = 1) -> int:
    """Get the maximum depth of a tree."""
    if "children" not in node or not node["children"]:
        return current_depth
    return max(get_max_depth(child, current_depth + 1) for child in node["children"])


def get_leaf_values(node: TreeNode) -> List[str]:
    """Get all leaf node values."""
    if "children" not in node or not node["children"]:
        return [node["value"]]
    leaves = []
    for child in node["children"]:
        leaves.extend(get_leaf_values(child))
    return leaves


@env.task
async def create_tree() -> TreeNode:
    """Create a self-referential tree structure."""
    return TreeNode(
        value="root",
        children=[
            TreeNode(
                value="child1",
                children=[
                    TreeNode(value="grandchild1a"),
                    TreeNode(value="grandchild1b"),
                ],
            ),
            TreeNode(
                value="child2",
                children=[
                    TreeNode(value="grandchild2a"),
                ],
            ),
            TreeNode(value="child3"),  # Leaf node - no children key
        ],
    )


@env.task
async def process_tree(tree: TreeNode) -> TreeOutput:
    """Process a self-referential tree structure."""
    return TreeOutput(
        total_nodes=count_nodes(tree),
        max_depth=get_max_depth(tree),
        leaf_values=get_leaf_values(tree),
    )


# ============================================================================
# Mixed TypedDict with multiple NotRequired fields
# ============================================================================


class UserProfile(TypedDict):
    """User profile with multiple optional fields."""

    username: str
    email: str
    display_name: NotRequired[str]
    bio: NotRequired[str]
    avatar_url: NotRequired[str]
    settings: NotRequired[dict]


class ProfileSummary(TypedDict):
    """Summary of a user profile."""

    username: str
    fields_provided: List[str]
    fields_missing: List[str]


@env.task
async def create_minimal_profile() -> UserProfile:
    """Create a profile with only required fields."""
    return UserProfile(username="alice", email="alice@example.com")


@env.task
async def create_full_profile() -> UserProfile:
    """Create a profile with all fields."""
    return UserProfile(
        username="bob",
        email="bob@example.com",
        display_name="Bob Smith",
        bio="Software engineer passionate about data pipelines.",
        avatar_url="https://example.com/avatars/bob.png",
        settings={"theme": "dark", "notifications": True},
    )


@env.task
async def summarize_profile(profile: UserProfile) -> ProfileSummary:
    """Summarize which optional fields are present in a profile.

    This demonstrates that NotRequired fields are properly absent when not provided.
    """
    all_optional_fields = ["display_name", "bio", "avatar_url", "settings"]
    fields_provided = [f for f in all_optional_fields if f in profile]
    fields_missing = [f for f in all_optional_fields if f not in profile]

    return ProfileSummary(
        username=profile["username"],
        fields_provided=fields_provided,
        fields_missing=fields_missing,
    )


# ============================================================================
# Nested TypedDict with NotRequired at multiple levels
# ============================================================================


class Address(TypedDict):
    """Address with optional fields."""

    street: str
    city: str
    country: str
    postal_code: NotRequired[str]
    state: NotRequired[str]


class ContactInfo(TypedDict):
    """Contact information with optional nested address."""

    phone: str
    address: NotRequired[Address]


class Organization(TypedDict):
    """Organization with nested optional fields."""

    name: str
    contact: NotRequired[ContactInfo]
    website: NotRequired[str]


@env.task
async def create_minimal_org() -> Organization:
    """Create an organization with only required fields."""
    return Organization(name="Acme Corp")


@env.task
async def create_full_org() -> Organization:
    """Create an organization with all nested optional fields."""
    return Organization(
        name="TechStartup Inc",
        website="https://techstartup.example.com",
        contact=ContactInfo(
            phone="+1-555-0123",
            address=Address(
                street="123 Innovation Way",
                city="San Francisco",
                country="USA",
                state="CA",
                postal_code="94105",
            ),
        ),
    )


@env.task
async def describe_org(org: Organization) -> str:
    """Describe an organization, handling missing optional fields gracefully."""
    parts = [f"Organization: {org['name']}"]

    if "website" in org:
        parts.append(f"Website: {org['website']}")

    if "contact" in org:
        contact = org["contact"]
        parts.append(f"Phone: {contact['phone']}")
        if "address" in contact:
            addr = contact["address"]
            addr_parts = [addr["street"], addr["city"]]
            if "state" in addr:
                addr_parts.append(addr["state"])
            addr_parts.append(addr["country"])
            if "postal_code" in addr:
                addr_parts.append(addr["postal_code"])
            parts.append(f"Address: {', '.join(addr_parts)}")

    return " | ".join(parts)


# ============================================================================
# Workflow outputs
# ============================================================================


class WorkflowOutputs(TypedDict):
    """Outputs from the optional fields workflow."""

    response_without_tools: AIResponseOutput
    response_with_tools: AIResponseOutput
    verification_without_tools: dict
    verification_with_tools: dict
    tree_output: TreeOutput
    minimal_profile_summary: ProfileSummary
    full_profile_summary: ProfileSummary
    minimal_org_description: str
    full_org_description: str


@env.task
async def optional_fields_workflow() -> WorkflowOutputs:
    """Workflow demonstrating NotRequired field handling."""
    print("=== TypedDict Optional Fields Workflow ===\n")

    # Test NotRequired fields being absent vs present
    print("1. Testing AI response without tool_calls...")
    response_without = await create_response_without_tool_calls()
    output_without = await process_ai_response(response=response_without)
    verification_without = await verify_notrequired_fields_absent(response=response_without)
    print(f"   Output: {output_without}")
    print(f"   Verification: {verification_without}")

    print("\n2. Testing AI response with tool_calls...")
    response_with = await create_response_with_tool_calls()
    output_with = await process_ai_response(response=response_with)
    verification_with = await verify_notrequired_fields_absent(response=response_with)
    print(f"   Output: {output_with}")
    print(f"   Verification: {verification_with}")

    # Test self-referential TypedDict
    print("\n3. Testing self-referential tree structure...")
    tree = await create_tree()
    tree_output = await process_tree(tree=tree)
    print(f"   Tree output: {tree_output}")

    # Test multiple NotRequired fields
    print("\n4. Testing profile with minimal fields...")
    minimal_profile = await create_minimal_profile()
    minimal_summary = await summarize_profile(profile=minimal_profile)
    print(f"   Summary: {minimal_summary}")

    print("\n5. Testing profile with all fields...")
    full_profile = await create_full_profile()
    full_summary = await summarize_profile(profile=full_profile)
    print(f"   Summary: {full_summary}")

    # Test nested optional fields
    print("\n6. Testing organization with minimal fields...")
    minimal_org = await create_minimal_org()
    minimal_org_desc = await describe_org(org=minimal_org)
    print(f"   Description: {minimal_org_desc}")

    print("\n7. Testing organization with all nested fields...")
    full_org = await create_full_org()
    full_org_desc = await describe_org(org=full_org)
    print(f"   Description: {full_org_desc}")

    print("\n=== Workflow Complete ===")

    return WorkflowOutputs(
        response_without_tools=output_without,
        response_with_tools=output_with,
        verification_without_tools=verification_without,
        verification_with_tools=verification_with,
        tree_output=tree_output,
        minimal_profile_summary=minimal_summary,
        full_profile_summary=full_summary,
        minimal_org_description=minimal_org_desc,
        full_org_description=full_org_desc,
    )


if __name__ == "__main__":
    flyte.init_from_config()

    print("Running TypedDict optional fields workflow...")
    run = flyte.run(optional_fields_workflow)
    print(f"Run URL: {run.url}")
    run.wait()
    print("Workflow completed!")
    outputs = run.outputs()
    print(f"Outputs: {outputs}")
