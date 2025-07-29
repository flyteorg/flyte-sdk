# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=0.2.0b16",
#    "langchain-core==0.3.66",
#    "langchain-openai==0.3.24",
#    "langchain-community==0.3.26",
#    "beautifulsoup4==4.13.4",
#    "docker==7.1.0",
# ]
# ///

import tempfile
from typing import Optional

from pydantic import BaseModel, Field

import flyte
from flyte.extras import ContainerTask
from flyte.io import File

REGISTRY = "ghcr.io/flyteorg"

env = flyte.TaskEnvironment(
    name="code_runner",
    secrets=[flyte.Secret(key="OPENAI_API_KEY", as_env_var="OPENAI_API_KEY")],
    image=flyte.Image.from_uv_script(
        __file__,
        name="code-runner-agent",
        registry=REGISTRY,
        arch=("linux/amd64", "linux/arm64"),
    ),
)


class Code(BaseModel):
    """Schema for code solutions to questions about LCEL."""

    prefix: str = Field(default="", description="Description of the problem and approach")
    imports: str = Field(default="", description="Code block with just import statements")
    code: str = Field(default="", description="Code block not including import statements")


class AgentState(BaseModel):
    messages: list[dict[str, str]] = Field(default_factory=list)
    generation: Code = Field(default_factory=Code)
    iterations: int = 0
    error: str = "no"
    output: Optional[str] = None


async def generate_code_gen_chain(debug: bool):
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    # Grader prompt
    code_gen_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a coding assistant with expertise in Python.
You are able to execute the Flyte v2 code locally in a sandbox environment.

Use the following pattern to execute the code:

```
if __name__ == "__main__":
    flyte.init()
    print(flyte.run(...))
```

Your response will be shown to the user.
Here is a full set of documentation:

-------
{context}
-------

Answer the user question based on the above provided documentation.
Ensure any code you provide can be executed with all required imports and variables defined.
Structure your answer with a description of the code solution.
Then list the imports. And finally list the functioning code block.
Here is the user question:""",
            ),
            ("placeholder", "{messages}"),
        ]
    )

    expt_llm = "gpt-4o" if not debug else "gpt-4o-mini"
    llm = ChatOpenAI(temperature=0, model=expt_llm)

    code_gen_chain = code_gen_prompt | llm.with_structured_output(Code)

    return code_gen_chain


@env.task
async def docs_retriever(url: str) -> str:
    from bs4 import BeautifulSoup
    from langchain_community.document_loaders.recursive_url_loader import (
        RecursiveUrlLoader,
    )

    loader = RecursiveUrlLoader(url=url, max_depth=20, extractor=lambda x: BeautifulSoup(x, "html.parser").text)
    docs = loader.load()

    # Sort the list based on the URLs and get the text
    d_sorted = sorted(docs, key=lambda x: x.metadata["source"])
    d_reversed = list(reversed(d_sorted))

    concatenated_content = "\n\n\n --- \n\n\n".join([doc.page_content for doc in d_reversed])
    return concatenated_content


@env.task
async def generate(question: str, state: AgentState, concatenated_content: str, debug: bool) -> AgentState:
    """
    Generate a code solution

    Args:
        state (dict): The current graph state
        concatenated_content (str): The concatenated docs content
        debug (bool): Debug mode

    Returns:
        state (dict): New key added to state, generation
    """

    print("---GENERATING CODE SOLUTION---")

    # State
    messages = state.messages
    iterations = state.iterations
    error = state.error

    # We have been routed back to generation with an error
    if error == "yes":
        messages += [
            {
                "role": "user",
                "content": (
                    "Now, try again. Invoke the code tool to structure the output "
                    "with a prefix, imports, and code block:"
                ),
            }
        ]

    code_gen_chain = await generate_code_gen_chain(debug)

    # Solution
    code_solution = code_gen_chain.invoke(
        {
            "context": concatenated_content,
            "messages": (messages if messages else [{"role": "user", "content": question}]),
        }
    )

    messages += [
        {
            "role": "assistant",
            "content": f"{code_solution.prefix} \n Imports: {code_solution.imports} \n Code: {code_solution.code}",
        }
    ]

    return AgentState(
        messages=messages,
        generation=code_solution,
        iterations=iterations + 1,
        error=error,
        output=state.output,
    )


code_runner_task = ContainerTask(
    name="run_flyte_v2",
    image="ghcr.io/flyteorg/flyte:py3.13-v0.2.0b14",
    input_data_dir="/var/inputs",
    output_data_dir="/var/outputs",
    inputs={"script": File},
    outputs={"result": str, "exit_code": str},
    command=[
        "/bin/bash",
        "-c",
        (
            "set -o pipefail && "
            "uv run --script /var/inputs/script > /var/outputs/result 2>&1; "
            "echo $? > /var/outputs/exit_code"
        ),
    ],
)


@env.task
async def code_check(state: AgentState) -> AgentState:
    """
    Check code

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, error
    """

    print("---CHECKING CODE---")

    # State
    messages = state.messages
    code_solution = state.generation
    iterations = state.iterations

    # Get solution components
    imports = code_solution.imports.strip()
    code = code_solution.code.strip()

    # Create temp file for imports
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as imports_file:
        imports_file.write(imports + "\n")
        imports_path = imports_file.name

    # Create temp file for code body
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as code_file:
        code_file.write(imports + "\n" + code + "\n")
        code_path = code_file.name

    # Check imports
    import_output, import_exit_code = await code_runner_task(script=await File.from_local(imports_path))

    if import_exit_code.strip() != "0":
        print("---CODE IMPORT CHECK: FAILED---")
        error_message = [
            {
                "role": "user",
                "content": f"Your solution failed the import test: {import_output}",
            }
        ]
        messages += error_message
        return AgentState(
            generation=code_solution,
            messages=messages,
            iterations=iterations,
            error="yes",
            output=import_output,
        )
    else:
        print("---CODE IMPORT CHECK: PASSED---")

    # Check execution
    code_output, code_exit_code = await code_runner_task(script=await File.from_local(code_path))

    if code_exit_code.strip() != "0":
        print("---CODE BLOCK CHECK: FAILED---")
        error_message = [
            {
                "role": "user",
                "content": f"Your solution failed the code execution test: {code_output}",
            }
        ]
        messages += error_message
        return AgentState(
            generation=code_solution,
            messages=messages,
            iterations=iterations,
            error="yes",
            output=code_output,
        )
    else:
        print("---CODE BLOCK CHECK: PASSED---")

    # No errors
    print("---NO CODE TEST FAILURES---")

    return AgentState(
        generation=code_solution,
        messages=messages,
        iterations=iterations,
        error="no",
        output=code_output,
    )


@env.task
async def reflect(state: AgentState, concatenated_content: str, debug: bool) -> AgentState:
    """
    Reflect on errors

    Args:
        state (dict): The current graph state
        concatenated_content (str): Concatenated docs content
        debug (bool): Debug mode

    Returns:
        state (dict): New key added to state, reflection
    """

    print("---REFLECTING---")

    # State
    messages = state.messages
    iterations = state.iterations
    code_solution = state.generation

    # Prompt reflection
    code_gen_chain = await generate_code_gen_chain(debug)

    # Add reflection
    reflections = code_gen_chain.invoke({"context": concatenated_content, "messages": messages})

    messages += [
        {
            "role": "assistant",
            "content": f"Here are reflections on the error: {reflections}",
        }
    ]

    return AgentState(
        generation=code_solution,
        messages=messages,
        iterations=iterations,
        error=state.error,
        output=state.output,
    )


@env.task
async def main(
    question: str = (
        "Define a two-task pattern where the second catches OOM from the first and retries with more memory."
    ),
    # # failure case: the image does not include scikit-learn.
    # question: str = "Write a task that loads the Iris dataset, normalizes the features, and returns the result.",
    url: str = "https://pre-release-v2.docs-builder.pages.dev/docs/byoc/user-guide/",
    flag: str = "reflect",
    max_iterations: int = 3,
    debug: bool = False,
) -> str:
    concatenated_content = await docs_retriever(url=url)

    state: AgentState = AgentState()
    iterations = 0

    while True:
        with flyte.group(f"code-generation-pass-{iterations + 1}"):
            state = await generate(question, state, concatenated_content, debug)
            state = await code_check(state)

            error = state.error
            iterations = state.iterations

            if error == "no" or iterations >= max_iterations:
                print("---DECISION: FINISH---")
                code_solution = state.generation

                prefix = code_solution.prefix
                imports = code_solution.imports
                code = code_solution.code

                code_output = state.output

                return f"""{prefix}

    {imports}
    {code}

    Result of code execution:
    {code_output}
    """
            else:
                print("---DECISION: RE-TRY SOLUTION---")
                if flag == "reflect":
                    state = await reflect(state, concatenated_content, debug)


if __name__ == "__main__":
    flyte.init_from_config("config.yaml")
    run = flyte.run(main)
    print(run.url)
