import sys

import polyglot_hello

import flyte

env = flyte.TaskEnvironment(
    name="hello_polyglot",
    resources=flyte.Resources(memory="250Mi"),
    image=flyte.Image.from_debian_base().with_pip_packages("polyglot-hello"),
)


@env.task
def hello_for_code(code: str) -> tuple[str, str, str]:
    greeting = polyglot_hello.get_by_code(code)
    return greeting.code, greeting.name, greeting.hello


@env.task
def main(letter: str) -> dict[str, str]:
    if not isinstance(letter, str) or len(letter) != 1 or not letter.isalpha():
        raise ValueError("letter must be a single alphabetic character, e.g., 'e' or 'S'")

    letter_lower = letter.lower()

    languages = polyglot_hello.list_languages()

    matching_primary_codes: list[str] = []
    for index in range(len(languages)):
        greeting = polyglot_hello.get_by_index(index)
        all_codes = [greeting.code, *list(greeting.codes or [])]
        all_codes_lower = [c.lower() for c in all_codes]
        if any(c.startswith(letter_lower) for c in all_codes_lower):
            matching_primary_codes.append(greeting.code)

    results = list(flyte.map(hello_for_code, matching_primary_codes))
    print(results)
    return {code: hello for (code, _name, hello) in results}


if __name__ == "__main__":
    flyte.init_from_config("../config.yaml")

    input_letter = sys.argv[1] if len(sys.argv) > 1 else "e"
    execution = flyte.run(main, letter=input_letter)

    print(execution.name)
    print(execution.url)

    execution.wait(execution)
