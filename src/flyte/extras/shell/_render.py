from __future__ import annotations

import pathlib
import re
import shlex
from typing import Any, Tuple

from ._types import FlagSpec, Stderr, Stdout, _classify_input

_PLACEHOLDER_RE = re.compile(r"\{(inputs|flags|outputs)\.([a-zA-Z_][a-zA-Z0-9_]*)\}")
_DICT_SEP = "\x1e"


def _render_command(
    script: str,
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    flag_specs: dict[str, FlagSpec],
    input_data_dir: pathlib.Path,
    output_data_dir: pathlib.Path,
) -> Tuple[str, list[str]]:
    kinds = {name: _classify_input(name, tp) for name, tp in inputs.items()}

    preamble_lines: list[str] = []
    positional_templates: list[str] = []
    slot_var_for: dict[str, str] = {}
    flag_emitted: set[str] = set()
    dict_decoded: set[str] = set()

    def alloc_slot(name: str) -> str:
        if name in slot_var_for:
            return slot_var_for[name]

        idx = len(positional_templates) + 1
        positional_templates.append(f"{{{{.inputs.{name}}}}}")
        var = f"_VAL_{name.upper()}"
        preamble_lines.append(f'{var}="${idx}"')
        slot_var_for[name] = var
        return var

    def ensure_dict_decoded(name: str) -> str:
        val_var = alloc_slot(name)
        arr_var = f"_ARR_{name.upper()}"

        if name not in dict_decoded:
            dict_decoded.add(name)
            preamble_lines.append(
                f"{arr_var}=(); "
                f'if [ -n "${{{val_var}}}" ]; then '
                f"IFS=$'\\x1e' read -ra {arr_var} <<< \"${{{val_var}}}\"; "
                f"fi"
            )
        return arr_var

    def render_input_ref(name: str) -> str:
        kind = kinds.get(name)

        if kind is None:
            raise KeyError(f"{{inputs.{name}}} used in script but {name!r} is not declared in inputs.")
        if kind in ("file", "dir"):
            return str(input_data_dir / name)
        if kind == "list_file":
            return f"{input_data_dir / name}/*"
        if kind in ("scalar", "bool"):
            return f'"${{{alloc_slot(name)}}}"'
        if kind == "dict_str":
            arr = ensure_dict_decoded(name)
            return f'"${{{arr}[@]}}"'
        raise AssertionError(kind)

    def render_flag_ref(name: str) -> str:
        if name not in flag_specs:
            raise KeyError(
                f"{{flags.{name}}} used in script but {name!r} is not in inputs. "
                f"Did you forget to declare it or did you misspell?"
            )

        kind = kinds[name]
        spec = flag_specs[name]
        flag_var = f"_FLAG_{name.upper()}"

        if name not in flag_emitted:
            flag_emitted.add(name)
            preamble_lines.append(
                _emit_flag_setter(
                    name,
                    kind,
                    spec,
                    flag_var,
                    alloc_slot,
                    ensure_dict_decoded,
                    input_data_dir,
                )
            )
        if kind in ("list_file", "dict_str"):
            return f'"${{{flag_var}[@]}}"'
        return f"${{{flag_var}}}"

    def replace(match: re.Match) -> str:
        ns, name = match.group(1), match.group(2)

        if ns == "inputs":
            return render_input_ref(name)
        if ns == "flags":
            return render_flag_ref(name)
        if ns == "outputs":
            if name not in outputs:
                raise KeyError(f"{{outputs.{name}}} references an unknown output. Declared outputs: {list(outputs)}")
            spec = outputs[name]
            if isinstance(spec, (Stdout, Stderr)):
                raise KeyError(
                    f"{{outputs.{name}}} is not valid for Stdout/Stderr collectors — "
                    f"the wrapper redirects the script's stream to the canonical "
                    f"path directly, so you don't (and shouldn't) write to it from "
                    f"your script."
                )
            return str(output_data_dir / name)
        raise KeyError(match.group(0))

    body = _PLACEHOLDER_RE.sub(replace, script)

    leftover = re.search(r"\{[a-zA-Z_]\w*\.[a-zA-Z_]\w*\}", body)
    if leftover:
        raise ValueError(
            f"Unrecognized placeholder in script: {leftover.group(0)}. "
            f"Use {{inputs.<name>}}, {{flags.<name>}}, or {{outputs.<name>}}."
        )

    if preamble_lines:
        body = "\n".join(preamble_lines) + "\n" + body
    return body, positional_templates


def _emit_flag_setter(
    name: str,
    kind: str,
    spec: FlagSpec,
    flag_var: str,
    alloc_slot,
    ensure_dict_decoded,
    input_data_dir: pathlib.Path,
) -> str:
    flag = spec.flag
    sep = spec.separator

    if kind == "bool":
        val_var = alloc_slot(name)
        return f'if [ "${{{val_var}}}" = "true" ]; then {flag_var}={shlex.quote(flag)}; else {flag_var}=""; fi'
    if kind == "scalar":
        val_var = alloc_slot(name)
        return (
            f'if [ -n "${{{val_var}}}" ]; then '
            f'{flag_var}={shlex.quote(flag + sep)}"${{{val_var}}}"; '
            f'else {flag_var}=""; fi'
        )
    if kind in ("file", "dir"):
        path = input_data_dir / name
        return f"{flag_var}={shlex.quote(flag + sep + str(path))}"
    if kind == "list_file":
        dirpath = input_data_dir / name
        if spec.list_mode == "join":
            return (
                f"{flag_var}=({shlex.quote(flag)}); "
                f'for _f in {dirpath}/*; do {flag_var}+=("$_f"); done; '
                f'if [ "${{#{flag_var}[@]}}" -le 1 ]; then {flag_var}=(); fi'
            )
        if spec.list_mode == "repeat":
            return (
                f"{flag_var}=(); "
                f"for _f in {dirpath}/*; do "
                f'if [ -e "$_f" ]; then {flag_var}+=({shlex.quote(flag)} "$_f"); fi; '
                f"done"
            )
        if spec.list_mode == "comma":
            return (
                f'_joined=""; '
                f"for _f in {dirpath}/*; do "
                f'if [ -e "$_f" ]; then _joined="${{_joined}}${{_joined:+,}}$_f"; fi; '
                f"done; "
                f'if [ -n "$_joined" ]; then '
                f'{flag_var}=({shlex.quote(flag)} "$_joined"); '
                f"else {flag_var}=(); fi"
            )
        raise AssertionError(spec.list_mode)
    if kind == "dict_str":
        arr_var = ensure_dict_decoded(name)
        if spec.dict_mode == "pairs":
            return f'{flag_var}=("${{{arr_var}[@]}}")'
        if spec.dict_mode == "equals":
            return (
                f"{flag_var}=(); "
                f"for ((_i=0; _i<${{#{arr_var}[@]}}; _i+=2)); do "
                f'{flag_var}+=({shlex.quote(flag)}"${{{arr_var}[_i]}}=${{{arr_var}[_i+1]}}"); '
                f"done"
            )
        raise AssertionError(spec.dict_mode)
    raise AssertionError(kind)
