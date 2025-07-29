from typing import TYPE_CHECKING, Callable, Dict, Optional

if TYPE_CHECKING:
    pass


class Docstring(object):
    def __init__(self, docstring: Optional[str] = None, callable_: Optional[Callable] = None):
        import docstring_parser

        self._parsed_docstring: docstring_parser.Docstring

        if docstring is not None:
            self._parsed_docstring = docstring_parser.parse(docstring)
        elif callable_.__doc__ is not None:
            self._parsed_docstring = docstring_parser.parse(callable_.__doc__)

    @property
    def input_descriptions(self) -> Dict[str, Optional[str]]:
        return {p.arg_name: p.description for p in self._parsed_docstring.params}

    @property
    def output_descriptions(self) -> Dict[str, Optional[str]]:
        return {p.return_name: p.description for p in self._parsed_docstring.many_returns if p.return_name is not None}

    @property
    def short_description(self) -> Optional[str]:
        return self._parsed_docstring.short_description

    @property
    def long_description(self) -> Optional[str]:
        return self._parsed_docstring.long_description
