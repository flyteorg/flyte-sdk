from typing import Set

from click import Option, UsageError


class RequiresMixin:
    """Mixin that enforces that certain options must be present when this option is used."""

    def __init__(self, *args, **kwargs):
        self.requires: Set[str] = set(kwargs.pop("requires", []))
        self.requires_error_format = kwargs.pop(
            "requires_error_msg", "Illegal usage: option '{name}' requires '{required}' to be specified"
        )
        super().__init__(*args, **kwargs)

    def handle_parse_result(self, ctx, opts, args):
        self_present = self.name in opts and opts[self.name] is not None
        if self_present and self.requires:
            missing = [req for req in self.requires if req not in opts or opts[req] is None or opts[req] is False]
            if missing:
                raise UsageError(self.requires_error_format.format(name=self.name, required=", ".join(missing)))
        return super().handle_parse_result(ctx, opts, args)


class MutuallyExclusiveMixin:
    def __init__(self, *args, **kwargs):
        self.mutually_exclusive = set(kwargs.pop("mutually_exclusive", []))
        self.error_format = kwargs.pop(
            "error_msg", "Illegal usage: options '{name}' and '{invalid}' are mutually exclusive"
        )
        super().__init__(*args, **kwargs)

    def handle_parse_result(self, ctx, opts, args):
        self_present = self.name in opts and opts[self.name] is not None
        others_intersect = self.mutually_exclusive.intersection(opts)
        others_present = others_intersect and any(opts[value] is not None for value in others_intersect)

        if others_present:
            if self_present:
                raise UsageError(self.error_format.format(name=self.name, invalid=", ".join(self.mutually_exclusive)))
            else:
                self.prompt = None

        return super().handle_parse_result(ctx, opts, args)


# See https://stackoverflow.com/a/37491504/499285 and https://stackoverflow.com/a/44349292/499285
class MutuallyExclusiveOption(MutuallyExclusiveMixin, Option):
    def __init__(self, *args, **kwargs):
        mutually_exclusive = kwargs.get("mutually_exclusive", [])
        help = kwargs.get("help", "")
        if mutually_exclusive:
            kwargs["help"] = help + f" Mutually exclusive with {', '.join(mutually_exclusive)}."
        super().__init__(*args, **kwargs)


class RequiresOption(RequiresMixin, Option):
    """Option that requires other options to be present."""

    def __init__(self, *args, **kwargs):
        requires = kwargs.get("requires", [])
        help = kwargs.get("help", "")
        if requires:
            kwargs["help"] = help + f" Requires {', '.join(requires)}."
        super().__init__(*args, **kwargs)


class DependentOption(RequiresMixin, MutuallyExclusiveMixin, Option):
    """Option that supports both 'requires' and 'mutually_exclusive' constraints."""

    def __init__(self, *args, **kwargs):
        requires = kwargs.get("requires", [])
        mutually_exclusive = kwargs.get("mutually_exclusive", [])
        help = kwargs.get("help", "")
        if mutually_exclusive:
            help = help + f" Mutually exclusive with {', '.join(mutually_exclusive)}."
        if requires:
            help = help + f" Requires {', '.join(requires)}."
        kwargs["help"] = help
        super().__init__(*args, **kwargs)
