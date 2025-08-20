from click import Option, UsageError


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
