def parse_description(description: str, max_length: int = 255):
    """
    Parse and truncate a description string to fit within a maximum length.

    If the description exceeds max_length, it will be truncated and suffixed with
    "...(tr.)" to indicate truncation.

    Args:
        description: The description string to parse.
        max_length: Maximum allowed length for the description. Defaults to 255.

    Returns:
        The parsed description string, truncated if necessary.
    """
    if len(description) <= max_length:
        return description
    if max_length >= 8:
        return description[: max_length - 8] + "...(tr.)"
    return description[:max_length]
