from .cache import CachePolicy
from .policy_function_body import FunctionBodyPolicy


def get_default_policies() -> list[CachePolicy]:
    """
    Get default cache policies.
    """
    return [FunctionBodyPolicy()]
