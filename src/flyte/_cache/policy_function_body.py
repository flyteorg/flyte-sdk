import ast
import hashlib
import inspect
import textwrap

from .cache import CachePolicy, VersionParameters


class FunctionBodyPolicy(CachePolicy):
    """
    A class that implements a versioning mechanism for functions by generating
    a SHA-256 hash of the function's source code combined with a salt.
    """

    def get_version(self, salt: str, params: VersionParameters) -> str:
        """
        This method generates a version string for a function by hashing the function's source code
        combined with a salt.

        :param salt: A string that is used to salt the hash.
        :param params: VersionParameters object that contains the parameters (e.g. function, ImageSpec, etc.) that are
                       used to generate the version.

        :return: A string that represents the version of the function.
        """
        if params.func is None:
            return ""

        source = inspect.getsource(params.func)
        dedented_source = textwrap.dedent(source)

        # Parse the source code into an Abstract Syntax Tree (AST)
        parsed_ast = ast.parse(dedented_source)

        # Convert the AST into a string representation
        ast_bytes = ast.dump(parsed_ast, include_attributes=False).encode("utf-8")

        # Combine the AST bytes with the salt (encoded into bytes)
        combined_data = ast_bytes + salt.encode("utf-8")

        # Return the SHA-256 hash of the combined data (AST + salt)
        return hashlib.sha256(combined_data).hexdigest()
