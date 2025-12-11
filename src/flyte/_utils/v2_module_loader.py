import sys
import importlib
from pathlib import Path
from contextlib import contextmanager


class ModuleLoader:
    """
    Loads Python modules from a user-specified directory using *standard* Python
    import semantics. Supports:

    1. Flat directory of .py files
    2. Namespace packages (PEP 420)
    3. Regular packages inside a repo (including src/ layout), not installed
    4. Regular packages installed in editable mode (pip install -e .)

    The logic:
      - Detect regular packages via __init__.py
      - Fall back to namespace detection based on cwd()
      - Derive module names using package_prefix + file structure
      - Modify sys.path only when necessary
      - Import using importlib.import_module
    """

    def __init__(self, load_dir: Path | str, verbose=False):
        load_dir = Path(load_dir)
        assert load_dir.exists()
        self.load_dir = load_dir.resolve()
        self.verbose = verbose

        # These are computed by compute_import_plan()
        self.root_dir_for_sys_path = None
        self.package_prefix = ""

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def looks_like_pkg_name(name: str) -> bool:
        return name.isidentifier() and name not in {"src", "code", "scripts"}

    def log(self, *args):
        if self.verbose:
            print("[ModuleLoader]", *args)

    # -------------------------------------------------------------------------
    # STEP 1 + STEP 2: Compute import plan
    # -------------------------------------------------------------------------

    def compute_import_plan(self):
        """
        Determines:
          - self.root_dir_for_sys_path
          - self.package_prefix
        according to the algorithm described in the PR.
        """

        load_dir = self.load_dir

        # ---- STEP 1: Look for regular package ancestors ----
        pkg_ancestors = []
        cur = load_dir

        while True:
            if (cur / "__init__.py").is_file():
                pkg_ancestors.append(cur)

            parent = cur.parent
            if parent == cur:
                break
            cur = parent

        if pkg_ancestors:
            # We are inside a regular package
            top_pkg_dir = pkg_ancestors[-1]
            root_dir = top_pkg_dir.parent
            rel_path = load_dir.relative_to(root_dir)
            parts = rel_path.parts

            self.package_prefix = ".".join(parts)
            top_pkg_name = parts[0]

            # Check if package is already importable (editable install etc.)
            if importlib.util.find_spec(top_pkg_name) is not None:
                self.root_dir_for_sys_path = None
                self.log("Regular package already importable (editable or installed).")
            else:
                self.root_dir_for_sys_path = root_dir
                self.log("Regular package NOT importable, will prepend root_dir:", root_dir)

            self.log("Detected regular package:",
                     f"top_pkg_name={top_pkg_name}",
                     f"package_prefix={self.package_prefix}")
            return

        # ---- STEP 2: Namespace package or flat directory ----

        cwd = Path.cwd()
        self.root_dir_for_sys_path = cwd  # your rule: use cwd as root for namespace cases
        rel_path = load_dir.relative_to(cwd)

        # Check if all components look like package names → namespace package
        if all(self.looks_like_pkg_name(p) for p in rel_path.parts):
            self.package_prefix = ".".join(rel_path.parts)
            self.log("Detected namespace package, prefix=", self.package_prefix)
        else:
            self.package_prefix = ""
            self.log("Detected flat directory (no prefix).")

    # -------------------------------------------------------------------------
    # Discover modules
    # -------------------------------------------------------------------------

    def discover_modules(self):
        """
        Returns a list of module names to import.
        """
        load_dir = self.load_dir
        prefix = self.package_prefix
        module_names = set()

        for path in load_dir.rglob("*.py"):
            if path.name == "__init__.py":
                rel = path.parent.relative_to(load_dir)
            else:
                rel = path.relative_to(load_dir).with_suffix("")

            parts = rel.parts
            if not parts:
                continue

            base = ".".join(parts)
            full_name = f"{prefix}.{base}" if prefix else base
            module_names.add(full_name)

        modules = sorted(module_names)
        self.log("Modules discovered:", modules)
        return modules

    # -------------------------------------------------------------------------
    # Context manager for modifying sys.path
    # -------------------------------------------------------------------------

    @contextmanager
    def maybe_modify_sys_path(self):
        path = self.root_dir_for_sys_path
        if path is None:
            yield
            return

        s = str(path)
        self.log("Temporarily adding to sys.path:", s)
        sys.path.insert(0, s)
        try:
            yield
        finally:
            self.log("Removing from sys.path:", s)
            try:
                sys.path.remove(s)
            except ValueError:
                pass

    # -------------------------------------------------------------------------
    # Main entry point
    # -------------------------------------------------------------------------

    def load_all(self):
        """
        Compute plan → discover module names → import modules.

        Returns {module_name: module_object}.
        """

        self.compute_import_plan()
        modules = self.discover_modules()

        imported = {}
        with self.maybe_modify_sys_path():
            for name in modules:
                self.log("Importing:", name)
                imported[name] = importlib.import_module(name)

        return imported
