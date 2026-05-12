"""
Code bundling — package user source into an artifact a remote worker can
download and run.

When a task runs on a remote cluster, its Python files need to get from the
developer's machine into the container. This module does that packaging.
It takes a source directory plus a few knobs, produces a versioned,
content-addressed archive, uploads it, and hands back a :class:`CodeBundle`
handle that the backend attaches to the task spec.

--------------------------------------------------------------------------
The bundle
--------------------------------------------------------------------------

A :class:`flyte.models.CodeBundle` is exactly one of:

* **tgz** — a gzipped tarball of source files. The common case.

  Archive members use POSIX-relative paths (so a file at
  ``<from_dir>/utils/helper.py`` becomes ``utils/helper.py`` in the tar),
  mtimes and uid/gid are stripped so the output is byte-deterministic, and
  the tarball filename is ``fast<digest>.tar.gz`` where ``<digest>`` is
  the md5 over the sorted list of ``(file-content, POSIX-relative-path)``
  pairs. The digest is the bundle's ``computed_version``, so two
  identical layouts in different parent directories produce the same
  bundle filename and version.

* **pkl** — a cloudpickle of the in-memory ``TaskTemplate`` or
  ``AppEnvironment``, gzipped. Used when there is no single user file to
  walk (Jupyter, REPL). The file is ``code_bundle.pkl.gz``; its version
  is the md5 of the gzipped bytes.

Either way the result is small, versioned, and self-contained.

--------------------------------------------------------------------------
What goes into a tgz bundle
--------------------------------------------------------------------------

Files are selected by ``copy_style``:

``"loaded_modules"`` (default)
    Walk ``sys.modules`` and keep only modules whose ``__file__`` sits
    under ``from_dir`` (and not under site-packages or the stdlib). The
    bundle contains only Python files the current process actually
    imported — unused siblings never ship.

``"all"``
    Walk ``from_dir`` on disk and include every file that is not excluded
    by ``.gitignore`` or the standard ignore list (``__pycache__``,
    ``*.pyc``, ``.venv``, ``.git``, …). Use when the task needs files it
    doesn't import — data, configs, templates, binary assets.

``"none"``
    Discover nothing from disk. Only meaningful when the caller also
    passes ``Environment.include`` — otherwise there is nothing to ship
    and the builder raises.

``"custom"``
    Don't discover; bundle an explicit caller-supplied list of relative
    paths. Used by the CLI when it has already determined the minimal
    file set.

On top of that, ``Environment.include`` contributes extra paths that are
unioned into whatever ``copy_style`` discovered. Includes are resolved
relative to the file where the environment was declared, deduplicated
across environments, and must live under ``from_dir``.

--------------------------------------------------------------------------
Public API
--------------------------------------------------------------------------

* :func:`build_code_bundle` — walk ``from_dir`` using ``copy_style``,
  tar+gzip, upload, return a tgz :class:`CodeBundle`. The main entry
  point.
* :func:`build_code_bundle_from_relative_paths` — bundle an explicit list
  of relative paths (no discovery). Used for the includes-only case and
  for ``copy_style="custom"``.
* :func:`build_pkl_bundle` — cloudpickle the task/app in memory and
  upload. Returns a pkl :class:`CodeBundle`.
* :func:`download_bundle` — the counterpart that runs on the worker:
  fetch the tgz/pkl and extract it into the task's working directory.
"""

from ._ignore import GitIgnore, IgnoreGroup, StandardIgnore
from ._utils import CopyFiles
from .bundle import (
    build_code_bundle,
    build_code_bundle_from_relative_paths,
    build_pkl_bundle,
    download_bundle,
)

__all__ = [
    "CopyFiles",
    "build_code_bundle",
    "build_code_bundle_from_relative_paths",
    "build_pkl_bundle",
    "default_ignores",
    "download_bundle",
]


default_ignores = [GitIgnore, StandardIgnore, IgnoreGroup]
