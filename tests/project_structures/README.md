# Test Project Structures for Module Loading

This directory contains test project structures for validating the module loader.

## Structure 1: Flat Directory
**Path:** `1_flat_directory/`
**Files:** `foo.py`, `bar.py`
**Expected behavior:**
- Load from: `1_flat_directory/`
- Module names: `foo`, `bar`
- No package prefix

## Structure 2: Namespace Package
**Path:** `2_namespace_package/my_pkg/tasks/`
**Files:** `foo.py`, `bar.py`
**Expected behavior:**
- Load from: `2_namespace_package/my_pkg/tasks/`
- Module names: `my_pkg.tasks.foo`, `my_pkg.tasks.bar`
- No `__init__.py` files (PEP 420 namespace package)
- Must run from `2_namespace_package/` as cwd

## Structure 3a: Regular Package (Classic Layout)
**Path:** `3a_regular_package/my_pkg/tasks/`
**Files:** `__init__.py`, `foo.py`, `bar.py`, `shared.py`
**Expected behavior:**
- Load from: `3a_regular_package/my_pkg/tasks/`
- Module names: `my_pkg.tasks.foo`, `my_pkg.tasks.bar`
- Tests absolute and relative imports
- Must run from `3a_regular_package/` as cwd

## Structure 3b: Regular Package (Src Layout)
**Path:** `3b_src_layout/src/my_pkg/tasks/`
**Files:** `__init__.py`, `foo.py`, `bar.py`, `shared.py`
**Expected behavior:**
- Load from: `3b_src_layout/src/my_pkg/tasks/`
- Module names: `my_pkg.tasks.foo`, `my_pkg.tasks.bar`
- Tests src/ layout pattern
- Must run from `3b_src_layout/` as cwd
- For editable install test: `pip install -e 3b_src_layout/`

## Usage in Tests

Each structure tests different import scenarios:
- Flat: Simple script loading
- Namespace: PEP 420 namespace packages
- Regular: Traditional packages with `__init__.py`
- Src layout: Modern src/ based projects
