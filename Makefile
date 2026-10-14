# Default registry for image builds
REGISTRY ?= ghcr.io/flyteorg
# Default name for connector image
CONNECTOR_IMAGE_NAME ?= flyte-connector

# Default target: show all available targets
.PHONY: help
help:
	@echo "Available targets:"
	@awk '/^[a-zA-Z0-9_\-]+:/ && !/^\./ {print "  " $$1}' $(MAKEFILE_LIST) | sed 's/://'

.DEFAULT_GOAL := help

.PHONY: prek-install
prek-install:
	curl --proto '=https' --tlsv1.2 -LsSf https://github.com/j178/prek/releases/download/v0.3.5/prek-installer.sh | sh

.PHONY: fmt
fmt:
	uv run python -m ruff format
	uv run python -m ruff check . --fix

.PHONY: mypy
mypy:
	uv run python -m mypy --config-file pyproject.toml \
		src/ \
		examples/

.PHONY: ty
ty:
	uv run ty check \
		src/ \
		examples/

.PHONY: uvlock
uvlock:
	bash maint_tools/uvlock.sh

.PHONY: lint
lint-fix:
	uv run python -m ruff check . --fix

.PHONY: dist
dist: clean
    # export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_FLYTE=0.0.1b0 to build with specific version
	uv run python -m build --wheel --installer uv

.PHONY: clean-plugins
clean-plugins:
	rm -f dist/flyteplugins_*.whl
	rm -rf plugins/**/dist/
	rm -rf plugins/**/build/

.PHONY: dist-plugins
dist-plugins: clean-plugins
    # set FLYTE_PLUGIN_DIST to the directory of a specific plugin to build
	for plugin in $${FLYTE_PLUGIN_DIST:-plugins/*}; do \
		if [ -d "$$plugin" ]; then \
			uv run python -m build --wheel --installer uv --outdir ./dist "$$plugin"; \
		fi \
	done

dist-all: dist dist-plugins

.PHONY: clean
clean: 
	rm -rf dist/
	rm -rf plugins/**/dist/
	rm -rf build/
	rm -rf plugins/**/build/
	rm -rf src/flyte.egg-info

# The import profiles pin which modules the startup-critical paths (`import flyte`, the `a0` task
# runtime, `flyte run --local`) are allowed to load. See maint_tools/import_profile.py.
# They run in a throwaway environment holding only Flyte and its required dependencies at the
# locked versions, so the result does not depend on what else is installed locally.
IMPORT_PROFILE = uv run --quiet --isolated --no-default-groups --frozen --python 3.13 python maint_tools/import_profile.py

.PHONY: update-import-profile
update-import-profile:
	$(IMPORT_PROFILE) update

.PHONY: check-import-profile
check-import-profile:
	$(IMPORT_PROFILE) check

.PHONY: unit_test
unit_test: ## Test the code with pytest
	@echo "🚀 Testing code: Running unit tests..."
	@uv run python -m pytest -k "not integration and not sandbox" tests


# Test plugins with pytest
# Usage:
# To run all plugin tests: `make unit_test_plugins`
# To run a specific plugin test: `FLYTE_PLUGIN=plugins/openai make unit_test_plugins`
.PHONY: unit_test_plugins
unit_test_plugins:
	@for plugin in $${FLYTE_PLUGIN:-plugins/*}; do \
		if [ -d "$$plugin/tests" ]; then \
			echo "🚀 Testing plugin: $$plugin..."; \
			( cd "$$plugin" && uv run python -m pytest tests/ ); \
		fi \
	done

.PHONY: dev-rs-dist
dev-rs-dist:
	cd rs_controller && $(MAKE) build-wheels
	$(MAKE) dist
	uv run python maint_tools/build_default_image.py --registry $(REGISTRY) --name $(CONNECTOR_IMAGE_NAME)
	uv pip install --find-links ./rs_controller/dist --no-index --force-reinstall --no-deps flyte_controller_base

.PHONY: cli-docs-gen
cli-docs-gen: ## Generate CLI documentation
	@echo "📖 Generating CLI documentation..."
	@uv run flyte gen docs --type markdown
	@uv run flyte gen docs --type json | uv run python maint_tools/check_cli_json.py

.PHONY: check-docstrings
check-docstrings: ## Reject reStructuredText and NumPy sections in docstrings
	@uv run python maint_tools/check_docstring_style.py
