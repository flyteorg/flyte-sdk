# Default target: show all available targets
.PHONY: help
help:
	@echo "Available targets:"
	@awk '/^[a-zA-Z0-9_\-]+:/ && !/^\./ {print "  " $$1}' $(MAKEFILE_LIST) | sed 's/://'

.DEFAULT_GOAL := help

.PHONY: fmt
fmt:
	uv run python -m ruff format
	uv run python -m ruff check . --fix

.PHONY: mypy
mypy:
	uv run python -m mypy src/ --config-file pyproject.toml

.PHONY: lint
lint-fix:
	uv run python -m ruff check .

.PHONY: dist
dist: clean
    # export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_FLYTE=0.0.1b0 to build with specific version
	uv run python -m build --wheel --installer uv

.PHONY: dist
dist-plugins: clean
	for plugin in plugins/*; do \
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

.PHONY: update-import-profile
update-import-profile:
	PYTHONPROFILEIMPORTTIME=1 python -c 'import flyte' 2&> import_profiles/flyte_importtime.txt

.PHONY: check-import-profile
check-import-profile:
	@echo "Checking import profile..."
	PYTHONPROFILEIMPORTTIME=1 python -c 'import flyte' 2&> updated_flyte_importtime.txt
	awk '{print $$NF}' import_profiles/flyte_importtime.txt > import_profiles/filtered_flyte_importtime.txt
	awk '{print $$NF}' updated_flyte_importtime.txt > updated_filtered_flyte_importtime.txt
	diff import_profiles/filtered_flyte_importtime.txt updated_filtered_flyte_importtime.txt || (echo "Import profile mismatch!" && exit 1)
	rm -f updated_flyte_importtime.txt updated_filtered_flyte_importtime.txt

.PHONY: copy-protos
copy-protos: export CLOUD_REPO_PATH ?= ../cloud
copy-protos:
	uv run ./maint_tools/copy_pb_python_from_cloud.py ${CLOUD_REPO_PATH}


.PHONY: unit_test
unit_test: ## Test the code with pytest
	@echo "🚀 Testing code: Running unit tests..."
	@uv run python -m pytest -k "not integration and not sandbox" tests


# Test plugins with pytest
# Usage:
# To run all plugin tests: `make unit_test_plugins`
# To run a specific plugin test: `make unit_test_plugins FLYTE_PLUGINS=openai`
.PHONY: unit_test_plugins
unit_test_plugins:
	@for plugin in $${FLYTE_PLUGINS:-plugins/*}; do \
		if [ -d "$$plugin/tests" ]; then \
			echo "🚀 Testing plugin: $$plugin..."; \
			cd "$$plugin" && uv run python -m pytest tests/ && cd ../../..; \
		fi \
	done


.PHONY: cli-docs-gen
cli-docs-gen: ## Generate CLI documentation
	@echo "📖 Generating CLI documentation..."
	@uv run flyte gen docs --type markdown
