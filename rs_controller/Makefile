# Makefile for building wheel-builder images and wheels using those images
# This is for making the main flyte base with_local_v2() image, which needs to
# be multi-arch to run on live clusters.


# Build both architecture-specific builder images
build-builders: build-arm64 build-amd64

# Build the arm64 builder image
build-arm64:
	docker buildx build --platform linux/arm64 --build-arg ARCH=aarch64 -f Dockerfile.maturinx -t wheel-builder:arm64 .

# Build the amd64 builder image (emulated on arm64 host)
build-amd64:
	docker buildx build --platform linux/amd64 --build-arg ARCH=x86_64 -f Dockerfile.maturinx -t wheel-builder:amd64 .

CARGO_CACHE_DIR := docker_cargo_cache
DIST_DIRS       := dist
# Derive version from git tags, convert to PEP 440 format
VERSION         := $(shell git describe --tags --always --dirty 2>/dev/null | sed -E 's/^v//; s/(-[0-9]+)-g([0-9a-f]+)/.dev\1+g\2/; s/\.dev-/\.dev/; s/-dirty/.dirty/')

dist-dirs:
	mkdir -p $(DIST_DIRS) $(CARGO_CACHE_DIR)

# Add the below to use local flyteidl2
# -v /Users/ytong/go/src/github.com/flyteorg/flyte/gen/rust:/Users/ytong/go/src/github.com/flyteorg/flyte/gen/rust

define BUILD_WHEELS_RECIPE
docker run --rm \
  -v $(PWD):/io \
  -v $(PWD)/docker_cargo_cache:/root/.cargo/registry \
  -v $(CLOUD_REPO):/cloud \
  wheel-builder:$(1) /bin/bash -c "\
    cd /io; \
    sed -i 's/^version = .*/version = \"$(VERSION)\"/' pyproject.toml; \
    /opt/python/cp310-cp310/bin/maturin build --release --find-interpreter --out /io/dist/ \
  "
endef

# Build wheels for arm64 (depends on dist-dirs)
build-wheels-arm64: dist-dirs
	$(call BUILD_WHEELS_RECIPE,arm64)

# Build wheels for amd64 (depends on dist-dirs)
build-wheels-amd64: dist-dirs
	$(call BUILD_WHEELS_RECIPE,amd64)

clean_dist:
	rm -rf $(DIST_DIRS)/*whl

build-wheels: clean_dist build-wheels-arm64 build-wheels-amd64 build-wheel-local

# This is for Mac users, since the other targets won't build macos wheels (only local arch so probably arm64)
build-wheel-local: dist-dirs
	@echo "Building version: $(VERSION)"
	sed -i.bak 's/^version = .*/version = "$(VERSION)"/' pyproject.toml && rm pyproject.toml.bak
	python -m build --wheel --outdir dist

check-nightly-toolchain:
	@rustup toolchain list | grep -q "nightly" || (echo "Error: nightly toolchain not found. Run 'rustup toolchain install nightly'" && exit 1)

.PHONY: fmt
fmt: check-nightly-toolchain
	cargo +nightly fmt --all

.PHONY: check-fmt
check-fmt: check-nightly-toolchain
	cargo +nightly fmt --check

.PHONY: lint
lint:
	cargo clippy --all-targets -- -D warnings

.PHONY: lint-fix
lint-fix:
	cargo clippy --all-targets --fix
