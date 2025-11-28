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

dist-dirs:
	mkdir -p $(DIST_DIRS) $(CARGO_CACHE_DIR)

define BUILD_WHEELS_RECIPE
docker run --rm \
  -v $(PWD):/io \
  -v $(PWD)/docker_cargo_cache:/root/.cargo/registry \
  -v $(CLOUD_REPO):/cloud \
  wheel-builder:$(1) /bin/bash -c "\
    cd /io; \
    /opt/python/cp310-cp310/bin/maturin build --release --find-interpreter --out /io/dist/ \
  "
endef

# Build wheels for arm64 (depends on dist-dirs)
build-wheels-arm64: dist-dirs
	$(call BUILD_WHEELS_RECIPE,arm64)

# Build wheels for amd64 (depends on dist-dirs)
build-wheels-amd64: dist-dirs
	$(call BUILD_WHEELS_RECIPE,amd64)

build-wheels: build-wheels-arm64 build-wheels-amd64


# This is for Mac users, since the other targets won't build macos wheels (only local arch so probably arm64)
build-wheel-local: dist-dirs
	python -m build --wheel --outdir dist

