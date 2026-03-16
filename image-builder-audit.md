# Image Builder: Current Behavior on `main`

Assume Python 3.13, flyte version `v2.0.7`, non-dev release build (dev mode uses hash-based tags instead of version-based preset tags).

| Case | Construction | Builds? (local) | Builds? (remote) | Final URI (local) | Final URI (remote) | Notes |
|------|-------------|-----------------|------------------|-------------------|--------------------|-------|
| 1 | `from_debian_base()` | Yes ¹ | Yes ¹ | `ghcr.io/flyteorg/flyte:py3.13-v2.0.7` | e.g. `376129846803...ecr.../union/dogfood:flyte-py3.13-v2.0.7` ² | |
| 2 | `from_debian_base().clone()` | Yes ¹ | Yes ¹ | `ghcr.io/flyteorg/flyte:<hash>` | e.g. `376129846803...ecr.../union/dogfood:flyte-<hash>` ² | |
| 3 | `from_debian_base().with_*()` | Yes ¹ | Yes ¹ | `ghcr.io/flyteorg/flyte:<hash>` | e.g. `376129846803...ecr.../union/dogfood:flyte-<hash>` ² | |
| 4 | `from_debian_base(name="my-image")` | Yes ¹ | Yes ¹ | `ghcr.io/flyteorg/my-image:<hash>` | e.g. `376129846803...ecr.../union/dogfood:my-image-<hash>` ² | |
| 5 | `from_debian_base(name="my-image", registry="myregistry.io")` | Yes ¹ | Yes ¹ | `myregistry.io/my-image:<hash>` | e.g. `myregistry.io/my-image:<hash>` ³ | |
| 6 | `from_debian_base().clone(name="my-image")` | Yes ¹ | Yes ¹ | `ghcr.io/flyteorg/my-image:<hash>` | e.g. `376129846803...ecr.../union/dogfood:my-image-<hash>` ² | |
| 7 | `from_debian_base().with_*().clone(name="my-image")` | Yes ¹ | Yes ¹ | `ghcr.io/flyteorg/my-image:<hash>` | e.g. `376129846803...ecr.../union/dogfood:my-image-<hash>` ² | |
| 8 | `from_base(uri)` | No | No | `<uri>` | `<uri>` | Returns caller-supplied URI as-is |
| 9 | `from_base(uri).clone()` | No | No | `<uri>` | `<uri>` | Returns caller-supplied URI as-is |
| 10 | `from_base(uri).clone(name="flyte-base")` | Yes ¹ | Yes ¹ | `flyte-base:<hash>` | e.g. `376129846803...ecr.../union/dogfood:flyte-base-<hash>` ² | |
| 11 | `from_base(uri).clone(name=..., extendable=True).with_*()` | Yes ¹ | Yes ¹ | `my-image:<hash>` | e.g. `376129846803...ecr.../union/dogfood:my-image-<hash>` ² | |
| 12 | `from_base(uri).clone(name=...).with_*()` | N/A | N/A | — | — | Construction-time `ValueError: Cannot add additional layers to a non-extendable image. Please create the image with extendable=True in the clone() call.` |
| 13 | `from_dockerfile(...)` | Yes ¹ | Error if build proceeds ⁴ | `myregistry.io/my-df:<hash>` | — | Remote: `ImageBuildError: Custom Dockerfile is not supported with remote image builder.You can use local image builder instead.` |
| 14 | `from_base(uri).clone(registry="myregistry.io")` | No | No | `<uri>` | `<uri>` | Registry silently ignored — `uri` property requires both `registry` and `name`; falls back to `base_image` |
| 15 | `from_dockerfile(...).with_*()` | N/A | N/A | — | — | Construction-time `ValueError: Cannot add additional layers to a non-extendable image...` ⁵ |
| 16 | `from_uv_script(script, name="my-uv-app")` | Yes ¹ | Yes ¹ | `ghcr.io/flyteorg/my-uv-app:<hash>` | e.g. `376129846803...ecr.../union/dogfood:my-uv-app-<hash>` ² | Delegates to `from_debian_base(name=..., install_flyte=False)` + `UVScript` layer |
| 17 | `from_base(uri).clone(extendable=True).clone(addl_layer=...)` | N/A | N/A | — | — | Construction-time `ValueError: Cannot add additional layer ... to an image without name. Please first clone().` |
| 18 | `from_base(uri).clone(name="my-image", extendable=True).clone(addl_layer=...)` | Yes ¹ | Yes ¹ | `my-image:<hash>` | e.g. `376129846803...ecr.../union/dogfood:my-image-<hash>` ² | Same as Case 11; `clone(addl_layer=...)` is what `with_*()` calls internally |

**¹ Build decision logic:** The image is built only when the first non-throwing existence checker returns not-found (`None`). If a checker returns a URI, the build is skipped. If all checkers raise exceptions, the image is **assumed to exist** and the build is skipped. The `force=True` option bypasses existence checks entirely; `dry_run=True` also bypasses checks.

**² Backend-determined URI:** The SDK submits a `target_image` of `<name>:<tag>` (when using the default `ghcr.io/flyteorg` registry) to the remote builder backend. The final fully-qualified URI is returned by the backend in action outputs. On the dogfood environment this is observed as `376129846803.dkr.ecr.us-east-2.amazonaws.com/union/dogfood:<name>-<tag>`, but this pattern is environment-specific, not code-derived.

**³ Custom registry remote:** When `registry` differs from the default (`ghcr.io/flyteorg`), the SDK submits `target_image` as `<registry>/<name>:<tag>` (preserving the custom registry). The final URI is still backend-determined.

**⁴ Dockerfile remote:** The remote builder rejects custom Dockerfiles only when a build is actually attempted. If the existence checker finds the image already exists, the build is skipped and no error is raised.

**⁵ Dockerfile layering:** `from_dockerfile()` sets `extendable=False`, so `.with_*()` hits the non-extendable guard first. Even if `extendable=True` is forced via an intermediate `.clone(extendable=True)`, a separate guard rejects it: `ValueError: Flyte current cannot add additional layers to a Dockerfile-based Image. Please amend the dockerfile directly.`
