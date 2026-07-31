FLYTE_SYS_PATH = "_F_SYS_PATH"  # The paths that will be appended to sys.path at runtime

# Literal-metadata key carrying the id of the artifact a value came from, formatted
# "org/project/domain/name@version". Matches the v1 artifact service's tracking key
# (artifacts/pkg/lib.ArtifactKey) so provenance reads uniformly across both stacks.
ARTIFACT_TRACKER_KEY = "_ua"
