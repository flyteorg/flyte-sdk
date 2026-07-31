FLYTE_SYS_PATH = "_F_SYS_PATH"  # The paths that will be appended to sys.path at runtime

# Literal-metadata key carrying the id of the artifact a value came from, formatted
# "org/project/domain/name@version". Matches the v1 artifact service's tracking key
# (artifacts/pkg/lib.ArtifactKey) so provenance reads uniformly across both stacks.
ARTIFACT_TRACKER_KEY = "_ua"

# Literal-metadata key carrying the artifact metadata a task attached to an output via
# flyte.artifacts.new(...): compact JSON {name, version?, description?, data?, card?}.
# The leaseworker reads this key to extract generated artifacts when the task declared
# produces_artifacts. The Go-side reader (leaseworker/artifacts.go) must stay in sync.
ARTIFACT_PRODUCED_KEY = "_uap"
