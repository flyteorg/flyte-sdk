from flyteplugins.bio import env as bio_env
from flyteplugins.bio.bedtools import env as bedtools_env


def test_bio_env_aggregates_tool_family_envs():
    assert bio_env.name == "bio"
    assert bio_env.depends_on == [bedtools_env]
