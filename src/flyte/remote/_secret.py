from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator, Literal, Union

import rich.repr
from flyteidl2.secret import definition_pb2, payload_pb2

from flyte._initialize import ensure_client, get_client, get_init_config
from flyte.remote._common import ToJSONMixin
from flyte.syncify import syncify

SecretTypes = Literal["regular", "image_pull"]


def _resolve_scope(cfg, cluster_pool: str | None, op: str) -> tuple[str, str]:
    """Return (project, domain) for a secret request, validating cluster_pool exclusivity."""
    if cluster_pool:
        if cfg.project or cfg.domain:
            raise ValueError(
                f"Project `{cfg.project}` or domain `{cfg.domain}` should not be set when "
                f"{op} a secret scoped to cluster pool `{cluster_pool}`."
            )
        return "", ""
    return cfg.project, cfg.domain


async def _secrets_service_for(cluster_pool: str | None, org: str | None):
    """Resolve the SecretService to dispatch to.

    For cluster-pool-scoped operations, ask the cluster-aware wrapper to pre-resolve
    the per-cluster client. cluster_pool is SDK-side routing metadata and is not
    carried in the secret request proto.
    """
    secrets_service = get_client().secrets_service
    if not cluster_pool:
        return secrets_service
    from flyte.remote._client.controlplane import ClusterAwareSecretService

    if not isinstance(secrets_service, ClusterAwareSecretService):
        raise RuntimeError(
            f"cluster_pool routing requires the cluster-aware secrets service; got {type(secrets_service).__name__}."
        )
    return await secrets_service.client_for_cluster_pool(org or "", cluster_pool)


@dataclass
class Secret(ToJSONMixin):
    pb2: definition_pb2.Secret

    @syncify
    @classmethod
    async def create(
        cls,
        name: str,
        value: Union[str, bytes],
        type: SecretTypes = "regular",
        cluster_pool: str | None = None,
    ):
        """
        Create a new secret.

        :param name: The name of the secret.
        :param value: The secret value as a string or bytes.
        :param type: Type of secret - either "regular" or "image_pull".
        :param cluster_pool: Optional cluster pool name. When set, the secret is scoped
            to the cluster pool and project/domain must not be set.
        """
        ensure_client()
        cfg = get_init_config()

        project, domain = _resolve_scope(cfg, cluster_pool, op="creating")

        if type == "regular":
            secret_type = definition_pb2.SecretType.SECRET_TYPE_GENERIC

        else:
            secret_type = definition_pb2.SecretType.SECRET_TYPE_IMAGE_PULL_SECRET
            if project or domain:
                raise ValueError(
                    f"Project `{project}` or domain `{domain}` should not be set when creating the image pull secret."
                )

        if isinstance(value, str):
            secret = definition_pb2.SecretSpec(
                type=secret_type,
                string_value=value,
            )
        else:
            secret = definition_pb2.SecretSpec(
                type=secret_type,
                binary_value=value,
            )
        request = payload_pb2.CreateSecretRequest(
            id=definition_pb2.SecretIdentifier(
                organization=cfg.org,
                project=project,
                domain=domain,
                name=name,
            ),
            secret_spec=secret,
        )
        svc = await _secrets_service_for(cluster_pool, cfg.org)
        await svc.create_secret(request=request)  # type: ignore

    @syncify
    @classmethod
    async def get(cls, name: str, cluster_pool: str | None = None) -> Secret:
        """
        Retrieve a secret by name.

        :param name: The name of the secret to retrieve.
        :param cluster_pool: Optional cluster pool name. When set, the secret is looked up
            in the cluster pool scope and project/domain must not be set.
        :return: A Secret object.
        """
        ensure_client()
        cfg = get_init_config()
        project, domain = _resolve_scope(cfg, cluster_pool, op="getting")
        request = payload_pb2.GetSecretRequest(
            id=definition_pb2.SecretIdentifier(
                organization=cfg.org,
                project=project,
                domain=domain,
                name=name,
            )
        )
        svc = await _secrets_service_for(cluster_pool, cfg.org)
        resp = await svc.get_secret(request=request)
        return Secret(pb2=resp.secret)

    @syncify
    @classmethod
    async def listall(cls, limit: int = 10, cluster_pool: str | None = None) -> AsyncIterator[Secret]:
        """
        List all secrets in the current project and domain.

        :param limit: Maximum number of secrets to return per page.
        :param cluster_pool: Optional cluster pool name. When set, secrets are listed
            from the cluster pool scope and project/domain must not be set.
        :return: An async iterator of Secret objects.
        """
        ensure_client()
        cfg = get_init_config()
        project, domain = _resolve_scope(cfg, cluster_pool, op="listing")
        svc = await _secrets_service_for(cluster_pool, cfg.org)
        per_cluster_tokens = None
        while True:
            request = payload_pb2.ListSecretsRequest(
                organization=cfg.org,
                project=project,
                domain=domain,
                per_cluster_tokens=per_cluster_tokens,
                limit=limit,
            )
            resp = await svc.list_secrets(request=request)  # type: ignore
            per_cluster_tokens = resp.per_cluster_tokens
            round_items = [v for _, v in per_cluster_tokens.items() if v]
            has_next = any(round_items)
            for r in resp.secrets:
                yield cls(r)
            if not has_next:
                break

    @syncify
    @classmethod
    async def delete(cls, name, cluster_pool: str | None = None):
        """
        Delete a secret by name.

        :param name: The name of the secret to delete.
        :param cluster_pool: Optional cluster pool name. When set, the secret is looked up
            in the cluster pool scope and project/domain must not be set.
        """
        ensure_client()
        cfg = get_init_config()
        project, domain = _resolve_scope(cfg, cluster_pool, op="deleting")

        request = payload_pb2.DeleteSecretRequest(
            id=definition_pb2.SecretIdentifier(
                organization=cfg.org,
                project=project,
                domain=domain,
                name=name,
            )
        )
        svc = await _secrets_service_for(cluster_pool, cfg.org)
        await svc.delete_secret(request=request)  # type: ignore

    @property
    def name(self) -> str:
        """
        Get the name of the secret.
        """
        return self.pb2.id.name

    @property
    def type(self) -> str:
        """
        Get the type of the secret as a string ("regular" or "image_pull").
        """
        if self.pb2.secret_metadata.type == definition_pb2.SecretType.SECRET_TYPE_GENERIC:
            return "regular"
        elif self.pb2.secret_metadata.type == definition_pb2.SecretType.SECRET_TYPE_IMAGE_PULL_SECRET:
            return "image_pull"
        raise ValueError("unknown type")

    def __rich_repr__(self) -> rich.repr.Result:
        """
        Rich representation of the Secret object for pretty printing.
        """
        yield "project", self.pb2.id.project or "-"
        yield "domain", self.pb2.id.domain or "-"
        yield "name", self.name
        yield "type", self.type
        yield "created_time", self.pb2.secret_metadata.created_time.ToDatetime().isoformat()
        yield "status", definition_pb2.OverallStatus.Name(self.pb2.secret_metadata.secret_status.overall_status)
        cluster_status = ", ".join(
            f"{s.cluster.name}: {definition_pb2.SecretPresenceStatus.Name(s.presence_status)}"
            for s in self.pb2.secret_metadata.secret_status.cluster_status
        )
        yield "cluster_status", cluster_status
