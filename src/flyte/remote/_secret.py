from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator, Literal, Union

import rich.repr
from flyteidl2.secret import definition_pb2, payload_pb2

from flyte._initialize import ensure_client, get_client, get_init_config
from flyte.remote._common import ToJSONMixin
from flyte.syncify import syncify

SecretTypes = Literal["regular", "image_pull"]


@dataclass
class Secret(ToJSONMixin):
    pb2: definition_pb2.Secret

    @syncify
    @classmethod
    async def create(cls, name: str, value: Union[str, bytes], type: SecretTypes = "regular"):
        """
        Create a new secret.

        :param name: The name of the secret.
        :param value: The secret value as a string or bytes.
        :param type: Type of secret - either "regular" or "image_pull".
        """
        ensure_client()
        cfg = get_init_config()
        project = cfg.project
        domain = cfg.domain

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
        await get_client().secrets_service.CreateSecret(  # type: ignore
            request=payload_pb2.CreateSecretRequest(
                id=definition_pb2.SecretIdentifier(
                    organization=cfg.org,
                    project=project,
                    domain=domain,
                    name=name,
                ),
                secret_spec=secret,
            ),
        )

    @syncify
    @classmethod
    async def get(cls, name: str) -> Secret:
        """
        Retrieve a secret by name.

        :param name: The name of the secret to retrieve.
        :return: A Secret object.
        """
        ensure_client()
        cfg = get_init_config()
        resp = await get_client().secrets_service.GetSecret(
            request=payload_pb2.GetSecretRequest(
                id=definition_pb2.SecretIdentifier(
                    organization=cfg.org,
                    project=cfg.project,
                    domain=cfg.domain,
                    name=name,
                )
            )
        )
        return Secret(pb2=resp.secret)

    @syncify
    @classmethod
    async def listall(cls, limit: int = 10) -> AsyncIterator[Secret]:
        """
        List all secrets in the current project and domain.

        :param limit: Maximum number of secrets to return per page.
        :return: An async iterator of Secret objects.
        """
        ensure_client()
        cfg = get_init_config()
        per_cluster_tokens = None
        while True:
            resp = await get_client().secrets_service.ListSecrets(  # type: ignore
                request=payload_pb2.ListSecretsRequest(
                    organization=cfg.org,
                    project=cfg.project,
                    domain=cfg.domain,
                    per_cluster_tokens=per_cluster_tokens,
                    limit=limit,
                ),
            )
            per_cluster_tokens = resp.per_cluster_tokens
            round_items = [v for _, v in per_cluster_tokens.items() if v]
            has_next = any(round_items)
            for r in resp.secrets:
                yield cls(r)
            if not has_next:
                break

    @syncify
    @classmethod
    async def delete(cls, name):
        """
        Delete a secret by name.

        :param name: The name of the secret to delete.
        """
        ensure_client()
        cfg = get_init_config()
        await get_client().secrets_service.DeleteSecret(  # type: ignore
            request=payload_pb2.DeleteSecretRequest(
                id=definition_pb2.SecretIdentifier(
                    organization=cfg.org,
                    project=cfg.project,
                    domain=cfg.domain,
                    name=name,
                )
            )
        )

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
        yield (
            "cluster_status",
            {
                s.cluster.name: definition_pb2.SecretPresenceStatus.Name(s.presence_status)
                for s in self.pb2.secret_metadata.secret_status.cluster_status
            },
        )
