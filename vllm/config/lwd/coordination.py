# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cache coordination metadata only; no key reads, clients or server startup."""

from typing import Annotated, Any
from urllib.parse import urlsplit

from pydantic import ConfigDict, Field, StrictBool, field_validator
from pydantic.dataclasses import dataclass

from .topology import LwdPort, LwdString


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class LwdCoordinationConfig:
    enabled: StrictBool = False
    control_url: LwdString | None = None
    tenant_key_file: LwdString | None = None
    consumer_id: LwdString | None = None
    listen_host: LwdString = "0.0.0.0"
    listen_port: LwdPort = 8100
    instance_id: LwdString = "cloud-0"
    connect_timeout: Annotated[float, Field(strict=True, gt=0, allow_inf_nan=False)] = (
        10.0
    )

    @field_validator("control_url")
    @classmethod
    def validate_url(cls, value: str | None) -> str | None:
        if value is not None:
            url = urlsplit(value)
            if url.scheme not in ("http", "https") or not url.hostname:
                raise ValueError("control_url must be an HTTP(S) endpoint")
        return value

    @classmethod
    def from_dict(
        cls, raw: dict[str, Any], *, cloud_id: int | None = None
    ) -> "LwdCoordinationConfig":
        defaults = {} if cloud_id is None else {"instance_id": f"cloud-{cloud_id}"}
        return cls(**(defaults | raw))

    def validate_role(self, role: str) -> None:
        if (
            self.enabled
            and role == "edge"
            and (not self.control_url or not self.tenant_key_file)
        ):
            raise ValueError(
                "edge coordination requires control_url and tenant_key_file"
            )
