from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from typing import ClassVar as _ClassVar

DESCRIPTOR: _descriptor.FileDescriptor

class Deployment(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []
    UNSPECIFIED: _ClassVar[Deployment]
    BYOC: _ClassVar[Deployment]
    SERVERLESS: _ClassVar[Deployment]
UNSPECIFIED: Deployment
BYOC: Deployment
SERVERLESS: Deployment
