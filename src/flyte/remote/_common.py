import json
from typing import Literal, Tuple

from flyteidl2.common import list_pb2
from google.protobuf.json_format import MessageToDict, MessageToJson


class ToJSONMixin:
    """
    A mixin class that provides a method to convert an object to a JSON-serializable dictionary.
    """

    def to_dict(self) -> dict:
        """
        Convert the object to a JSON-serializable dictionary.

        Returns:
            dict: A dictionary representation of the object.
        """
        if hasattr(self, "pb2"):
            return MessageToDict(self.pb2)
        else:
            return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    def to_json(self) -> str:
        """
        Convert the object to a JSON string.

        Returns:
            str: A JSON string representation of the object.
        """
        return MessageToJson(self.pb2) if hasattr(self, "pb2") else json.dumps(self.to_dict())


def sorting(sort_by: Tuple[str, Literal["asc", "desc"]] | None = None) -> list_pb2.Sort:
    sort_by = sort_by or ("created_at", "asc")
    return list_pb2.Sort(
        key=sort_by[0],
        direction=(list_pb2.Sort.ASCENDING if sort_by[1] == "asc" else list_pb2.Sort.DESCENDING),
    )


def filtering(created_by_subject: str | None = None, *filters: list_pb2.Filter) -> list[list_pb2.Filter]:
    filter_list = list(filters) if filters else []
    if created_by_subject:
        filter_list.append(
            list_pb2.Filter(
                function=list_pb2.Filter.Function.EQUAL,
                field="created_by",
                values=[created_by_subject],
            ),
        )
    return filter_list
