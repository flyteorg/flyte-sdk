import json
from dataclasses import dataclass
from datetime import datetime, timezone
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


@dataclass
class TimeFilter:
    """
    Filter for time-based fields (e.g. created_at, updated_at).

    :param after: Return only entries at or after this datetime (inclusive).
    :param before: Return only entries before this datetime (exclusive).
    """

    after: datetime | None = None
    before: datetime | None = None


def time_filtering(field_name: str, tf: TimeFilter) -> list[list_pb2.Filter]:
    """
    Build GREATER_THAN_OR_EQUAL / LESS_THAN Filter objects for a timestamp field.

    :param field_name: The name of the field to filter on (e.g. "created_at", "updated_at").
    :param tf: The TimeFilter specifying the after/before bounds.
    :return: A list of protobuf Filter objects.
    """
    filters = []
    if tf.after is not None:
        filters.append(
            list_pb2.Filter(
                function=list_pb2.Filter.Function.GREATER_THAN_OR_EQUAL,
                field=field_name,
                values=[tf.after.astimezone(timezone.utc).isoformat()],
            )
        )
    if tf.before is not None:
        filters.append(
            list_pb2.Filter(
                function=list_pb2.Filter.Function.LESS_THAN,
                field=field_name,
                values=[tf.before.astimezone(timezone.utc).isoformat()],
            )
        )
    return filters


def sorting(sort_by: Tuple[str, Literal["asc", "desc"]] | None = None) -> list_pb2.Sort:
    """
    Create a protobuf Sort object from a sorting tuple.

    :param sort_by: Tuple of (field_name, direction) for sorting, defaults to ("created_at", "asc").
    :return: A protobuf Sort object.
    """
    sort_by = sort_by or ("created_at", "asc")
    return list_pb2.Sort(
        key=sort_by[0],
        direction=(list_pb2.Sort.ASCENDING if sort_by[1] == "asc" else list_pb2.Sort.DESCENDING),
    )


def filtering(created_by_subject: str | None = None, *filters: list_pb2.Filter) -> list[list_pb2.Filter]:
    """
    Create a list of filter objects, optionally including a filter by creator subject.

    :param created_by_subject: Optional subject to filter by creator.
    :param filters: Additional filters to include.
    :return: A list of protobuf Filter objects.
    """
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
