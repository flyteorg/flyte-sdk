import json

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
