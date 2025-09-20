
@dataclass
class Card(object):
    text: str
    card_type: CardType = field(default=CardType.UNKNOWN, init=False)

    def serialize_to_string(self, ctx: FlyteContext, variable_name: str) -> typing.Tuple[str, str]:
        # only upload if we're running a real task execution
        if ctx.execution_state and ctx.execution_state.mode == ExecutionState.Mode.TASK_EXECUTION:
            if ctx.user_space_params and ctx.user_space_params.output_metadata_prefix:
                output_location = ctx.user_space_params.output_metadata_prefix
                reader = StringIO(self.text)
                to_path = ctx.file_access.put_raw_data(
                    reader, upload_prefix=output_location, file_name=f"card_{variable_name}", skip_raw_data_prefix=True
                )
                logger.debug(
                    f"Artifact card detected for {variable_name}, attempting to upload under {output_location}"
                )
                logger.info(f"Card uploaded to {to_path} for {variable_name}")

                c = artifacts_pb2.Card(
                    uri=to_path,
                    type=self.card_type,
                )

                s = c.SerializeToString()
                encoded = b64encode(s).decode("utf-8")

                return _CARD_METADATA_KEY, encoded

        logger.debug(f"Artifact card found on {variable_name}, but not uploading, starts with {self.text[0:100]}")
        return _CARD_METADATA_KEY, self.text[0:100]


@dataclass
class DataCard(Card):
    """
    :param text: DataCard contents.
    :param card_type:
    """

    card_type: CardType = CardType.DATASET

    @classmethod
    def from_obj(cls, card_obj: typing.Any) -> Card:
        return cls(text=str(card_obj))


@dataclass
class ModelCard(Card):
    """
    :param text: ModelCard contents.
    :param card_type:
    """

    card_type: CardType = CardType.MODEL

    @classmethod
    def from_obj(cls, card_obj: typing.Any) -> Card:
        return cls(text=str(card_obj))
