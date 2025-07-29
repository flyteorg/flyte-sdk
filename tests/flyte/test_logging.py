import mock
import pytest

from flyte._logging import log


@pytest.mark.asyncio
@mock.patch("flyte._logging.logger")
async def test_logging(mock_logger):
    logs = []

    def mock_log(*args, **kwargs):
        logs.append((args, kwargs))

    mock_logger.log.side_effect = mock_log
    mock_logger.getEffectiveLevel.return_value = 11

    # Cover all the ways it might be invoked
    @log
    async def test_func() -> str:
        return "Hello World"

    @log()
    async def test_func_empty() -> str:
        return "Hello World"

    @log(entry=False)
    async def test_func_exit_false() -> str:
        return "Hello World"

    await test_func()
    await test_func_empty()
    await test_func_exit_false()

    assert len(logs) == 5
