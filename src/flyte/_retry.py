from dataclasses import dataclass
from datetime import timedelta
from typing import Union


@dataclass
class RetryStrategy:
    """
    Retry strategy for the task or task environment. Retry strategy is optional or can be a simple number of retries.

    Example:
    - This will retry the task 5 times.
    ```
    @task(retries=5)
    def my_task():
        pass
    ```
    - This will retry the task 5 times with a maximum backoff of 10 seconds and a backoff factor of 2.
    ```
    @task(retries=RetryStrategy(count=5, max_backoff=10, backoff=2))
    def my_task():
        pass
    ```

    :param count: The number of retries.
    :param backoff: The maximum backoff time for retries. This can be a float or a timedelta.
    :param backoff: The backoff exponential factor. This can be an integer or a float.
    """

    count: int
    backoff: Union[float, timedelta, None] = None
    backoff_factor: Union[int, float, None] = None
