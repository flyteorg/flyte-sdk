# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=0.2.0b17",
#    "aioboto3>=11.3.0",
#    "asyncio",
# ]
# ///

import asyncio
import json
import os
from typing import List

import aioboto3

import flyte

env = flyte.TaskEnvironment(
    name="sqs_processor",
    resources=flyte.Resources(memory="500Mi", cpu=1),
    image=flyte.Image.from_uv_script(
        __file__,
        name="flyte",
    ).with_pip_packages("unionai-reuse>=0.1.3"),
    reusable=flyte.ReusePolicy(
        replicas=3,  # Minimum of 2 replicas to ensure no starvation of tasks
        idle_ttl=300,  # Idle time to keep the task environment alive
    ),
)
# Default Queue configuration (same as generator)
DEFAULT_QUEUE_ARN = os.getenv("QUEUE_ARN")


def get_queue_url_from_arn(queue_arn: str) -> str:
    """Convert an SQS ARN to a queue URL"""
    parts = queue_arn.split(":")
    region = parts[3]
    account = parts[4]
    queue_name = parts[5]

    return f"https://sqs.{region}.amazonaws.com/{account}/{queue_name}"


@env.task
async def process_message(message: dict) -> str:
    """
    Process a single message asynchronously

    Args:
        message: The SQS message body

    Returns:
        The extracted word from the message
    """
    # Parse the message body
    body = json.loads(message["Body"])

    # Extract the word
    word = body.get("word", "unknown")

    print(f"Task Processing message {body.get('message_id')}: {word}")

    # Simulate some processing time
    # await asyncio.sleep(1)

    return word


@env.task
async def main(queue_arn: str = DEFAULT_QUEUE_ARN, max_messages: int = 10) -> List[str]:
    """
    Main async function to receive and process messages from an SQS queue

    Args:
        queue_arn: The ARN of the SQS queue to read from
        max_messages: Maximum number of messages to process

    Returns:
        List of processed results
    """
    queue_url = get_queue_url_from_arn(queue_arn)

    # Create an async session
    session = aioboto3.Session(region_name="us-east-2")

    print(f"Processing up to {max_messages} messages from queue: {queue_url}")

    results = []
    tasks = []
    messages_received = 0

    # Use aioboto3's async context manager
    async with session.client("sqs") as sqs:
        # Process messages until we've received the maximum number
        while messages_received < max_messages:
            # Wait for a single message using native async
            response = await sqs.receive_message(
                QueueUrl=queue_url,
                AttributeNames=["All"],
                MaxNumberOfMessages=1,  # Get only one message at a time
                WaitTimeSeconds=20,  # Long polling timeout (max 20 seconds)
            )

            messages = response.get("Messages", [])
            if not messages:
                continue

            message = messages[0]  # We're only getting one message at a time
            messages_received += 1

            print(f"Message {messages_received}/{max_messages} received, dispatching processing task")

            # Create a task for processing
            process_task = asyncio.create_task(process_message(message))
            tasks.append(process_task)

            # Delete the message after we've started processing
            await sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=message["ReceiptHandle"])

    # Wait for all tasks to complete
    if tasks:
        print(f"Waiting for {len(tasks)} tasks to complete...")
        completed_tasks = await asyncio.gather(*tasks)
        results.extend(completed_tasks)

    print(f"Successfully processed {len(results)} messages")
    return results


if __name__ == "__main__":
    flyte.init_from_config("../../config.yaml")  # establish remote connection from within your script.
    run = flyte.run(main, queue_arn=DEFAULT_QUEUE_ARN, max_messages=10)  # run remotely inline and pass data.
    print(run.url)
