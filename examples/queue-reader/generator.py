# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=0.2.0b21",
#    "boto3",
# ]
# ///

"""
---
name: SQS Message Generator
description: Generates and sends 10 JSON messages to an AWS SQS queue
author: Union
tags:
  - aws
  - sqs
  - json
---
"""

import asyncio
import json
import os
import random

import boto3

# Queue configuration
QUEUE_ARN = os.getenv("QUEUE_ARN")


def get_queue_url_from_arn(queue_arn):
    """Convert an SQS ARN to a queue URL"""
    parts = queue_arn.split(":")
    region = parts[3]
    account = parts[4]
    queue_name = parts[5]

    return f"https://sqs.{region}.amazonaws.com/{account}/{queue_name}"


def generate_message(message_id):
    """Generate a simple JSON message with a random word and incrementing ID"""
    # Sample dictionary of words
    word_list = [
        "apple",
        "banana",
        "cherry",
        "diamond",
        "elephant",
        "flamingo",
        "giraffe",
        "hamburger",
        "igloo",
        "jungle",
        "kangaroo",
        "lemon",
        "mountain",
        "notebook",
        "orange",
        "penguin",
        "quasar",
        "rainbow",
        "strawberry",
        "tiger",
        "umbrella",
        "volcano",
        "waterfall",
        "xylophone",
        "zebra",
    ]

    return {"message_id": message_id, "word": random.choice(word_list)}


async def send_messages_to_queue(queue_url, count=10):
    """Send multiple messages to the configured SQS queue"""
    sqs = boto3.client("sqs", region_name="us-east-2")

    print(f"Sending {count} messages to queue: {queue_url}")

    for i in range(count):
        message_id = i + 1  # Incrementing message ID starting from 1
        message = generate_message(message_id)
        sqs.send_message(QueueUrl=queue_url, MessageBody=json.dumps(message))
        print(f"Sent message {message_id}/{count} with content: {message['word']}")
        await asyncio.sleep(0.1)

    print(f"Successfully sent {count} messages to the queue.")


async def main():
    """Main function to generate and send messages"""
    queue_url = get_queue_url_from_arn(QUEUE_ARN)
    await send_messages_to_queue(queue_url, count=10)


if __name__ == "__main__":
    asyncio.run(main())
