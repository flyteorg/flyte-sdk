# /// script
# requires-python = "==3.13"
# dependencies = [
#    "daytona",
# ]
# ///


"""
Example: Daytona Sandbox
"""

import os

from daytona import CreateSandboxFromSnapshotParams, Daytona, DaytonaConfig

# Define the configuration
api_key = os.getenv("DAYTONA_API_KEY")
if not api_key:
    raise ValueError("DAYTONA_API_KEY is not set")

config = DaytonaConfig(api_key=api_key)

# Initialize the Daytona client
daytona = Daytona(config)

params = CreateSandboxFromSnapshotParams(
    ephemeral=True,
    auto_stop_internal=1,
)

# Create the Sandbox instance
sandbox = daytona.create(params)

# Run the code securely inside the Sandbox
response = sandbox.process.code_run('print("Hello World from code!")')
if response.exit_code != 0:
    print(f"Error: {response.exit_code} {response.result}")
else:
    print(response.result)

sandbox.stop()
