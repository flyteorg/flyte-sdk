#!/bin/bash

# Set custom environment variables
export AWS_ACCESS_KEY_ID=test123
export AWS_SECRET_ACCESS_KEY=test
export AWS_ENDPOINT_URL=http://localhost:4566
export FLYTE_AWS_ACCESS_KEY_ID=test
export FLYTE_AWS_SECRET_ACCESS_KEY=test
export FLYTE_AWS_ENDPOINT=http://localhost:4566
export _U_EP_OVERRIDE=dns:///localhost:8090
# Set a prompt prefix (e.g., [venv]) and preserve existing PS1
export PS1="[faws] $PS1"

# Optional: message
echo "Entered [venv] environment. Type 'exit' to leave."
