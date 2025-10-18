# Use Node.js Alpine as base image for smaller size
FROM node:20-alpine

# Install bash and other required tools
RUN apk add --no-cache bash git curl python3 py3-pip

# Install Claude Code globally
RUN npm install -g @anthropic-ai/claude-code


# Ensure claude-code is in PATH
ENV PATH="/usr/local/bin:$PATH"

# Default command
CMD ["bash"]