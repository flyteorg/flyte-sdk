FROM n8nio/runners:2.4.8
USER root
COPY n8n-task-runners.json /etc/n8n-task-runners.json
RUN cd /opt/runners/task-runner-javascript && pnpm add moment uuid
RUN cd /opt/runners/task-runner-python && uv pip install numpy pandas
USER runner
