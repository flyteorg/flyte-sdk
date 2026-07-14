"""Email notification example.

Sends an email when a run reaches a terminal phase. Both a plain-text body and
an HTML body are demonstrated; when html_body is provided the email is sent as
multipart (plain text + HTML).

Usage:
    export NOTIFICATION_EMAIL="oncall@example.com"
    python examples/notification/email.py

Template variables available in subject/body:
    {{.Run.Name}}, {{.Run.Project}}, {{.Run.Domain}}, {{.Phase}}, {{.Error}}
"""

import os

import flyte
from flyte import notify
from flyte.models import ActionPhase
from flyte.remote import Run

env = flyte.TaskEnvironment(
    name="notify_email",
    resources=flyte.Resources(memory="250Mi"),
)

NOTIFICATION_EMAIL = os.environ.get("NOTIFICATION_EMAIL", "oncall@example.com")


@env.task
def compute(x: int, y: int) -> int:
    return x + y


# HTML email on success.
email_success = notify.Email(
    on_phase=ActionPhase.SUCCEEDED,
    recipients=(NOTIFICATION_EMAIL,),
    subject="✅ Run {{.Run.Name}} succeeded",
    body="Run: {{.Run.Name}}\nProject/Domain: {{.Run.Project}}/{{.Run.Domain}}\nPhase: {{.Phase}}\n",
    html_body=(
        "<h2>✅ Run Succeeded</h2>"
        "<ul>"
        "<li><b>Run:</b> {{.Run.Name}}</li>"
        "<li><b>Project/Domain:</b> {{.Run.Project}}/{{.Run.Domain}}</li>"
        "<li><b>Phase:</b> {{.Phase}}</li>"
        "</ul>"
    ),
)

# Plain-text alert on failure, aborted, or timeout, with a cc.
email_failure = notify.Email(
    on_phase=(ActionPhase.FAILED, ActionPhase.ABORTED, ActionPhase.TIMED_OUT),
    recipients=(NOTIFICATION_EMAIL,),
    cc=(NOTIFICATION_EMAIL,),
    subject="🚨 ALERT: Run {{.Run.Name}} did not succeed ({{.Phase}})",
    body="Run: {{.Run.Name}}\nPhase: {{.Phase}}\nError: {{.Error}}\n",
)


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.with_runcontext(
        env_vars={"_U_USE_ACTIONS": "1", "NOTIFICATION_EMAIL": NOTIFICATION_EMAIL},
        notifications=(email_failure, email_success),
    ).run(compute, x=3, y=7)
    assert isinstance(run, Run)

    print(run.name)
    print(run.url)
    run.wait()
