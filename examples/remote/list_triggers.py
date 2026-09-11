import flyte
from flyte.remote import Project, Trigger, Run, TimeFilter
from datetime import timezone, datetime
from flyte.models import ActionPhase

def list_projects():
    flyte.init_from_config("config.yaml")
    for p in Project.listall():
        print(" ", p.pb2.id, p.pb2.name)

    for p in Project.listall(archived=True):
        print(" ", p.pb2.id, p.pb2.name)

    # direct lookup; raises NotFound if it's really gone
    try:
        print("direct get:", Project.get("my-project"))
    except Exception as e:
        print("direct get failed:", e)


def list_triggers():
    flyte.init_from_config("config.yaml", project="my-project", domain="development")

    for t in Trigger.listall(limit=1000):
        if t.is_active:
            Trigger.update(name=t.name, task_name=t.task_name, active=False)
            print("deactivated", t.name, t.task_name)
        else:
            print("already off", t.name, t.task_name)


def list_runs():
    flyte.init_from_config("config.yaml", project="my-project", domain="development")
    dt = datetime(2026, 8, 15, 0, 0, 0, tzinfo=timezone.utc)
    tf = TimeFilter(after=dt)
    for r in Run.listall(limit=5000, project="my-project", domain="development", created_at=tf, in_phase=(ActionPhase.QUEUED,)):
        if not r.done() and r.name.startswith("u"):
            r.abort(reason="cleanup: stale trigger runs after actions cutover")
            print("aborted", r.name, r.phase)

if __name__ == "__main__":
    list_runs()