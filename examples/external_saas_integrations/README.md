# External SaaS integrations

Recipes for driving GitHub, Slack, Linear, ClickUp, and Jira from Flyte.

There is deliberately **no Flyte client plugin** for these products. Each vendor
already ships (or the community maintains) a Python client that is tested
against the live API by people who get deprecation notices first, and a task is
just a function — so calling `PyGithub` or `slack_sdk` from a task needs nothing
in between. A wrapper here would only add a surface to keep in sync with someone
else's release calendar.

What *is* Flyte's job, and what these examples use from it:

- **`flyte.extras.webhooks`** — ships with flyte: one app that authenticates an inbound
  delivery with the product's own scheme, normalizes it into a single event
  model, and launches a run once per event key with `run_once`.
- **`flyteplugins-webhooks-<product>`** — one small package per product,
  contributing just its verification and parsing.
- **`flyte.new_condition`** — park a run on a human decision, with a typed
  payload coming back. `flyteplugins.github.review_pr` wraps this into a PR
  review gate, which is the one place a plugin beats calling the vendor SDK:
  the condition is Flyte's, not GitHub's.

## The examples

| File | What it shows | Client |
| --- | --- | --- |
| `webhook_receiver.py` | One app receiving from all five products, launching a task per event | `flyteplugins-webhooks-*` |
| `github_pr_review_gate.py` | Human-gated merge, on `flyteplugins.github.review_pr` | plugin + `PyGithub` |
| `github_triage_pr.py` | Label, comment, and report a check run | `PyGithub` |
| `slack_notify.py` | Post, thread, react, answer a mention | `slack_sdk` |
| `linear_triage_issue.py` | Query a backlog and comment, over GraphQL | `gql` |
| `clickup_manage_ticket.py` | Open and close tickets, with a status pre-check | `httpx` |
| `jira_manage_ticket.py` | Open, transition, and search issues | `jira` |

Linear and ClickUp ship no official Python SDK. Linear's API is a single GraphQL
endpoint, so `gql` is the maintained client; ClickUp's is a handful of REST
calls, so `httpx` directly beats a thin third-party wrapper.

## Putting it together

The receiver and the tasks are separate on purpose: the app authenticates and
dispatches, the tasks do the work and can be run, tested, and retried on their
own.

```bash
# 1. deploy the tasks the receiver will launch
flyte deploy examples/external_saas_integrations/github_triage_pr.py env
flyte deploy examples/external_saas_integrations/slack_notify.py env

# 2. run one directly, to confirm credentials work before any webhook is involved
flyte run examples/external_saas_integrations/github_triage_pr.py triage_pr \
    --repo <owner>/<repo> --number <pr>

# 3. deploy the receiver and point each provider at the URL its dashboard shows
python examples/external_saas_integrations/webhook_receiver.py
```

Task names are qualified by their environment when deployed — `triage_pr` in
`github_triage_pr.py` becomes `github-triage.triage_pr`, which is what the
receiver looks up. That qualifier is also what keeps the two `triage_issue`
tasks here (Linear's and Jira's) from colliding.

`plugins/webhooks/README.md` has the full end-to-end testing guide, including
how to wire up each provider and what to check at every step.
