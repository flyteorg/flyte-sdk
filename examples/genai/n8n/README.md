# Flyte-hosted n8n app

In this example, we'll deploy a production-ready n8n app using Flyte.

## Prerequisites

In this example, we'll use a postgres database hosted on Supabase. You'll need to create a Supabase project and get the
database credentials: https://supabase.com

Thne create postgres database secrets in Flyte:

```bash
flyte create secret n8n_postgres_password --value <password>
flyte create secret n8n_encryption_key --value <encryption_key>
```

Install the example requirements:

```bash
uv pip install kubernetes
```

Then deploy the app:

```bash
python n8n_app.py
```

This will deploy two apps and one task environment:
1. `n8n-app` - The main n8n app. This contains a sidecar container that runs the n8n javascript and python task runners.
2. `flyte-n8n-webhook-app` - A webhook app that allows you to trigger Flyte tasks from n8n workflows.
3. `flyte-n8n-webhook-task` - A task environment that contains some toy Flyte task that can be triggered via the webhook.
