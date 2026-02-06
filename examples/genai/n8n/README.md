# Flyte-hosted n8n app

In this example, we'll deploy a production-ready n8n app using Flyte.

Create postgres database credentials

```bash
flyte create secret n8n_postgres_password --value <password>
flyte create secret n8n_postgres_host --value <host>
```
