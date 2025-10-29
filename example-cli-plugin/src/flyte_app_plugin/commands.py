"""Commands for managing Flyte apps.

This module defines CLI commands for the 'app' entity:
- flyte get app
- flyte create app
- flyte delete app
"""

import rich_click as click


@click.command()
@click.argument("name", required=False)
@click.option("--project", "-p", help="Project to list apps from")
@click.option("--domain", "-d", default="development", help="Domain to list apps from")
@click.pass_obj
def get_app(obj, name: str | None, project: str | None, domain: str):
    """
    Get apps from Flyte.

    If NAME is provided, retrieves a specific app. Otherwise, lists all apps.

    Examples:

        # List all apps
        $ flyte get app

        # Get a specific app
        $ flyte get app my-app

        # Get apps from a specific project
        $ flyte get app --project my-project
    """
    from flyte.cli._common import CLIConfig

    cli_config: CLIConfig = obj

    # Access global CLI configuration
    endpoint = cli_config.endpoint or "local"
    org = cli_config.org or "default"

    if name:
        click.echo(f"🔍 Getting app '{name}' from {endpoint}")
        click.echo(f"   Project: {project or 'all'}")
        click.echo(f"   Domain: {domain}")
        click.echo(f"   Organization: {org}")
        click.echo()
        click.echo("📦 App Details:")
        click.echo(f"   Name: {name}")
        click.echo(f"   Status: Running")
        click.echo(f"   Version: 1.0.0")
        click.echo(f"   Endpoint: https://{name}.{endpoint}")
    else:
        click.echo(f"📋 Listing apps from {endpoint}")
        click.echo(f"   Project: {project or 'all'}")
        click.echo(f"   Domain: {domain}")
        click.echo(f"   Organization: {org}")
        click.echo()
        click.echo("Available apps:")
        # This would normally query the Flyte backend
        apps = ["demo-app", "prod-app", "test-app"]
        for app in apps:
            click.echo(f"  • {app}")


@click.command()
@click.argument("name")
@click.option("--project", "-p", required=True, help="Project to create app in")
@click.option("--domain", "-d", default="development", help="Domain to create app in")
@click.option("--image", help="Container image for the app")
@click.option("--replicas", type=int, default=1, help="Number of replicas")
@click.option("--description", help="Description of the app")
@click.pass_obj
def create_app(
    obj,
    name: str,
    project: str,
    domain: str,
    image: str | None,
    replicas: int,
    description: str | None,
):
    """
    Create a new app in Flyte.

    Creates a new app entity that can be deployed and managed.

    Examples:

        # Create a basic app
        $ flyte create app my-app --project my-project

        # Create an app with custom configuration
        $ flyte create app my-app --project my-project \\
            --image myregistry/myapp:latest \\
            --replicas 3 \\
            --description "My production app"
    """
    from flyte.cli._common import CLIConfig

    cli_config: CLIConfig = obj

    endpoint = cli_config.endpoint or "local"
    org = cli_config.org or "default"

    click.echo(f"🚀 Creating app '{name}' on {endpoint}")
    click.echo(f"   Project: {project}")
    click.echo(f"   Domain: {domain}")
    click.echo(f"   Organization: {org}")

    if image:
        click.echo(f"   Image: {image}")

    click.echo(f"   Replicas: {replicas}")

    if description:
        click.echo(f"   Description: {description}")

    # Simulate app creation
    click.echo()
    click.echo("✨ App configuration:")
    click.echo(f"   Name: {name}")
    click.echo(f"   Status: Creating...")
    click.echo()
    click.echo("✅ App created successfully!")
    click.echo(f"   URL: https://{name}.{endpoint}/{project}/{domain}")
    click.echo()
    click.echo("💡 Next steps:")
    click.echo(f"   • Deploy your app: flyte deploy app {name}")
    click.echo(f"   • Check status: flyte get app {name}")


@click.command()
@click.argument("name")
@click.option("--project", "-p", help="Project containing the app")
@click.option("--domain", "-d", default="development", help="Domain containing the app")
@click.option("--force", "-f", is_flag=True, help="Force deletion without confirmation")
@click.pass_obj
def delete_app(obj, name: str, project: str | None, domain: str, force: bool):
    """
    Delete an app from Flyte.

    Removes an app and all its associated resources.

    Examples:

        # Delete an app (with confirmation)
        $ flyte delete app my-app

        # Force delete without confirmation
        $ flyte delete app my-app --force

        # Delete from specific project
        $ flyte delete app my-app --project my-project
    """
    from flyte.cli._common import CLIConfig

    cli_config: CLIConfig = obj

    endpoint = cli_config.endpoint or "local"
    org = cli_config.org or "default"

    click.echo(f"🗑️  Deleting app '{name}' from {endpoint}")
    click.echo(f"   Project: {project or 'default'}")
    click.echo(f"   Domain: {domain}")
    click.echo(f"   Organization: {org}")
    click.echo()

    if not force:
        # Ask for confirmation
        click.confirm(
            f"Are you sure you want to delete app '{name}'? This cannot be undone.",
            abort=True,
        )

    # Simulate deletion
    click.echo("🔄 Deleting app resources...")
    click.echo("   • Stopping replicas...")
    click.echo("   • Removing configurations...")
    click.echo("   • Cleaning up resources...")
    click.echo()
    click.echo(f"✅ App '{name}' deleted successfully!")
