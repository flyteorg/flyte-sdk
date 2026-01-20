"""
Test script to manually call the StateService Watch method.

This script follows the same initialization pattern as runtime.py:
1. Reads config from a YAML file (or uses default config locations)
2. Initializes the Flyte client using init_from_config
3. Creates a ControllerClient for state/queue services
4. Calls Watch on the state service

Usage:
    python test_watch_manual.py --config ./watch-config.yaml --run-name my-run --parent-action my-action
"""

import asyncio
import sys
from pathlib import Path

import click
from flyteidl2.common import identifier_pb2
from flyteidl2.workflow import state_service_pb2

from flyte import init_from_config
from flyte._initialize import get_init_config
from flyte._internal.controllers.remote._client import ControllerClient
from flyte._logging import logger


async def watch_actions(
    config_path: str | None,
    project: str | None,
    domain: str | None,
    run_name: str,
    parent_action_name: str,
    max_responses: int | None = None,
):
    """
    Watch for action updates using the StateService.Watch() method.

    Args:
        config_path: Path to Flyte config file (or None to use defaults)
        project: Project name (overrides config)
        domain: Domain name (overrides config)
        run_name: Run name to watch
        parent_action_name: Parent action name to watch
        max_responses: Optional limit on number of responses to receive
    """

    # Step 1: Initialize using config file (same as runtime.py does with init_in_cluster)
    print("=" * 80)
    print("Initializing Flyte Client from Config")
    print("=" * 80)

    if config_path:
        config_path_obj = Path(config_path).expanduser()
        if not config_path_obj.exists():
            logger.error(f"Config file not found: {config_path_obj}")
            sys.exit(1)
        logger.info(f"Loading config from: {config_path_obj}")
    else:
        logger.info("No config specified, will search default locations")

    # Initialize using config file (this sets up the global client)
    await init_from_config.aio(
        path_or_config=config_path,
        project=project,
        domain=domain,
    )

    # Get the initialized config
    init_config = get_init_config()

    # Use project/domain from config if not provided as arguments
    final_project = project or init_config.project
    final_domain = domain or init_config.domain
    org = init_config.org

    if not final_project or not final_domain:
        logger.error("Project and domain must be provided either in config or as arguments")
        sys.exit(1)

    logger.info(f"Project: {final_project}")
    logger.info(f"Domain: {final_domain}")
    logger.info(f"Run Name: {run_name}")
    logger.info(f"Parent Action: {parent_action_name}")
    print("=" * 80)

    # Step 2: Create ControllerClient (the smaller client with state_service)
    # We need to recreate it from the config since init_from_config creates ClientSet (the larger client)
    print("\nCreating ControllerClient for state service access...")

    # Read the platform config from the initialized config
    # Note: We need to get the endpoint/api_key from environment or config
    from flyte.config import auto as get_config

    cfg = get_config(config_path)

    # Create the ControllerClient using the same config as the main client
    if cfg.platform.endpoint:
        controller_client = await ControllerClient.for_endpoint(
            cfg.platform.endpoint,
            # insecure=cfg.platform.insecure,
            # insecure_skip_verify=cfg.platform.insecure_skip_verify,
            # ca_cert_file_path=cfg.platform.ca_cert_file_path,
            # client_id=cfg.platform.client_id,
            # client_credentials_secret=cfg.platform.client_credentials_secret,
            # auth_type=cfg.platform.auth_mode,
        )
    else:
        logger.error("No endpoint configured in config file")
        sys.exit(1)

    # Get the state service
    state_service = controller_client.state_service
    logger.info(f"State service type: {type(state_service).__name__}")

    # Step 3: Create the watch request
    print("\n" + "=" * 80)
    print("Starting Watch")
    print("=" * 80)
    print("Press Ctrl+C to stop watching\n")

    watch_request = state_service_pb2.WatchRequest(
        parent_action_id=identifier_pb2.ActionIdentifier(
            name=parent_action_name,
            run=identifier_pb2.RunIdentifier(
                project=final_project,
                domain=final_domain,
                name=run_name,
                org=org,
            ),
        ),
    )

    # Step 4: Call Watch and process responses
    try:
        watcher = state_service.Watch(
            watch_request,
            wait_for_ready=True,
        )

        response_count = 0
        async for response in watcher:
            response_count += 1
            print(f"\n--- Response #{response_count} ---")

            # Check for sentinel (control message)
            if response.HasField("control_message") and response.control_message.sentinel:
                print("✓ Received SENTINEL - cache is synced")
                logger.debug(f"  Control message: {response.control_message}")

            # Check for action update
            if response.HasField("action_update"):
                action_update = response.action_update
                from flyteidl2.common import phase_pb2

                phase_name = phase_pb2.ActionPhase.Name(action_update.phase)
                print(f"✓ Received ACTION UPDATE")
                print(f"  Action ID: {action_update.action_id.name}")
                print(f"  Phase: {phase_name}")
                print(f"  Output URI: {action_update.output_uri or 'N/A'}")

                if action_update.HasField("error"):
                    print(f"  Error: {action_update.error}")

            # Stop after max_responses if specified
            if max_responses and response_count >= max_responses:
                print(f"\n(Stopping after {max_responses} responses)")
                break

    except asyncio.CancelledError:
        print("\n✓ Watch cancelled by user (Ctrl+C)")
    except KeyboardInterrupt:
        print("\n✓ Watch interrupted by user (Ctrl+C)")
    except Exception as e:
        logger.error(f"Error during watch: {type(e).__name__}: {e}", exc_info=True)
    finally:
        print("\nClosing controller client...")
        await controller_client.close()
        print("✓ Done!")


@click.command()
@click.option(
    "--config",
    "-c",
    type=str,
    default=None,
    help="Path to Flyte config file (default: search standard locations)",
)
@click.option(
    "--project",
    "-p",
    type=str,
    default=None,
    help="Project name (overrides config)",
)
@click.option(
    "--domain",
    "-d",
    type=str,
    default=None,
    help="Domain name (overrides config)",
)
@click.option(
    "--run-name",
    "-r",
    type=str,
    required=True,
    help="Run name to watch",
)
@click.option(
    "--parent-action",
    "-a",
    type=str,
    required=True,
    help="Parent action name to watch",
)
@click.option(
    "--max-responses",
    "-m",
    type=int,
    default=None,
    help="Maximum number of responses to receive before stopping (optional)",
)
def main(
    config: str | None,
    project: str | None,
    domain: str | None,
    run_name: str,
    parent_action: str,
    max_responses: int | None,
):
    """
    Test script to manually watch action updates via StateService.Watch().

    This demonstrates the two different clients in the Flyte SDK:

    1. ClientSet (larger) - from flyte.remote._client.controlplane
       - Has admin, task, app, run, dataproxy services
       - Used for general Flyte operations

    2. ControllerClient (smaller) - from flyte._internal.controllers.remote._client
       - Has state_service and queue_service
       - Used by the Informer to watch action updates

    Examples:

        # Use config file
        python test_watch_manual.py -c ./watch-config.yaml -r my-run -a my-action

        # Override project/domain from command line
        python test_watch_manual.py -c ./watch-config.yaml -p myproject -d dev -r my-run -a my-action

        # Limit to first 10 responses
        python test_watch_manual.py -c ./watch-config.yaml -r my-run -a my-action -m 10
    """
    asyncio.run(
        watch_actions(
            config_path=config,
            project=project,
            domain=domain,
            run_name=run_name,
            parent_action_name=parent_action,
            max_responses=max_responses,
        )
    )


if __name__ == "__main__":
    main()
