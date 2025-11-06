#!/usr/bin/env python3
"""
DNS Diagnostic Script for Flyte gRPC Channel Creation

This script continuously attempts to create Flyte gRPC channels to reproduce
intermittent DNS resolution errors (socket.gaierror).

Usage:
    python dns_diagnostic.py

The script will read from ~/.flyte/demo.yaml and continuously test connection creation.
"""

import asyncio
import os
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import grpc
from flyteidl.admin import project_pb2

import flyte.config as config
from flyte._logging import initialize_logger, logger
from flyte.remote._client.controlplane import ClientSet


def get_dns_info():
    """Collect current DNS configuration information."""
    dns_info = {
        "timestamp": datetime.now().isoformat(),
        "hostname": socket.gethostname(),
        "fqdn": socket.getfqdn(),
    }

    # Try to get DNS servers (macOS)
    try:
        result = subprocess.run(
            ["scutil", "--dns"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        # Extract just nameserver lines
        nameservers = [
            line.strip()
            for line in result.stdout.split("\n")
            if "nameserver" in line.lower()
        ]
        dns_info["nameservers"] = nameservers[:10]  # Limit output
    except Exception as e:
        dns_info["nameservers"] = f"Error: {e}"

    # Check /etc/resolv.conf (if accessible)
    try:
        with open("/etc/resolv.conf", "r") as f:
            resolv_conf = [line.strip() for line in f if line.strip() and not line.startswith("#")]
            dns_info["resolv_conf"] = resolv_conf
    except Exception as e:
        dns_info["resolv_conf"] = f"Error: {e}"

    return dns_info


def test_dns_resolution(hostname: str):
    """Test DNS resolution at Python level and collect timing information."""
    result = {
        "hostname": hostname,
        "success": False,
        "duration_ms": None,
        "addresses": None,
        "error": None,
    }

    try:
        start = time.time()
        addr_info = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
        duration_ms = (time.time() - start) * 1000

        result["success"] = True
        result["duration_ms"] = round(duration_ms, 2)
        result["addresses"] = [addr[4][0] for addr in addr_info]
    except socket.gaierror as e:
        result["error"] = {
            "type": "socket.gaierror",
            "errno": e.errno,
            "message": str(e),
            "args": e.args,
        }
    except Exception as e:
        result["error"] = {
            "type": type(e).__name__,
            "message": str(e),
        }

    return result


def extract_hostname(endpoint: str) -> str:
    """Extract hostname from endpoint URL."""
    from flyte._utils.org_discovery import hostname_from_url

    hostname = hostname_from_url(endpoint)
    # Remove port if present
    if ":" in hostname:
        hostname = hostname.rsplit(":", 1)[0]
    return hostname


async def test_channel_creation(cfg: config.Config, iteration: int):
    """
    Attempt to create a Flyte gRPC channel similar to flyte.init().

    Returns (success: bool, info: dict)
    """
    result = {
        "iteration": iteration,
        "timestamp": datetime.now().isoformat(),
        "success": False,
        "error": None,
        "dns_test": None,
        "dns_info": None,
        "duration_ms": None,
    }

    endpoint = cfg.platform.endpoint
    if not endpoint:
        result["error"] = "No endpoint configured"
        return False, result

    # Extract hostname for DNS testing
    hostname = extract_hostname(endpoint)

    # First, test DNS resolution at Python level
    result["dns_test"] = test_dns_resolution(hostname)

    try:
        start = time.time()

        # Create client exactly like flyte.init does
        client = await ClientSet.for_endpoint(
            endpoint,
            insecure=cfg.platform.insecure,
            insecure_skip_verify=cfg.platform.insecure_skip_verify,
            auth_type=cfg.platform.auth_mode,
            ca_cert_file_path=cfg.platform.ca_cert_file_path,
            command=cfg.platform.command,
            proxy_command=cfg.platform.proxy_command,
            client_id=cfg.platform.client_id,
            client_credentials_secret=cfg.platform.client_credentials_secret,
            rpc_retries=cfg.platform.rpc_retries,
            http_proxy_url=cfg.platform.http_proxy_url,
        )

        # Make an actual RPC call to force DNS resolution
        # gRPC channels are lazy - they don't resolve DNS until first use
        request = project_pb2.ProjectListRequest(limit=1)
        await client.metadata_service.ListProjects(request, timeout=5)

        duration_ms = (time.time() - start) * 1000
        result["duration_ms"] = round(duration_ms, 2)
        result["success"] = True

        # Close the channel immediately
        await client.close(grace=1.0)

        return True, result

    except grpc.aio.AioRpcError as e:
        # gRPC error - check if it's DNS-related
        is_dns_error = (
            e.code() == grpc.StatusCode.UNAVAILABLE
            and ("DNS resolution failed" in e.details() or "nodename nor servname" in e.details())
        )

        result["error"] = {
            "type": "grpc.aio.AioRpcError",
            "code": str(e.code()),
            "details": e.details(),
            "is_dns_error": is_dns_error,
            "debug_error_string": e.debug_error_string() if hasattr(e, "debug_error_string") else None,
            "traceback": traceback.format_exc(),
        }

        # Collect DNS diagnostics if it's a DNS error
        if is_dns_error:
            result["dns_info"] = get_dns_info()

            # Try nslookup if available
            try:
                nslookup_result = subprocess.run(
                    ["nslookup", hostname],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                result["nslookup"] = {
                    "stdout": nslookup_result.stdout,
                    "stderr": nslookup_result.stderr,
                    "returncode": nslookup_result.returncode,
                }
            except Exception as nslookup_error:
                result["nslookup"] = f"Error: {nslookup_error}"

            # Try dig if available
            try:
                dig_result = subprocess.run(
                    ["dig", "+short", hostname],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                result["dig"] = {
                    "stdout": dig_result.stdout,
                    "stderr": dig_result.stderr,
                    "returncode": dig_result.returncode,
                }
            except Exception as dig_error:
                result["dig"] = f"Error: {dig_error}"

            # Check network interfaces
            try:
                ifconfig_result = subprocess.run(
                    ["ifconfig"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                # Extract just active interfaces with inet addresses
                interfaces = []
                current_interface = None
                for line in ifconfig_result.stdout.split("\n"):
                    if line and not line.startswith("\t") and not line.startswith(" "):
                        current_interface = line.split(":")[0]
                    elif "inet " in line and current_interface:
                        interfaces.append(f"{current_interface}: {line.strip()}")
                result["network_interfaces"] = interfaces
            except Exception as ifconfig_error:
                result["network_interfaces"] = f"Error: {ifconfig_error}"

        return False, result

    except socket.gaierror as e:
        # DNS resolution error at socket level - collect extensive diagnostics
        result["error"] = {
            "type": "socket.gaierror",
            "errno": e.errno,
            "message": str(e),
            "args": e.args,
            "traceback": traceback.format_exc(),
        }

        # Collect current DNS info when error occurs
        result["dns_info"] = get_dns_info()

        # Try nslookup if available
        try:
            nslookup_result = subprocess.run(
                ["nslookup", hostname],
                capture_output=True,
                text=True,
                timeout=5,
            )
            result["nslookup"] = {
                "stdout": nslookup_result.stdout,
                "stderr": nslookup_result.stderr,
                "returncode": nslookup_result.returncode,
            }
        except Exception as nslookup_error:
            result["nslookup"] = f"Error: {nslookup_error}"

        # Try dig if available
        try:
            dig_result = subprocess.run(
                ["dig", "+short", hostname],
                capture_output=True,
                text=True,
                timeout=5,
            )
            result["dig"] = {
                "stdout": dig_result.stdout,
                "stderr": dig_result.stderr,
                "returncode": dig_result.returncode,
            }
        except Exception as dig_error:
            result["dig"] = f"Error: {dig_error}"

        # Check network interfaces
        try:
            ifconfig_result = subprocess.run(
                ["ifconfig"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            # Extract just active interfaces with inet addresses
            interfaces = []
            current_interface = None
            for line in ifconfig_result.stdout.split("\n"):
                if line and not line.startswith("\t") and not line.startswith(" "):
                    current_interface = line.split(":")[0]
                elif "inet " in line and current_interface:
                    interfaces.append(f"{current_interface}: {line.strip()}")
            result["network_interfaces"] = interfaces
        except Exception as ifconfig_error:
            result["network_interfaces"] = f"Error: {ifconfig_error}"

        return False, result

    except Exception as e:
        # Other errors
        result["error"] = {
            "type": type(e).__name__,
            "message": str(e),
            "traceback": traceback.format_exc(),
        }
        return False, result


def print_result(success: bool, info: dict):
    """Pretty print the test result."""
    if success:
        print(f"✓ [{info['iteration']}] {info['timestamp']} - SUCCESS (took {info['duration_ms']}ms)")
        if info["dns_test"]["success"]:
            print(
                f"  DNS resolution: {info['dns_test']['hostname']} -> "
                f"{info['dns_test']['addresses']} (took {info['dns_test']['duration_ms']}ms)"
            )
    else:
        print(f"✗ [{info['iteration']}] {info['timestamp']} - FAILED")

        # Print DNS test result
        if info["dns_test"]:
            if info["dns_test"]["success"]:
                print(
                    f"  Pre-flight DNS test: SUCCESS "
                    f"({info['dns_test']['hostname']} -> {info['dns_test']['addresses']}, "
                    f"took {info['dns_test']['duration_ms']}ms)"
                )
            else:
                print(f"  Pre-flight DNS test: FAILED")
                print(f"    Error: {info['dns_test']['error']}")

        # Print error details
        if info["error"]:
            print(f"  Error Type: {info['error']['type']}")
            if "errno" in info["error"]:
                print(f"  Error Number: {info['error']['errno']}")
            if "code" in info["error"]:
                print(f"  gRPC Status Code: {info['error']['code']}")
                print(f"  Is DNS Error: {info['error'].get('is_dns_error', 'N/A')}")
            if "message" in info["error"]:
                print(f"  Error Message: {info['error']['message']}")
            if "details" in info["error"]:
                print(f"  Error Details: {info['error']['details']}")

        # Print DNS info
        if info.get("dns_info"):
            print(f"\n  === DNS Configuration at Error Time ===")
            print(f"  Hostname: {info['dns_info']['hostname']}")
            print(f"  FQDN: {info['dns_info']['fqdn']}")
            if "nameservers" in info["dns_info"]:
                print(f"  Nameservers (via scutil):")
                if isinstance(info["dns_info"]["nameservers"], list):
                    for ns in info["dns_info"]["nameservers"][:5]:
                        print(f"    {ns}")
                else:
                    print(f"    {info['dns_info']['nameservers']}")
            if "resolv_conf" in info["dns_info"]:
                print(f"  /etc/resolv.conf:")
                if isinstance(info["dns_info"]["resolv_conf"], list):
                    for line in info["dns_info"]["resolv_conf"]:
                        print(f"    {line}")
                else:
                    print(f"    {info['dns_info']['resolv_conf']}")

        # Print nslookup result
        if info.get("nslookup"):
            print(f"\n  === nslookup test ===")
            if isinstance(info["nslookup"], dict):
                print(f"  Return code: {info['nslookup']['returncode']}")
                if info["nslookup"]["stdout"]:
                    print(f"  Output:\n    " + "\n    ".join(info["nslookup"]["stdout"].split("\n")[:10]))
                if info["nslookup"]["stderr"]:
                    print(f"  Stderr: {info['nslookup']['stderr']}")
            else:
                print(f"  {info['nslookup']}")

        # Print dig result
        if info.get("dig"):
            print(f"\n  === dig test ===")
            if isinstance(info["dig"], dict):
                print(f"  Return code: {info['dig']['returncode']}")
                if info["dig"]["stdout"]:
                    print(f"  Output: {info['dig']['stdout'].strip()}")
                if info["dig"]["stderr"]:
                    print(f"  Stderr: {info['dig']['stderr']}")
            else:
                print(f"  {info['dig']}")

        # Print network interfaces
        if info.get("network_interfaces"):
            print(f"\n  === Active Network Interfaces ===")
            if isinstance(info["network_interfaces"], list):
                for iface in info["network_interfaces"]:
                    print(f"    {iface}")
            else:
                print(f"    {info['network_interfaces']}")

        # Print traceback for debugging
        if "traceback" in info["error"]:
            print(f"\n  === Full Traceback ===")
            print("  " + "\n  ".join(info["error"]["traceback"].split("\n")))

        print()


async def run_diagnostic():
    """Main diagnostic loop."""
    # Load config from ~/.flyte/demo.yaml
    config_path = Path.home() / ".flyte" / "demo.yaml"

    print(f"DNS Diagnostic Script for Flyte Channel Creation")
    print(f"=" * 70)
    print(f"Config file: {config_path}")

    if not config_path.exists():
        print(f"ERROR: Config file not found at {config_path}")
        print(f"Please create the config file or update the path in the script.")
        sys.exit(1)

    # Load config
    cfg = config.auto(config_path)
    print(f"Endpoint: {cfg.platform.endpoint}")
    print(f"Insecure: {cfg.platform.insecure}")
    print(f"Auth mode: {cfg.platform.auth_mode}")
    print(f"=" * 70)
    print()

    # Collect initial DNS info
    initial_dns = get_dns_info()
    print("Initial DNS Configuration:")
    print(f"  Hostname: {initial_dns['hostname']}")
    print(f"  FQDN: {initial_dns['fqdn']}")
    if isinstance(initial_dns.get("nameservers"), list):
        print(f"  Nameservers:")
        for ns in initial_dns["nameservers"][:5]:
            print(f"    {ns}")
    print()

    print("Starting continuous channel creation test...")
    print("Press Ctrl+C to stop")
    print()

    iteration = 0
    success_count = 0
    failure_count = 0
    wait_seconds = 2

    try:
        while True:
            iteration += 1

            success, info = await test_channel_creation(cfg, iteration)

            if success:
                success_count += 1
            else:
                failure_count += 1

            print_result(success, info)

            # Print summary every 10 iterations
            if iteration % 10 == 0:
                print(
                    f"--- Summary: {iteration} attempts, "
                    f"{success_count} successes, {failure_count} failures ---"
                )
                print()

            # If we had a failure, you might want to collect more info
            if not success:
                print(f"Waiting {wait_seconds}s before next attempt...\n")

            await asyncio.sleep(wait_seconds)

    except KeyboardInterrupt:
        print("\n" + "=" * 70)
        print("Diagnostic stopped by user")
        print(f"Final Summary:")
        print(f"  Total attempts: {iteration}")
        print(f"  Successes: {success_count}")
        print(f"  Failures: {failure_count}")
        if iteration > 0:
            print(f"  Success rate: {(success_count / iteration) * 100:.2f}%")
        print("=" * 70)


if __name__ == "__main__":
    # Initialize logger
    initialize_logger(log_level=None, log_format="console", enable_rich=True)

    # Run the diagnostic
    asyncio.run(run_diagnostic())
