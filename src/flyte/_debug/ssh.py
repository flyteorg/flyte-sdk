"""SSH-into-task debug over WebSocket.

Pod-side counterpart to ``_debug/vscode.py``. Instead of a browser code-server
this starts a real ``sshd`` bound to loopback and a small in-process server on
the debug port the dataplane already routes to (:6060) that (a) answers the
cluster's code-server HTTP readiness probe with 200 so the pod goes Ready, and
(b) terminates incoming WebSockets and bridges them to the local sshd.

```
desktop ssh -> ws stdio proxy (ProxyCommand) -> wss:// -> Cloudflare
   -> Envoy (per-pod :6060 route) -> pod: ws bridge -> 127.0.0.1:2222 sshd
```

We terminate the WebSocket in-process (rather than running ``wstunnel server``)
because the dataplane rewrites the request path to ``/?target_port=6060``, which
would clobber wstunnel's path-encoded tunnel addressing. The matching client is
``flyteplugins.union.remote._ws_proxy`` (a standard-WebSocket stdio proxy).

Triggered by the ``_F_E_SSH`` env var (parallel to ``_F_E_VS``); the user's
public key arrives via ``_F_SSH_PK``. See ``prds/ssh-debug-wstunnel.md``.
"""

import asyncio
import base64
import getpass
import hashlib
import os
import shutil
import struct
import subprocess
import sys
from contextlib import suppress
from pathlib import Path
from typing import Optional

from flyte._debug.constants import (
    DEFAULT_SSH_USER,
    DEFAULT_UP_SECONDS,
    FLYTE_SSH_PUBKEY_KEY,
    FLYTE_SSH_USER_KEY,
    SSH_DEBUG_DIR,
    SSH_READY_MESSAGE,
    SSHD_BIND_HOST,
    SSHD_PORT,
    WSTUNNEL_PORT,
)
from flyte._logging import logger

# Largest WebSocket frame payload we'll accept from a client (defense-in-depth;
# the peer is our own ws proxy behind the authenticated ingress). 64 MiB.
_MAX_WS_FRAME = 64 * 1024 * 1024


def _get_pubkey() -> str:
    """Read the authorized public key from the environment."""
    pubkey = os.getenv(FLYTE_SSH_PUBKEY_KEY)
    if not pubkey or not pubkey.strip():
        raise RuntimeError(
            f"SSH debug mode requires a public key. Set {FLYTE_SSH_PUBKEY_KEY} to the contents of your "
            "`~/.ssh/id_ed25519.pub` (e.g. via flyte.with_runcontext(env_vars=...))."
        )
    return pubkey.strip()


def _get_ssh_user() -> str:
    """The login user for the ssh shell (defaults to the container user)."""
    override = os.getenv(FLYTE_SSH_USER_KEY)
    if override:
        return override
    try:
        return getpass.getuser()
    except Exception:
        return DEFAULT_SSH_USER


class Sshd:
    """Manages the in-pod sshd lifecycle: locate/install, configure, start, stop.

    Everything sshd-specific lives here — finding (or apt-installing) the binary,
    generating the host key, writing the authorized_keys + a minimal key-only
    config, and running/terminating the daemon. Bound to loopback; only reachable
    via the WebSocket bridge.
    """

    def __init__(
        self,
        pubkey: str,
        *,
        user: Optional[str] = None,
        host: str = SSHD_BIND_HOST,
        port: int = SSHD_PORT,
        scratch_dir: Path = SSH_DEBUG_DIR,
    ):
        self.pubkey = pubkey.strip()
        self.user = user or _get_ssh_user()
        self.host = host
        self.port = port
        self.scratch = Path(scratch_dir)
        self._proc: Optional[asyncio.subprocess.Process] = None

    # -- setup ---------------------------------------------------------------

    @staticmethod
    def find_binary() -> str:
        """Locate the sshd binary, attempting a best-effort install if missing."""
        sshd = shutil.which("sshd")
        if sshd is None:
            for candidate in ("/usr/sbin/sshd", "/usr/local/sbin/sshd"):
                if os.path.exists(candidate):
                    sshd = candidate
                    break
        if sshd is not None:
            return sshd

        # Best-effort install on Debian/Ubuntu base images. Most images won't
        # have openssh-server; bake it in for fast cold starts.
        logger.info("sshd not found, attempting best-effort install of openssh-server...")
        try:
            subprocess.run(["apt-get", "update", "-y"], check=True, capture_output=True)
            subprocess.run(
                ["apt-get", "install", "-y", "--no-install-recommends", "openssh-server"],
                check=True,
                capture_output=True,
            )
        except Exception as e:
            raise RuntimeError(
                "sshd is not installed and automatic install failed. Bake `openssh-server` into the task "
                f"image to use ssh debug. Underlying error: {e}"
            )
        sshd = shutil.which("sshd") or "/usr/sbin/sshd"
        if not os.path.exists(sshd):
            raise RuntimeError("sshd still not found after install attempt; bake `openssh-server` into the image.")
        return sshd

    def _write_authorized_keys(self) -> Path:
        """Write the user's public key to an sshd-readable authorized_keys file."""
        self.scratch.mkdir(parents=True, exist_ok=True)
        auth_keys = self.scratch / "authorized_keys"
        auth_keys.write_text(self.pubkey + "\n")
        auth_keys.chmod(0o600)
        return auth_keys

    def _generate_host_key(self) -> Path:
        """Generate an ed25519 host key for the in-pod sshd (idempotent)."""
        host_key = self.scratch / "ssh_host_ed25519_key"
        if not host_key.exists():
            subprocess.run(
                ["ssh-keygen", "-t", "ed25519", "-f", str(host_key), "-N", "", "-q"],
                check=True,
                capture_output=True,
            )
        host_key.chmod(0o600)
        return host_key

    def _render_config(self, host_key: Path, authorized_keys: Path) -> Path:
        """Render a minimal, key-only sshd_config bound to loopback."""
        config_path = self.scratch / "sshd_config"
        pid_path = self.scratch / "sshd.pid"
        # internal-sftp: required by VS Code Remote-SSH (no external sftp-server binary).
        # StrictModes off avoids permission-mode failures on the scratch dir; key file is 0600.
        config_path.write_text(
            f"""\
Port {self.port}
ListenAddress {self.host}
HostKey {host_key}
PidFile {pid_path}
AuthorizedKeysFile {authorized_keys}
PasswordAuthentication no
PubkeyAuthentication yes
PermitRootLogin yes
KbdInteractiveAuthentication no
ChallengeResponseAuthentication no
UsePAM no
StrictModes no
PrintMotd no
X11Forwarding no
AllowTcpForwarding yes
PermitTunnel yes
Subsystem sftp internal-sftp
LogLevel INFO
"""
        )
        return config_path

    @staticmethod
    def _ensure_privsep_dir() -> None:
        """sshd needs a privilege-separation dir (/run/sshd) to exist."""
        for d in ("/run/sshd", "/var/run/sshd"):
            try:
                os.makedirs(d, exist_ok=True)
                os.chmod(d, 0o755)
            except Exception as e:
                logger.debug(f"Could not create privsep dir {d}: {e}")

    # -- lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        """Configure and launch sshd in the foreground as a child process."""
        binary = self.find_binary()
        self._ensure_privsep_dir()
        authorized_keys = self._write_authorized_keys()
        host_key = self._generate_host_key()
        config = self._render_config(host_key, authorized_keys)
        logger.info(f"Starting sshd ({binary}) on {self.host}:{self.port} for user {self.user!r}")
        # -D foreground, -e log to stderr, -f config.
        self._proc = await asyncio.create_subprocess_exec(binary, "-D", "-e", "-f", str(config))

    @property
    def returncode(self) -> Optional[int]:
        return self._proc.returncode if self._proc else None

    async def wait(self, timeout: Optional[float] = None) -> None:
        """Block until sshd exits (or *timeout* elapses, raising TimeoutError)."""
        if self._proc is None:
            return
        await asyncio.wait_for(self._proc.wait(), timeout=timeout)

    async def stop(self) -> None:
        """Terminate sshd and reap it (best-effort)."""
        if self._proc is None or self._proc.returncode is not None:
            return
        self._proc.terminate()
        with suppress(asyncio.TimeoutError):
            await asyncio.wait_for(self._proc.wait(), timeout=5)


def _write_debug_helpers(ctx) -> Optional[str]:
    """Drop debugger entrypoints into the workspace so you can pdb/VS Code the task.

    Like the code-server path, the ssh server blocks the runtime *before* the
    task body runs — so you attach a shell to a live pod and launch the task
    yourself. This writes:

    - ``.vscode/launch.json`` (reused from the code-server path) so VS Code
      Remote-SSH can run "Interactive Debugging" with the exact entrypoint args.
    - ``flyte-debug-task.sh`` — the same entrypoint under ``python -m pdb`` for a
      plain terminal debugger.

    Returns the path to the pdb helper script, or ``None`` if ctx is unavailable.
    """
    if ctx is None:
        return None
    try:
        import sys as _sys
        from pathlib import Path as _Path

        from flyte._debug.vscode import prepare_launch_json

        # Reuse the code-server launch.json generator (writes .vscode/launch.json).
        prepare_launch_json(ctx, pid=0)

        # Build the same program + args as the "Interactive Debugging" config and
        # wrap them in `python -m pdb` for a terminal session.
        launch_json = _Path(os.getcwd()) / ".vscode" / "launch.json"
        import json as _json

        cfg = _json.loads(launch_json.read_text())
        interactive = next(c for c in cfg["configurations"] if c["name"] == "Interactive Debugging")
        program = interactive["program"]
        args = interactive["args"]
        quoted = " ".join(f"'{a}'" for a in args)
        script = SSH_DEBUG_DIR / "flyte-debug-task.sh"
        script.write_text(
            "#!/usr/bin/env bash\n"
            "# Run the task entrypoint under pdb. Set a breakpoint() in your task,\n"
            "# or step from the top. Same args VS Code's 'Interactive Debugging' uses.\n"
            f"exec {_sys.executable} -m pdb {program} {quoted}\n"
        )
        script.chmod(0o755)
        return str(script)
    except Exception as e:
        logger.debug(f"Could not write debug helpers: {e}")
        return None


_WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


def _ws_accept_key(client_key: str) -> str:
    """RFC6455 Sec-WebSocket-Accept for a given Sec-WebSocket-Key."""
    digest = hashlib.sha1((client_key + _WS_GUID).encode()).digest()
    return base64.b64encode(digest).decode()


def _ws_frame(payload: bytes, opcode: int = 0x2) -> bytes:
    """Encode a server->client WebSocket frame (FIN set, unmasked)."""
    out = bytearray([0x80 | opcode])
    n = len(payload)
    if n < 126:
        out.append(n)
    elif n < 65536:
        out.append(126)
        out += struct.pack(">H", n)
    else:
        out.append(127)
        out += struct.pack(">Q", n)
    out += payload
    return bytes(out)


async def _ws_read_frame(reader: asyncio.StreamReader):
    """Read one client->server frame. Returns (opcode, payload) or None at EOF."""
    try:
        b0, b1 = await reader.readexactly(2)
    except asyncio.IncompleteReadError:
        return None
    opcode = b0 & 0x0F
    masked = b1 & 0x80
    length = b1 & 0x7F
    if length == 126:
        length = struct.unpack(">H", await reader.readexactly(2))[0]
    elif length == 127:
        length = struct.unpack(">Q", await reader.readexactly(8))[0]
    if length > _MAX_WS_FRAME:
        raise ValueError(f"WebSocket frame too large: {length} bytes")
    mask = await reader.readexactly(4) if masked else b""
    payload = await reader.readexactly(length) if length else b""
    if masked and payload:
        # Unmask in place; the masked direction (client->server) is ssh *input*,
        # which is small. Bulk output (server->client) is unmasked, so no cost there.
        ba = bytearray(payload)
        for i in range(length):
            ba[i] ^= mask[i & 3]
        payload = bytes(ba)
    return opcode, payload


async def _ws_to_tcp(reader: asyncio.StreamReader, tcp_writer: asyncio.StreamWriter, safe_send):
    """Pump client WS frames -> sshd bytes; answer pings; stop on close/EOF."""
    try:
        while True:
            frame = await _ws_read_frame(reader)
            if frame is None:
                break
            opcode, payload = frame
            if opcode == 0x8:  # close
                break
            if opcode == 0x9:  # ping -> must pong, or the client tears down the connection
                await safe_send(_ws_frame(payload, opcode=0xA))
                continue
            if opcode == 0xA:  # pong -> ignore
                continue
            if opcode in (0x0, 0x1, 0x2):  # continuation / text / binary -> ssh bytes
                tcp_writer.write(payload)
                await tcp_writer.drain()
    except Exception:
        pass
    finally:
        try:
            tcp_writer.close()
        except Exception:
            pass


async def _tcp_to_ws(tcp_reader: asyncio.StreamReader, safe_send):
    """Pump sshd bytes -> client WS binary frames; send a close frame at EOF."""
    try:
        while True:
            chunk = await tcp_reader.read(65536)
            if not chunk:
                break
            await safe_send(_ws_frame(chunk))
    except Exception:
        pass
    finally:
        try:
            await safe_send(_ws_frame(b"", opcode=0x8))  # close frame
        except Exception:
            pass


class WsBridge:
    """Readiness endpoint + WebSocket->sshd bridge on the routed debug port.

    Two jobs on the one port the dataplane routes to:
      1. Plain HTTP GET (the cluster's code-server readiness probe) -> 200, so the
         pod goes Ready and the dataplane actually routes to it (else 503).
      2. WebSocket upgrade -> complete the handshake and bridge the WS payload to
         the local sshd at ``sshd_host:sshd_port``.

    We terminate the WebSocket here (rather than running ``wstunnel server``)
    because the dataplane rewrites the request path to ``/?target_port=6060``,
    which clobbers wstunnel's path-encoded tunnel addressing. A raw WS bridge
    ignores the path entirely, so it survives the rewrite. The client side uses
    any standard WS tunnel (e.g. websocat, or the bundled stdio proxy).
    """

    def __init__(
        self,
        *,
        sshd_host: str = SSHD_BIND_HOST,
        sshd_port: int = SSHD_PORT,
        listen_port: int = WSTUNNEL_PORT,
    ):
        self.sshd_host = sshd_host
        self.sshd_port = sshd_port
        self.listen_port = listen_port
        self._server: Optional[asyncio.AbstractServer] = None

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._handle, "0.0.0.0", self.listen_port)
        logger.info(
            f"ws->sshd bridge + readiness on 0.0.0.0:{self.listen_port} -> sshd {self.sshd_host}:{self.sshd_port}"
        )

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            with suppress(Exception):
                await self._server.wait_closed()

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        # readuntil consumes exactly through the header terminator and leaves any
        # bytes after it (e.g. a coalesced first WS frame) in the reader buffer,
        # so the frame reader below doesn't lose them.
        try:
            head = await reader.readuntil(b"\r\n\r\n")
        except Exception:
            writer.close()
            return

        if b"upgrade: websocket" not in head.lower():
            # Readiness probe (or any non-WS request) -> 200 so k8s marks Ready.
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok"
            )
            try:
                await writer.drain()
            finally:
                writer.close()
            return

        # Parse Sec-WebSocket-Key and complete the handshake.
        key = None
        for line in head.split(b"\r\n"):
            if line.lower().startswith(b"sec-websocket-key:"):
                key = line.split(b":", 1)[1].strip().decode()
                break
        if not key:
            writer.write(b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\n\r\n")
            await writer.drain()
            writer.close()
            return

        accept = _ws_accept_key(key)
        writer.write(
            (
                "HTTP/1.1 101 Switching Protocols\r\n"
                "Upgrade: websocket\r\n"
                "Connection: Upgrade\r\n"
                f"Sec-WebSocket-Accept: {accept}\r\n\r\n"
            ).encode()
        )
        await writer.drain()

        # All frames to the client go through one lock: data, pong, and close can
        # otherwise interleave from the two pumps and corrupt the WS stream.
        write_lock = asyncio.Lock()

        async def safe_send(frame: bytes):
            async with write_lock:
                writer.write(frame)
                await writer.drain()

        try:
            tcp_reader, tcp_writer = await asyncio.open_connection(self.sshd_host, self.sshd_port)
        except Exception as e:
            logger.debug(f"ws bridge could not reach sshd: {e}")
            writer.close()
            return
        try:
            await asyncio.gather(
                _ws_to_tcp(reader, tcp_writer, safe_send),
                _tcp_to_ws(tcp_reader, safe_send),
            )
        finally:
            try:
                writer.close()
            except Exception:
                pass


async def _prepare_workspace(ctx) -> Optional[str]:
    """Download+extract the task code bundle and make login shells start there.

    The ssh server blocks the entrypoint *before* the normal code-bundle download
    runs, so — exactly like ``_start_vscode_server`` — we fetch it here. Otherwise
    you ssh in and the task code isn't on disk. Returns the absolute code dir.
    """
    if ctx is None:
        return None
    tgz = ctx.params.get("tgz")
    dest = ctx.params.get("dest") or "."
    code_dir = os.path.abspath(dest)
    if tgz:
        try:
            from flyte._internal.runtime.rusty import download_tgz

            await download_tgz(dest, ctx.params["version"], tgz)
            logger.info(f"Downloaded task code bundle into {code_dir}")
        except Exception as e:
            logger.warning(f"Could not download code bundle ({e}); the workspace may be empty.")

    # Login shells default to the user's $HOME (e.g. /root); drop a profile hook
    # so an interactive ssh session lands in the code dir instead.
    _write_login_cd_hook(code_dir)
    return code_dir


def _write_login_cd_hook(code_dir: str):
    """Make interactive login shells start in the task code directory."""
    try:
        os.makedirs("/etc/profile.d", exist_ok=True)
        Path("/etc/profile.d/zz-flyte-debug.sh").write_text(f'cd "{code_dir}" 2>/dev/null || true\n')
    except Exception as e:
        logger.debug(f"Could not write login-shell cd hook: {e}")


async def _start_ssh_server(ctx=None):
    """Start sshd + the in-process WS bridge, then block until uptime/death.

    ``ctx`` carries the runtime args (code bundle, dest, version) used to stage
    the workspace and the debug helpers.
    """
    sshd = Sshd(_get_pubkey(), user=_get_ssh_user())

    # Stage the task code (the entrypoint is blocked here before the normal
    # download), and land interactive ssh sessions in that directory.
    code_dir = await _prepare_workspace(ctx)

    await sshd.start()

    # The in-process WS bridge answers the readiness probe AND terminates the
    # WebSocket, forwarding to the local sshd. No wstunnel server needed on the
    # pod (the dataplane's path rewrite would break wstunnel's path addressing).
    bridge = WsBridge(sshd_host=sshd.host, sshd_port=sshd.port)
    await bridge.start()

    # Drop a launch.json + pdb helper so you can debug the task once attached.
    pdb_script = _write_debug_helpers(ctx)

    logger.info(SSH_READY_MESSAGE)
    if code_dir:
        logger.info(f"Task code is at {code_dir} (ssh sessions start there; open this folder in VS Code).")
    if pdb_script:
        logger.info("Once connected, debug the task with pdb:")
        logger.info(f"  ssh flyte-debug -t 'bash {pdb_script}'")
        logger.info("…or open the folder in VS Code Remote-SSH and run 'Interactive Debugging' (F5).")

    max_uptime = int(os.getenv("SSH_SERVER_MAX_UPTIME_SECONDS", str(DEFAULT_UP_SECONDS)))
    try:
        # Block until sshd exits or we hit the max-uptime ceiling.
        await sshd.wait(timeout=max_uptime)
        logger.info("sshd exited; shutting down SSH debug server.")
    except asyncio.TimeoutError:
        logger.info(f"SSH debug server exceeded max uptime ({max_uptime}s). Terminating...")
    finally:
        await bridge.stop()
        await sshd.stop()
    sys.exit()
