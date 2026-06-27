"""SSH-into-task debug over WebSocket.

Pod-side counterpart to ``_debug/vscode.py``. Instead of a browser code-server
this starts an in-process SSH server (asyncssh) bound to loopback and a small
in-process server on the debug port the dataplane already routes to (:6060) that
(a) answers the cluster's code-server HTTP readiness probe with 200 so the pod
goes Ready, and (b) terminates incoming WebSockets and bridges them to the local
SSH server.

```
desktop ssh -> ws stdio proxy (ProxyCommand) -> wss:// -> Cloudflare
   -> Envoy (per-pod :6060 route) -> pod: ws bridge -> 127.0.0.1:2222 asyncssh
```

We use asyncssh (a pure-Python SSH server) rather than the system ``sshd``
binary because the default task image runs as the non-root ``flyte`` user with a
root-owned venv — so neither ``apt-get install openssh-server`` nor the binary
is available at runtime. asyncssh is a flyte dependency (pure-Python, ~370 KB,
its only requirement ``cryptography`` already ships via pyOpenSSL), so it's
always present in the image with no runtime install. The ``import asyncssh`` is
kept local to the server methods (not module-level), so importing this module —
e.g. for the ``WsBridge`` — doesn't pull asyncssh; only actually starting the
ssh debug server does.

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
import sys
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import asyncssh

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


def _build_forwarding_server():
    """An ``SSHServer`` subclass that permits client-initiated TCP forwarding.

    VS Code Remote-SSH opens a dynamic SOCKS forward (``ssh -D``) over the
    connection to reach its in-pod server. asyncssh rejects ``direct-tcpip``
    channels by default, which surfaces client-side as ``channel open failed:
    Connection refused`` / ``Failed to set up socket for dynamic port forward …
    TCP port forwarding may be disabled``. Returning ``True`` from
    ``connection_requested`` makes asyncssh open the forwarded connection itself
    — the equivalent of OpenSSH's ``AllowTcpForwarding yes``. Safe here: a
    single-tenant debug pod with sshd on loopback behind authenticated ingress.

    Built lazily so importing this module doesn't import asyncssh.
    """
    import asyncssh

    class _ForwardingServer(asyncssh.SSHServer):
        def connection_requested(self, dest_host, dest_port, orig_host, orig_port):
            return True  # asyncssh opens the TCP connection and forwards bytes

    return _ForwardingServer


class SshServer:
    """In-process asyncssh server: generate keys, authorize the user, serve shells.

    Replaces a system ``sshd`` subprocess. Binds to loopback only; reachable
    solely via the WebSocket bridge. Each session spawns the container's login
    shell (or runs an ``ssh host <cmd>`` exec), with a PTY when the client asks
    for one. SFTP is handled by asyncssh's filesystem server so VS Code
    Remote-SSH (and ``scp``) work.
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
        self._acceptor: Optional["asyncssh.SSHAcceptor"] = None

    # -- setup ---------------------------------------------------------------

    def _write_authorized_keys(self) -> Path:
        """Write the user's public key to an authorized_keys file for asyncssh."""
        self.scratch.mkdir(parents=True, exist_ok=True)
        auth_keys = self.scratch / "authorized_keys"
        auth_keys.write_text(self.pubkey + "\n")
        auth_keys.chmod(0o600)
        return auth_keys

    def _host_key(self):
        """Load (or generate, in-process) the ed25519 host key. No ssh-keygen needed."""
        import asyncssh

        self.scratch.mkdir(parents=True, exist_ok=True)
        host_key = self.scratch / "ssh_host_ed25519_key"
        if host_key.exists():
            return asyncssh.read_private_key(str(host_key))
        key = asyncssh.generate_private_key("ssh-ed25519")
        host_key.write_bytes(key.export_private_key())
        host_key.chmod(0o600)
        return key

    # -- lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        """Start the in-process SSH server on loopback."""
        import asyncssh

        authorized_keys = self._write_authorized_keys()
        host_key = self._host_key()
        logger.info(f"Starting in-process asyncssh server on {self.host}:{self.port} for user {self.user!r}")
        self._acceptor = await asyncssh.listen(
            self.host,
            self.port,
            server_factory=_build_forwarding_server(),  # allow ssh -L/-D (VS Code dynamic forward)
            server_host_keys=[host_key],
            authorized_client_keys=str(authorized_keys),
            process_factory=self._handle_session,
            sftp_factory=asyncssh.SFTPServer,  # filesystem SFTP (VS Code Remote-SSH, scp)
            allow_scp=True,
            encoding=None,  # raw bytes; the PTY does its own echo/line-editing
            keepalive_interval=30,
        )

    async def _handle_session(self, process) -> None:
        """Run one SSH session: a PTY login shell, or an ``ssh host <cmd>`` exec."""
        import asyncssh

        try:
            command = process.command
            term = process.get_terminal_type()
            env = dict(os.environ)
            env.pop("SHLVL", None)
            # Mimic a real sshd login: seed HOME/USER/LOGNAME/SHELL from the passwd
            # entry. Containers run as the non-root `flyte` user but usually leave
            # HOME as "/", so without this VS Code Remote-SSH resolves `~` to "/"
            # instead of /home/flyte (and `cd` lands in the wrong place).
            with suppress(Exception):
                import pwd

                pw = pwd.getpwuid(os.getuid())
                if pw.pw_dir:
                    env["HOME"] = pw.pw_dir
                env["USER"] = env["LOGNAME"] = pw.pw_name
                if pw.pw_shell:
                    env.setdefault("SHELL", pw.pw_shell)
            shell = env.get("SHELL") or shutil.which("bash") or "/bin/bash"
            argv = [shell, "-c", command] if command else [shell, "-l"]
            if term:
                env["TERM"] = term
                await self._run_pty(process, argv, env)
            else:
                await self._run_pipe(process, argv, env)
        except asyncssh.ConnectionLost:
            pass
        except Exception as e:  # never let a session crash take down the listener
            logger.warning(f"ssh session error: {e}")
            with suppress(Exception):
                process.exit(1)

    @staticmethod
    def _set_winsize(fd: int, size) -> None:
        """Apply a (cols, rows, ...) terminal size to a PTY via TIOCSWINSZ."""
        import fcntl
        import termios

        cols, rows = (size[0] or 0), (size[1] or 0)
        if not cols or not rows:
            return
        with suppress(Exception):
            fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))

    async def _run_pty(self, process, argv, env) -> None:
        """Spawn the shell on a PTY and bridge it to the SSH channel (with resize)."""
        import pty

        import asyncssh

        loop = asyncio.get_running_loop()
        master_fd, slave_fd = pty.openpty()
        self._set_winsize(slave_fd, process.get_terminal_size())
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                preexec_fn=os.setsid,
                stdin=slave_fd,
                stdout=slave_fd,
                stderr=slave_fd,
                env=env,
            )
        finally:
            os.close(slave_fd)

        os.set_blocking(master_fd, False)
        drained = asyncio.Event()

        def _on_master_readable():
            try:
                data = os.read(master_fd, 65536)
            except (BlockingIOError, InterruptedError):
                return
            except OSError:
                data = b""
            if data:
                process.stdout.write(data)
            else:
                with suppress(Exception):
                    loop.remove_reader(master_fd)
                drained.set()

        loop.add_reader(master_fd, _on_master_readable)

        async def _stdin_to_pty():
            try:
                while True:
                    try:
                        data = await process.stdin.read(65536)
                    except asyncssh.TerminalSizeChanged as exc:
                        self._set_winsize(master_fd, (exc.width, exc.height, exc.pixwidth, exc.pixheight))
                        continue
                    except (asyncssh.BreakReceived, asyncssh.SignalReceived):
                        continue
                    if not data:
                        break
                    os.write(master_fd, data)
            except (asyncssh.ConnectionLost, BrokenPipeError, ConnectionResetError, OSError):
                pass

        stdin_task = loop.create_task(_stdin_to_pty())
        try:
            rc = await proc.wait()
            with suppress(asyncio.TimeoutError):
                await asyncio.wait_for(drained.wait(), timeout=1)
        finally:
            stdin_task.cancel()
            with suppress(Exception):
                loop.remove_reader(master_fd)
            with suppress(Exception):
                os.close(master_fd)
        with suppress(Exception):
            await process.stdout.drain()
        process.exit(rc if rc is not None and rc >= 0 else 1)

    async def _run_pipe(self, process, argv, env) -> None:
        """No PTY (typical for ``ssh host <cmd>`` / VS Code exec): pipe std streams."""
        from asyncio.subprocess import PIPE

        import asyncssh

        proc = await asyncio.create_subprocess_exec(
            *argv, stdin=PIPE, stdout=PIPE, stderr=PIPE, env=env, start_new_session=True
        )

        async def _feed_stdin():
            try:
                while True:
                    try:
                        data = await process.stdin.read(65536)
                    except (asyncssh.BreakReceived, asyncssh.SignalReceived, asyncssh.TerminalSizeChanged):
                        continue
                    if not data:
                        break
                    proc.stdin.write(data)
                    await proc.stdin.drain()
            except (asyncssh.ConnectionLost, BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                with suppress(Exception):
                    proc.stdin.close()

        async def _pump(src_reader, dst_writer):
            try:
                while True:
                    data = await src_reader.read(65536)
                    if not data:
                        break
                    dst_writer.write(data)
                    await dst_writer.drain()
            except (asyncssh.ConnectionLost, BrokenPipeError, ConnectionResetError, OSError):
                pass

        # stdin is a background pump: an exec command finishes when its *output*
        # closes and it exits, regardless of whether the client closed stdin.
        stdin_task = asyncio.get_running_loop().create_task(_feed_stdin())
        try:
            await asyncio.gather(_pump(proc.stdout, process.stdout), _pump(proc.stderr, process.stderr))
            rc = await proc.wait()
        finally:
            stdin_task.cancel()
        with suppress(Exception):
            await process.stdout.drain()
        process.exit(rc if rc is not None and rc >= 0 else 1)

    async def wait(self, timeout: Optional[float] = None) -> None:
        """Block until the server is closed (or *timeout* elapses, raising TimeoutError)."""
        if self._acceptor is None:
            return
        await asyncio.wait_for(self._acceptor.wait_closed(), timeout=timeout)

    async def stop(self) -> None:
        """Stop accepting and close the server (best-effort)."""
        if self._acceptor is None:
            return
        self._acceptor.close()
        with suppress(Exception):
            await self._acceptor.wait_closed()


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
    """Start the asyncssh server + the in-process WS bridge, then block until uptime/death.

    ``ctx`` carries the runtime args (code bundle, dest, version) used to stage
    the workspace and the debug helpers.
    """
    server = SshServer(_get_pubkey(), user=_get_ssh_user())

    # Stage the task code (the entrypoint is blocked here before the normal
    # download), and land interactive ssh sessions in that directory.
    code_dir = await _prepare_workspace(ctx)

    await server.start()

    # The in-process WS bridge answers the readiness probe AND terminates the
    # WebSocket, forwarding to the local sshd. No wstunnel server needed on the
    # pod (the dataplane's path rewrite would break wstunnel's path addressing).
    bridge = WsBridge(sshd_host=server.host, sshd_port=server.port)
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
        # Block until the server is closed or we hit the max-uptime ceiling.
        await server.wait(timeout=max_uptime)
        logger.info("SSH server closed; shutting down SSH debug server.")
    except asyncio.TimeoutError:
        logger.info(f"SSH debug server exceeded max uptime ({max_uptime}s). Terminating...")
    finally:
        await bridge.stop()
        await server.stop()
    sys.exit()
