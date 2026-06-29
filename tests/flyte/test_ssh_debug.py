"""Tests for the pod-side SSH-into-task debug server (flyte._debug.ssh).

Covers the pure WebSocket framing helpers, the SshServer key/auth setup, and an
end-to-end exercise of the in-process WS->sshd bridge against a fake echo "sshd".
"""

from __future__ import annotations

import asyncio
import os
import socket
import struct

import pytest

from flyte._debug import ssh


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _client_frame(payload: bytes, opcode: int = 0x2) -> bytes:
    """Build a masked client->server WebSocket frame (clients MUST mask, RFC6455)."""
    mask = os.urandom(4)
    masked = bytes(payload[i] ^ mask[i % 4] for i in range(len(payload)))
    n = len(payload)
    hdr = bytearray([0x80 | opcode])
    if n < 126:
        hdr.append(0x80 | n)
    elif n < 65536:
        hdr.append(0x80 | 126)
        hdr += struct.pack(">H", n)
    else:
        hdr.append(0x80 | 127)
        hdr += struct.pack(">Q", n)
    return bytes(hdr) + mask + masked


# ---------------------------------------------------------------------------
# WebSocket framing
# ---------------------------------------------------------------------------


class TestWsAcceptKey:
    def test_rfc6455_vector(self):
        # The canonical example from RFC 6455 section 1.3.
        assert ssh._ws_accept_key("dGhlIHNhbXBsZSBub25jZQ==") == "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="


class TestWsFrameRoundtrip:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("size", [0, 5, 125, 126, 200, 70000])
    async def test_masked_client_frame_decodes(self, size):
        payload = os.urandom(size)
        reader = asyncio.StreamReader()
        reader.feed_data(_client_frame(payload))
        reader.feed_eof()
        result = await ssh._ws_read_frame(reader)
        assert result is not None
        opcode, got = result
        assert opcode == 0x2
        assert got == payload

    @pytest.mark.asyncio
    async def test_server_frame_is_unmasked_and_roundtrips(self):
        payload = b"hello-from-server"
        frame = ssh._ws_frame(payload)
        # MASK bit (0x80 of the 2nd byte) must be clear on server frames.
        assert frame[1] & 0x80 == 0
        reader = asyncio.StreamReader()
        reader.feed_data(frame)
        reader.feed_eof()
        opcode, got = await ssh._ws_read_frame(reader)
        assert opcode == 0x2
        assert got == payload

    @pytest.mark.asyncio
    async def test_eof_returns_none(self):
        reader = asyncio.StreamReader()
        reader.feed_eof()
        assert await ssh._ws_read_frame(reader) is None

    @pytest.mark.asyncio
    async def test_oversize_frame_rejected(self, monkeypatch):
        monkeypatch.setattr(ssh, "_MAX_WS_FRAME", 10)
        reader = asyncio.StreamReader()
        reader.feed_data(_client_frame(b"x" * 50))
        reader.feed_eof()
        with pytest.raises(ValueError):
            await ssh._ws_read_frame(reader)


# ---------------------------------------------------------------------------
# SshServer: authorized keys / host key / user
# ---------------------------------------------------------------------------


class TestSshServerSetup:
    def test_authorized_keys_written_0600(self, tmp_path):
        server = ssh.SshServer("ssh-ed25519 AAAA... user@host", user="root", scratch_dir=tmp_path)
        p = server._write_authorized_keys()
        assert p.read_text() == "ssh-ed25519 AAAA... user@host\n"
        assert (p.stat().st_mode & 0o777) == 0o600

    def test_user_defaults_to_container_user(self, tmp_path, monkeypatch):
        monkeypatch.setenv("_F_SSH_USER", "flyte")
        assert ssh.SshServer("pk", scratch_dir=tmp_path).user == "flyte"

    def test_host_key_generated_and_reused(self, tmp_path):
        server = ssh.SshServer("pk", scratch_dir=tmp_path)
        key1 = server._host_key()
        key_file = tmp_path / "ssh_host_ed25519_key"
        assert key_file.exists()
        assert (key_file.stat().st_mode & 0o777) == 0o600
        # Second call must reuse the on-disk key, not regenerate it. (Private-key
        # serialization embeds a random check value, so compare the public key.)
        key2 = server._host_key()
        assert key1.export_public_key() == key2.export_public_key()

    def test_get_ssh_user_env_override(self, monkeypatch):
        monkeypatch.setenv("_F_SSH_USER", "flyte")
        assert ssh._get_ssh_user() == "flyte"

    def test_get_pubkey_missing_raises(self, monkeypatch):
        monkeypatch.delenv("_F_SSH_PK", raising=False)
        with pytest.raises(RuntimeError):
            ssh._get_pubkey()


# ---------------------------------------------------------------------------
# WS -> sshd bridge (end to end against a fake echo "sshd")
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bridge_readiness_probe_returns_200():
    bridge_port = _free_port()
    # nothing listening on sshd_port; the probe path must not touch it
    bridge = ssh.WsBridge(sshd_port=_free_port(), listen_port=bridge_port)
    await bridge.start()
    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", bridge_port)
        writer.write(b"GET /healthz HTTP/1.1\r\nHost: x\r\n\r\n")
        await writer.drain()
        resp = await asyncio.wait_for(reader.read(200), timeout=5)
        assert b"200 OK" in resp
        writer.close()
    finally:
        await bridge.stop()


@pytest.mark.asyncio
async def test_bridge_websocket_to_sshd_echo():
    sshd_port = _free_port()
    bridge_port = _free_port()

    # Fake "sshd": echo every byte back.
    async def _echo(r, w):
        while True:
            chunk = await r.read(4096)
            if not chunk:
                break
            w.write(chunk)
            await w.drain()
        w.close()

    fake_sshd = await asyncio.start_server(_echo, "127.0.0.1", sshd_port)
    bridge = ssh.WsBridge(sshd_host="127.0.0.1", sshd_port=sshd_port, listen_port=bridge_port)
    await bridge.start()
    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", bridge_port)
        # Send a WS upgrade (path is irrelevant to the bridge, mirroring the dataplane rewrite).
        writer.write(
            b"GET /?target_port=6060 HTTP/1.1\r\n"
            b"Host: x\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
            b"Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\nSec-WebSocket-Version: 13\r\n\r\n"
        )
        await writer.drain()
        head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=5)
        assert b"101 Switching Protocols" in head
        assert b"Sec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=" in head

        # Push bytes through the WS; expect them echoed back inside a server frame.
        writer.write(_client_frame(b"ping-through-tunnel"))
        await writer.drain()
        opcode, payload = await asyncio.wait_for(ssh._ws_read_frame(reader), timeout=5)
        assert opcode == 0x2
        assert payload == b"ping-through-tunnel"
        writer.close()
    finally:
        await bridge.stop()
        fake_sshd.close()


@pytest.mark.asyncio
async def test_wsbridge_start_stop_idempotent():
    bridge = ssh.WsBridge(sshd_port=_free_port(), listen_port=_free_port())
    await bridge.start()
    # serving now; probe succeeds
    reader, writer = await asyncio.open_connection("127.0.0.1", bridge.listen_port)
    writer.write(b"GET / HTTP/1.1\r\n\r\n")
    await writer.drain()
    assert b"200 OK" in await asyncio.wait_for(reader.read(100), timeout=5)
    writer.close()
    await bridge.stop()
    await bridge.stop()  # second stop is a no-op, must not raise
