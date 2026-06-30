from typing import Optional


def get_private_key(private_key_content: str, private_key_passphrase: Optional[str] = None) -> bytes:
    """Decode a PEM private key and return it in DER format."""
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import serialization

    private_key_bytes = private_key_content.strip().encode()
    password = private_key_passphrase.encode() if private_key_passphrase else None

    private_key = serialization.load_pem_private_key(
        private_key_bytes,
        password=password,
        backend=default_backend(),
    )

    return private_key.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
