from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.exceptions import InvalidSignature

def load_hex_private_key_from_file(file_path: str) -> ed25519.Ed25519PrivateKey:
    with open(file_path, "r") as f:
        hex_key = f.read().strip()
    return load_hex_private_key(hex_key)

def load_hex_public_key_from_file(file_path: str) -> ed25519.Ed25519PublicKey:
    with open(file_path, "r") as f:
        hex_key = f.read().strip()
    return load_hex_public_key(hex_key)

def load_hex_private_key(hex_key: str) -> ed25519.Ed25519PrivateKey:
    key_bytes = bytes.fromhex(hex_key)
    privkey = ed25519.Ed25519PrivateKey.from_private_bytes(key_bytes)
    return privkey


def load_hex_public_key(hex_key: str) -> ed25519.Ed25519PublicKey:
    key_bytes = bytes.fromhex(hex_key)
    pubkey = ed25519.Ed25519PublicKey.from_public_bytes(key_bytes)
    return pubkey


def create_signature_with_hex_private_key(message: bytes, hex_private_key: str):
    private_key = load_hex_private_key(hex_private_key)
    return create_signature(message=message, private_key=private_key)

def verify_signature_with_hex_public_key(
    message: bytes, signature: bytes, hex_public_key: str
):
    public_key = load_hex_public_key(hex_public_key)
    return verify_signature(message=message, signature=signature, public_key=public_key)


def create_signature(message: bytes, private_key: ed25519.Ed25519PrivateKey) -> bytes:
    signature = private_key.sign(message)
    return signature


def verify_signature(
    message: bytes, signature: bytes, public_key: ed25519.Ed25519PublicKey
):
    """
    Verify the signature of a given message using a public key.

    Args:
        message: The original message that was signed.
        signature: The signature to verify.
        public_key: The Ed25519 public key to use for verification.

    Returns:
        Whether or not the signature is valid.
    """
    try:
        public_key.verify(signature=signature, data=message)
        return True
    except InvalidSignature:
        return False
