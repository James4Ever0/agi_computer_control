from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

# generate private key
private_key = ed25519.Ed25519PrivateKey.generate()

# generate public key
public_key = private_key.public_key()

write_out = False

# TODO: confusion prevention by prefixing the keys with "pub-" and "priv-" respectively

if not write_out:
    print("public_key:", public_key.public_bytes_raw().hex())
    print("private_key:", private_key.private_bytes_raw().hex())
else:
    with open("public_key_ed25519.hex", "w") as f:
        f.write(public_key.public_bytes_raw().hex())

    with open("private_key_ed25519.hex", "w") as f:
        f.write(private_key.private_bytes_raw().hex())

    # the generated keys are of length 64 (32 bytes)

    # also save in PEM format
    with open("public_key_ed25519.pem", "wb") as f:
        f.write(
            public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )
        )

    with open("private_key_ed25519.pem", "wb") as f:
        f.write(
            private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )
