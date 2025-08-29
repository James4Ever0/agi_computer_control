Cybergod Central Bank System

A central bank system with chronic taxes, privileged accounts, and crypto currency integration.

Example code on crypto currency account management and transfer is at `test_create_crypto_account.py`.

This file contains the following sections:

- monero
- ethereum (prefered)
- moralis
- alchemy
- public_eth_rpc
- metamask_api
- sui
- solana
- bitcoin


Install requirements for crypto interactions:

```bash
pip install -r crypto-requirements.txt
```

The central bank system uses ed25519 keys for signature verification. The server and client use websockets for communication. Security measures like nonce and timestamps are implemented to prevent replay attacks.

Only admin accounts can do the following things:

- create new accounts
- mint new coins
- burn coins
- check privilege status of other accounts
- check account balance of other accounts
- change account status of other accounts

Admin accounts are predefined in server config at `privileged_keys` (public keys).

SSL is optional, but recommended for production.

Generate SSL certificates:

```bash
# this will generate the following files:
# - key.pem (private key)
# - cert.pem (certificate)
bash generate_ssl_keys.sh
```

Install central bank system requirements:

```bash
pip install -r requirements.txt
```

Generate an account for the central bank:

```bash
# in the file there is an option "write_out", toggle it to write the account to disk
python3 generate_ed25519_keys.py
```

The config file for server (template):

```yaml
privileged_keys: []
default_tax_rate: 0.1
ssl_keyfile:
ssl_certfile:
use_ssl: false
```

Run the central bank server:

```bash
# the server config file is "config.yaml" by default
# if you want to use SSL, set "use_ssl" to true and provide the paths to the SSL certificate and key files by "ssl_keyfile" and "ssl_certfile"
python3 bank-server.py
```

Run the client:

```bash
# those keys are in hexadecimal format
ADMIN_PUBKEY=...
ADMIN_PRIVKEY=...

USER1_PUBKEY=...
USER1_PRIVKEY=...

USER2_PUBKEY=...
USER2_PRIVKEY=...

# if you want to use ssl, export the following environment variable
export SSL_CERT=cert.pem

echo "Create an account (user1)"
python bank-client.py --private-key "$ADMIN_PRIVKEY" create_account --new-public-key "$USER1_PUBKEY"

echo "Create an account (user2)"
python bank-client.py --private-key "$ADMIN_PRIVKEY" create_account --new-public-key "$USER2_PUBKEY" --tax-free

echo "Mint credits (user1)"
python bank-client.py --private-key "$ADMIN_PRIVKEY" mint --account "$USER1_PUBKEY" --amount 100

echo "Mint credits (user2)"
python bank-client.py --private-key "$ADMIN_PRIVKEY" mint --account "$USER2_PUBKEY" --amount 100

echo "Burn credits (user1)"
python bank-client.py --private-key "$ADMIN_PRIVKEY" burn --account "$USER1_PUBKEY" --amount 10

echo "Check balance (user1)"
python bank-client.py --private-key "$USER1_PRIVKEY" balance --account "$USER1_PUBKEY"

echo "Check balance (user2)"
python bank-client.py --private-key "$USER2_PRIVKEY" balance --account "$USER2_PUBKEY"

echo "Transfer credits (user1 -> user2)"
python bank-client.py --private-key "$USER1_PRIVKEY" transfer --to-account "$USER2_PUBKEY" --amount 50

echo "Check tax (user1)"
python bank-client.py --private-key "$USER1_PRIVKEY" check_tax --account "$USER1_PUBKEY"

echo "Check tax (user2)"
python bank-client.py --private-key "$USER2_PRIVKEY" check_tax --account "$USER2_PUBKEY"

echo "Check account info (user2)"
python bank-client.py --private-key "$USER2_PRIVKEY" info --account "$USER2_PUBKEY"

echo "Check privilege (admin)"
python bank-client.py --private-key "$ADMIN_PRIVKEY" check_privilege --account "$ADMIN_PUBKEY" 

echo "Check privilege (user1)"
python bank-client.py --private-key "$USER1_PRIVKEY" check_privilege --account "$USER1_PUBKEY" 

echo "Check privilege (user2)"
python bank-client.py --private-key "$USER2_PRIVKEY" check_privilege --account "$USER2_PUBKEY"
```