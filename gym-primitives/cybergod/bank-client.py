import argparse
import json
import hashlib
from libcrypto import load_hex_private_key, create_signature
import websocket
import time
import logging
import ssl
import struct
import typing
import os

# TODO: add more fields, like private key, public key, etc.

logger = logging.getLogger(__name__)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.setLevel(logging.DEBUG)


class CentralBankClient:
    def __init__(
        self,
        server_ws: str,
        private_key_hex: str,
        certfile: typing.Optional[str] = None,
    ):
        """
        Initialize the CentralBankClient.

        Args:
            server_ws: The WebSocket URL of the Central Bank server.
            private_key_hex: The private key (hex string) of the account.
            certfile: The SSL certificate file path (optional).
        """
        self.server_ws = server_ws
        self.certfile = certfile
        if certfile is not None:
            logger.info("Using SSL certificate file: %s", certfile)
            self.server_ws = server_ws.replace("ws://", "wss://")
        self.private_key = load_hex_private_key(private_key_hex)
        self.public_key = self.private_key.public_key()
        self.public_key_hex = self.public_key.public_bytes_raw().hex()
        logger.info("Derived public key: %s", self.public_key_hex)

    def check_privilege(self, account: str) -> bool:
        # ask the server if given account is privileged.
        response = self.websocket_manager(endpoint="/check_privilege", payload=account)
        try:
            response_data = json.loads(response)
            assert type(response_data) == dict
            return response_data.get("message", "") == "Privileged"
        except AssertionError:
            logger.error("Invalid server response format")
            return False
        except json.JSONDecodeError:
            logger.error("Failed to decode server response")
            return False

    def websocket_manager(self, endpoint: str, payload: str):
        # sslopt: dict
        # reference: https://websocket-client.readthedocs.io/en/latest/faq.html#what-else-can-i-do-with-sslopts
        use_ssl = self.certfile is not None
        if use_ssl:
            logger.info("Using SSL for websocket connection")
            assert self.certfile
            assert os.path.isfile(self.certfile)
            ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            ssl_context.load_cert_chain(certfile=self.certfile)
            sslops = {"context": ssl_context}
            ws = websocket.create_connection(self.server_ws, sslopt=sslops)
        else:
            ws = websocket.create_connection(self.server_ws)
        try:
            message = ws.recv()
            if type(message) == bytes:
                message = message.decode("utf-8")
            assert type(message) == str, "Invalid message type"
            logger.info(f"Received nonce from server: {message}")
            nonce = message

            # Prepare message data
            timestamp = time.time()
            message_string = f"{endpoint}{payload}{nonce}{timestamp}"

            # Create signature
            signature = create_signature(
                message=message_string.encode(), private_key=self.private_key
            )

            # Prepare data to send
            data_to_send = {
                "message": payload,
                "client_public_key": self.public_key_hex,
                "endpoint": endpoint,
                "nonce": nonce,
                "timestamp": timestamp,
                "signature": signature.hex(),
            }

            ws.send(json.dumps(data_to_send))
            logger.info("Sent signed message to server")

            resp_opcode, msg = ws.recv_data()
            logger.info("Response opcode: " + str(resp_opcode))
            if resp_opcode == 8 and len(msg) >= 2:
                logger.info(
                    "Response close code: " + str(struct.unpack("!H", msg[0:2])[0])
                )
                logger.info("Response message: " + str(msg[2:]))
                return json.dumps(
                    {
                        "close_code": struct.unpack("!H", msg[0:2])[0],
                        "message": msg[2:].decode("utf-8", errors="replace"),
                    }
                )
            else:
                logger.info("Response message: " + str(msg))
                response = msg
            if type(response) == bytes:
                response = response.decode("utf-8")
            assert type(response) == str, "Server response is not a string"
            logger.info(f"Server response: {response}")
            return response
        finally:
            ws.getstatus
            ws.close()

    def burn(self, account: str, amount: float):
        # check privilege
        authorized = self.check_privilege(self.public_key_hex)
        if not authorized:
            logger.error("This account is not authorized to burn tokens")
            return {"error": "Not authorized"}
        # send request to server
        response = self.websocket_manager(
            endpoint="/burn", payload=json.dumps({"account": account, "amount": amount})
        )
        return json.loads(response)

    def create_account(
        self,
        new_public_key: str,
        tax_rate=None,
        tax_free=None,
    ):
        authorized = self.check_privilege(self.public_key_hex)
        if not authorized:
            logger.error("This account is not privileged, unable to create account")
            return
        data = {
            "public_key": new_public_key,
            "tax_rate": tax_rate,
            "tax_free": tax_free,
        }
        message = json.dumps(data)
        response = self.websocket_manager("/create_account", message)
        response = json.loads(response)
        return response

    def transfer(self, to_account: str, amount):
        endpoint = "/transfer"
        from_account = self.public_key_hex
        message = json.dumps(
            {
                "from_account": from_account,
                "to_account": to_account,
                "amount": amount,
                "timestamp": time.time(),
            }
        )
        signature = hashlib.sha256(f"{from_account}{message}".encode()).hexdigest()

        data = {
            "from_account": from_account,
            "to_account": to_account,
            "amount": amount,
            "signature": signature,
        }
        payload = json.dumps(data)
        response = self.websocket_manager(endpoint, payload)
        return json.loads(response)

    def mint(self, account: str, amount: float):
        endpoint = "/mint"
        params = {"account": account, "amount": amount}
        response = self.websocket_manager(endpoint=endpoint, payload=json.dumps(params))
        return json.loads(response)

    def get_balance(self, account: str):
        endpoint = "/balance"
        response = self.websocket_manager(endpoint=endpoint, payload=account)
        return json.loads(response)

    def update_account(
        self,
        account: str,
        tax_rate=None,
        tax_free=None,
        cannot_send_tx=None,
        cannot_receive_tx=None,
        disabled=None,
    ):
        url = "/update_account"
        data = {
            "account": account,
            "tax_rate": tax_rate,
            "tax_free": tax_free,
            "cannot_send_tx": cannot_send_tx,
            "cannot_receive_tx": cannot_receive_tx,
            "disabled": disabled,
        }
        payload = json.dumps(data)
        response = self.websocket_manager(endpoint=url, payload=payload)
        return json.loads(response)

    def get_info(self, account: str):
        url = "/info"
        response = self.websocket_manager(endpoint=url, payload=account)
        return json.loads(response)

    def check_tax(self, account: str):
        url = "/check_tax"
        response = self.websocket_manager(endpoint=url, payload=account)
        return json.loads(response)


def main():
    parser = argparse.ArgumentParser(description="Central Bank Client")
    parser.add_argument(
        "--server-ws", default="ws://127.0.0.1:12742/ws", help="Server WebSocket URL"
    )
    parser.add_argument("--private-key", required=True)

    SSL_CERT=os.environ.get("SSL_CERT", None)
    if SSL_CERT:
        assert os.path.isfile(SSL_CERT), "SSL_CERT file does not exist at {}".format(SSL_CERT)

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Create account command
    create_parser = subparsers.add_parser("create_account")
    create_parser.add_argument("--new-public-key", required=True)
    create_parser.add_argument("--tax-rate", type=float)
    create_parser.add_argument("--tax-free", action="store_true")

    # Transfer command
    transfer_parser = subparsers.add_parser("transfer")
    transfer_parser.add_argument("--to-account", required=True)
    transfer_parser.add_argument("--amount", type=float, required=True)

    # Mint command
    mint_parser = subparsers.add_parser("mint")
    mint_parser.add_argument("--account", required=True)
    mint_parser.add_argument("--amount", type=float, required=True)

    # Balance command
    balance_parser = subparsers.add_parser("balance")
    balance_parser.add_argument("--account", required=True)

    # Update account command
    update_account_parser = subparsers.add_parser("update_account")
    update_account_parser.add_argument("--account", required=True)
    update_account_parser.add_argument("--tax-rate", type=float)
    update_account_parser.add_argument("--tax-free", action="store_true")
    update_account_parser.add_argument("--cannot_send_tx", type=bool)
    update_account_parser.add_argument("--cannot_accept_tx", type=bool)
    update_account_parser.add_argument("--disabled", type=bool)

    # Check tax command
    check_tax_parser = subparsers.add_parser("check_tax")
    check_tax_parser.add_argument("--account", required=True)

    # Burn command
    burn_parser = subparsers.add_parser("burn")
    burn_parser.add_argument("--account", required=True)
    burn_parser.add_argument("--amount", type=float, required=True)

    # Get info command
    get_info_parser = subparsers.add_parser("info")
    get_info_parser.add_argument("--account", type=str, required=True)

    check_privilege_parser = subparsers.add_parser("check_privilege")
    check_privilege_parser.add_argument("--account", required=True)

    args = parser.parse_args()
    client = CentralBankClient(
        server_ws=args.server_ws,
        private_key_hex=args.private_key,
        certfile=SSL_CERT,
    )
    if args.command == "check_privilege":
        authorized = client.check_privilege(args.account)
        result = dict(authorized=authorized)
    elif args.command == "create_account":
        result = client.create_account(
            new_public_key=args.new_public_key,
            tax_rate=args.tax_rate,
            tax_free=args.tax_free,
        )
    elif args.command == "transfer":
        result = client.transfer(to_account=args.to_account, amount=args.amount)
    elif args.command == "mint":
        result = client.mint(account=args.account, amount=args.amount)
    elif args.command == "burn":
        result = client.burn(account=args.account, amount=args.amount)
    elif args.command == "info":
        result = client.get_info(args.account)
    elif args.command == "balance":
        result = client.get_balance(args.account)
    elif args.command == "update_account":
        result = client.update_account(
            account=args.account,
            tax_rate=args.tax_rate,
            tax_free=args.tax_free,
            cannot_send_tx=args.cannot_send_tx,
            cannot_receive_tx=args.cannot_receive_tx,
            disabled=args.disabled,
        )
    elif args.command == "check_tax":
        result = client.check_tax(args.account)
    else:
        raise ValueError("Invalid command: {}".format(args.command))

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
