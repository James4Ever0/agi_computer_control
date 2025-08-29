import fastapi
import uvicorn
import yaml
import time
from fastapi import WebSocket
from pydantic import BaseModel
from typing import Optional
import json
from tinydb import TinyDB, Query
from tinydb.operations import add, subtract
import tinydb.operations
import starlette.websockets

try:
    from enum import StrEnum
except ImportError:
    from strenum import StrEnum
import logging
import asyncio
import traceback
import uuid
import secrets
from libcrypto import verify_signature_with_hex_public_key

logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

config = yaml.safe_load(open("config.yaml", "r"))

app = fastapi.FastAPI()
db = TinyDB("db.json")
accounts_table = db.table("accounts")
transactions_table = db.table("transactions")


# TODO: use protobuf for packing data into bytes
# TODO: set attribute of account, like cannot transfer, cannot accept transfer, cannot mint, disable all features etc.
class ConnectionManager:
    def __init__(self):
        self.active_connections = {}

    async def connect(self, websocket: WebSocket):
        await asyncio.wait_for(websocket.accept(), timeout=10)
        nonce = secrets.token_urlsafe(16)
        self.active_connections[websocket] = nonce
        await asyncio.wait_for(websocket.send_text(nonce), timeout=10)
        return nonce

    async def disconnect(self, websocket: WebSocket):
        if (
            not websocket.client_state
            == starlette.websockets.WebSocketState.DISCONNECTED
        ):
            try:
                await websocket.close()
            except RuntimeError:
                logger.info("Websocket has closed")
        if websocket in self.active_connections:
            del self.active_connections[websocket]


manager = ConnectionManager()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    try:
        nonce = await manager.connect(websocket)
        data = await asyncio.wait_for(websocket.receive_text(), timeout=10)
        try:
            message_data = json.loads(data)

            # Verify timestamp
            current_time = time.time()
            if abs(current_time - message_data["timestamp"]) > 60:
                await websocket.send_text("Timestamp expired")
                return

            # Verify nonce
            if message_data["nonce"] != nonce:
                await websocket.send_text("Invalid nonce")
                return

            # Verify signature
            signature = bytes.fromhex(message_data["signature"])
            message_string = f"{message_data['endpoint']}{message_data['message']}{nonce}{message_data['timestamp']}"

            client_public_key_hex = message_data["client_public_key"]

            signature_valid = verify_signature_with_hex_public_key(
                message=message_string.encode(),
                signature=signature,
                hex_public_key=client_public_key_hex,
            )
            if not signature_valid:
                await websocket.send_text("Invalid signature")
                return
            logger.info(f"Verified message: {message_data['message']}")
            endpoint = message_data["endpoint"]
            logger.info("Message endpoint: %s" % endpoint)
            await route_requests(
                endpoint=endpoint,
                client_public_key=client_public_key_hex,
                message=message_data["message"],
                websocket=websocket,
            )
            # according to the endpoint, handle the message
            logger.info("Message verified successfully")
        except:
            trace_id = uuid.uuid4()
            logger.info("TraceID: %s", trace_id)
            logger.info(traceback.format_exc())
            await websocket.send_text("Error (TraceID: %s)" % trace_id)

    except fastapi.WebSocketDisconnect:
        print(f"Client disconnected: {websocket.client}")
    except asyncio.TimeoutError:
        print(f"Client timed out: {websocket.client}")
    finally:
        await manager.disconnect(websocket)


# Models
class Account(BaseModel):
    public_key: str
    balance: float = 0.0
    tax_rate: float = 0.01  # Default tax rate
    tax_free: bool = False
    disabled: bool = False
    cannot_send_tx: bool = False
    cannot_accept_tx: bool = False
    created_at: float = time.time()
    last_tax_time: float = time.time()


class TransactionType(StrEnum):
    transfer = "transfer"
    mint = "mint"
    burn = "burn"
    deposit = "deposit"
    withdraw = "withdraw"


class Transaction(BaseModel):
    from_account: Optional[str] = None
    to_account: str
    amount: float
    timestamp: float
    type: TransactionType


class CreateAccountRequest(BaseModel):
    public_key: str
    tax_rate: Optional[float] = None
    tax_free: Optional[bool] = None


class TransferRequest(BaseModel):
    from_account: str
    to_account: str
    amount: float
    signature: str


class AccountUpdateRequest(BaseModel):
    account: str
    tax_rate: Optional[float] = None
    tax_free: Optional[bool] = None
    cannot_send_tx: Optional[bool] = None
    cannot_accept_tx: Optional[bool] = None
    disabled: Optional[bool] = None


async def route_requests(
    endpoint: str, client_public_key: str, message: str, websocket: WebSocket
):
    if endpoint == "/check_privilege":
        await check_privilege(
            client_public_key=client_public_key, account=message, websocket=websocket
        )
    elif endpoint == "/balance":
        if message != client_public_key:
            if not client_public_key in config["privileged_keys"]:
                await websocket.close(code=1008, reason="Unauthorized balance check")
                return
        await get_balance(account_pubkey=message, websocket=websocket)
    elif endpoint == "/create_account":
        req = CreateAccountRequest.parse_raw(message)
        await create_account(admin_key=client_public_key, req=req, websocket=websocket)
    elif endpoint == "/burn":
        req = BurnRequest.parse_raw(message)
        await burn(admin_key=client_public_key, req=req, websocket=websocket)
    elif endpoint == "/info":
        account_pubkey = message
        if account_pubkey != client_public_key:
            if not client_public_key in config["privileged_keys"]:
                await websocket.close(
                    code=1008, reason="Unauthorized to get account info"
                )
                return
        await get_info(account_pubkey=account_pubkey, websocket=websocket)
    elif endpoint == "/transfer":
        req = TransferRequest.parse_raw(message)
        if req.from_account != client_public_key:
            await websocket.send_text("Unauthorized transfer")
            logger.info("Unauthorized transfer")
            return
        else:
            await transfer(req=req, websocket=websocket)
    elif endpoint == "/mint":
        req = MintRequest.parse_raw(message)
        await mint(req=req, admin_key=client_public_key, websocket=websocket)
    elif endpoint == "/update_account":
        req = AccountUpdateRequest.parse_raw(message)
        await update_account(req=req, admin_key=client_public_key, websocket=websocket)
    elif endpoint == "/check_tax":
        account = message
        if account != client_public_key:
            if account not in config["privileged_keys"]:
                await websocket.close(code=1008, reason="Unauthorized tax check")
                logger.info("Unauthorized tax check")
                return
        await check_tax(account=account, websocket=websocket)
    else:
        await websocket.send_text("Unknown endpoint: %s" % endpoint)
        logger.info("Unknown endpoint: %s" % endpoint)


def calculate_tax(account):
    logger.info("Calculating tax for account: %s" % account["public_key"])
    if account["tax_free"]:
        logger.info("Account is tax-free")
        return account

    current_time = time.time()
    logger.info("Current time: %s", current_time)
    tax_duration = current_time - account["last_tax_time"]

    # if we time the tax rate with account balance, we will get the tax amount according to the overall balance size

    # moreover, the tax system could be further enhanced by using a more sophisticated tax rate calculation

    # for example, we could use a progressive tax rate system where the tax rate increases as the account balance increases

    # or we could only tax the income generated by the account, rather than the entire balance

    # tax_amount = tax_duration * account["tax_rate"] * account["balance"]

    tax_amount = tax_duration * account["tax_rate"]

    logger.info("Tax rate: %s", account["tax_rate"])
    logger.info("Tax duration: %s", tax_duration)
    logger.info("Tax amount: %s", tax_amount)
    logger.info("Balance: %s", account["balance"])

    if tax_amount > account["balance"]:
        account["balance"] = 0
        logger.info("Tax amount exceeds balance, setting balance to 0")
    else:
        account["balance"] -= tax_amount
        logger.info("Tax deducted from balance")

    account["last_tax_time"] = current_time
    logger.info("Updated last tax time to %s", account["last_tax_time"])
    return account


# API Endpoints
async def create_account(
    req: CreateAccountRequest, admin_key: str, websocket: fastapi.WebSocket
):
    if req.public_key in config["privileged_keys"]:
        await websocket.close(
            code=1008, reason="Cannot create account for privileged keys"
        )
        return
    if admin_key not in config["privileged_keys"]:
        await websocket.close(code=1008, reason="Not authorized")
        return

    AccountQuery = Query()
    if accounts_table.contains(AccountQuery.public_key == req.public_key):
        await websocket.close(code=1008, reason="Account already exists")
        return

    account_data = {
        "public_key": req.public_key,
        "balance": 0.0,
        "tax_rate": req.tax_rate or config["default_tax_rate"],
        "tax_free": req.tax_free or False,
        "disabled": False,
        "cannot_send_tx": False,
        "cannot_accept_tx": False,
        "created_at": time.time(),
        "last_tax_time": time.time(),
    }

    accounts_table.insert(account_data)
    await websocket.send_json({"message": "Account created successfully"})


async def get_balance(account_pubkey: str, websocket: fastapi.WebSocket):
    AccountQuery = Query()
    account = accounts_table.get(AccountQuery.public_key == account_pubkey)
    if not account:
        await websocket.close(code=1008, reason="Account not found")
        return
    if account["disabled"]:
        await websocket.close(code=1008, reason="Account is disabled")
        return
    amount = _execute_tax(account_pubkey)
    await websocket.send_json({"balance": amount})


async def get_info(account_pubkey: str, websocket: fastapi.WebSocket):
    AccountQuery = Query()
    account = accounts_table.get(AccountQuery.public_key == account_pubkey)
    if not account:
        await websocket.close(code=1008, reason="Account not found")
        return
    if account["disabled"]:
        await websocket.close(code=1008, reason="Account is disabled")
        return
    _execute_tax(account_pubkey)
    account = accounts_table.get(AccountQuery.public_key == account_pubkey)
    try:
        assert account
    except AssertionError:
        await websocket.close(code=1008, reason="Account not found after tax execution")
        return
    await websocket.send_json(account)


class BurnRequest(BaseModel):
    amount: float
    account: str


async def transfer(req: TransferRequest, websocket: WebSocket):
    AccountQuery = Query()
    from_account = accounts_table.get(AccountQuery.public_key == req.from_account)
    to_account = accounts_table.get(AccountQuery.public_key == req.to_account)

    try:
        assert req.amount > 0
    except AssertionError:
        await websocket.close(code=1008, reason="Invalid amount")
        return

    if not from_account or not to_account:
        await websocket.close(code=1008, reason="Account not found")
        return

    if from_account["disabled"] or to_account["disabled"]:
        await websocket.close(code=1008, reason="Account is disabled")
        return

    if from_account["cannot_send_tx"] or to_account["cannot_accept_tx"]:
        await websocket.close(code=1008, reason="Account is locked")
        return

    # Execute taxes
    from_account_balance = _execute_tax(from_account["public_key"])
    _execute_tax(to_account["public_key"])

    if from_account_balance < req.amount:
        await websocket.close(code=1008, reason="Insufficient funds")
        return

    # Update balances
    accounts_table.update(
        subtract("balance", req.amount), AccountQuery.public_key == req.from_account
    )
    accounts_table.update(
        add("balance", req.amount), AccountQuery.public_key == req.to_account
    )

    # Record transaction
    transactions_table.insert(
        {
            "from_account": req.from_account,
            "to_account": req.to_account,
            "amount": req.amount,
            "timestamp": time.time(),
            "type": "transfer",
        }
    )

    await websocket.send_json({"message": "Transfer successful"})


class MintRequest(BaseModel):
    account: str
    amount: float


async def burn(req: BurnRequest, admin_key: str, websocket: fastapi.WebSocket):
    account = req.account
    amount = req.amount
    try:
        assert amount > 0
    except AssertionError:
        await websocket.close(code=1008, reason="Invalid amount")
        return
    if admin_key not in config["privileged_keys"]:
        await websocket.close(code=1008, reason="Not authorized")
        return
    if account in config["privileged_keys"]:
        await websocket.close(code=1008, reason="Cannot burn from privileged account")
        return

    AccountQuery = Query()

    if not accounts_table.contains(AccountQuery.public_key == account):
        await websocket.close(code=1008, reason="Account not found")
        return

    account_amount = _execute_tax(account)
    if amount < account_amount:
        accounts_table.update(
            subtract("balance", amount), AccountQuery.public_key == account
        )
    else:
        # Set balance to zero
        accounts_table.update(
            tinydb.operations.set("balance", 0.0), AccountQuery.public_key == account
        )

    transactions_table.insert(
        {
            "from_account": None,
            "to_account": account,
            "amount": min(amount, account_amount),
            "timestamp": time.time(),
            "type": "burn",
        }
    )

    await websocket.send_json({"message": "Burn successful"})


async def mint(req: MintRequest, admin_key: str, websocket: fastapi.WebSocket):
    account = req.account
    amount = req.amount
    try:
        assert amount > 0
    except AssertionError:
        await websocket.close(code=1008, reason="Invalid amount")
        return
    if admin_key not in config["privileged_keys"]:
        await websocket.close(code=1008, reason="Not authorized")
        return
    if account in config["privileged_keys"]:
        await websocket.close(code=1008, reason="Cannot mint to privileged account")
        return

    AccountQuery = Query()

    if not accounts_table.contains(AccountQuery.public_key == account):
        await websocket.close(code=1008, reason="Account not found")
        return

    _execute_tax(account)
    accounts_table.update(add("balance", amount), AccountQuery.public_key == account)

    transactions_table.insert(
        {
            "from_account": None,
            "to_account": account,
            "amount": amount,
            "timestamp": time.time(),
            "type": "mint",
        }
    )

    await websocket.send_json({"message": "Mint successful"})


async def check_tax(account: str, websocket: fastapi.WebSocket):
    AccountQuery = Query()
    account_data = accounts_table.get(AccountQuery.public_key == account)

    if not account_data:
        await websocket.close(code=1008, reason="Account not found")
        return

    await websocket.send_json(
        {"tax_rate": account_data["tax_rate"], "tax_free": account_data["tax_free"]}
    )


def _execute_tax(account: str):
    AccountQuery = Query()
    account_data = accounts_table.get(AccountQuery.public_key == account)

    if not account_data:
        raise ValueError("Account not found")

    account_data = calculate_tax(account_data)
    accounts_table.update(
        tinydb.operations.set("balance", account_data["balance"]),
        AccountQuery.public_key == account,
    )
    accounts_table.update(
        tinydb.operations.set("last_tax_time", account_data["last_tax_time"]),
        AccountQuery.public_key == account,
    )

    return account_data["balance"]


async def check_privilege(client_public_key: str, account: str, websocket: WebSocket):
    if client_public_key not in config["privileged_keys"]:
        await websocket.close(code=1008, reason="Not authorized")
        return
    if account not in config["privileged_keys"]:
        await websocket.send_json({"message": "Unprivileged"})
    else:
        await websocket.send_json({"message": "Privileged"})


async def update_account(
    req: AccountUpdateRequest, admin_key: str, websocket: WebSocket
):
    if admin_key not in config["privileged_keys"]:
        await websocket.close(code=1008, reason="Not authorized")
        return

    AccountQuery = Query()
    updates = {}
    if req.tax_rate is not None:
        updates["tax_rate"] = req.tax_rate
    if req.tax_free is not None:
        updates["tax_free"] = req.tax_free
    if req.cannot_accept_tx is not None:
        updates["cannot_accept_tx"] = req.cannot_accept_tx
    if req.cannot_send_tx is not None:
        updates["cannot_send_tx"] = req.cannot_send_tx
    if req.disabled is not None:
        updates["disabled"] = req.disabled
    if updates:
        accounts_table.update(updates, AccountQuery.public_key == req.account)

    await websocket.send_json({"message": "Account settings updated"})


def main():
    # TODO: to further enhance security, implement rate limiting, logging, https/wss and other security measures
    use_ssl = config['use_ssl']
    if use_ssl:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=12742,
            log_level="debug",
            ssl_keyfile=config["ssl_keyfile"],
            ssl_certfile=config["ssl_certfile"],
        )
    else:
        uvicorn.run(app, host="0.0.0.0", port=12742, log_level="debug")
    # SSL PARAMETERS
    # --------------
    # ssl_keyfile: str | PathLike[str] | None = None,
    # ssl_certfile: str | PathLike[str] | None = None,
    # ssl_keyfile_password: str | None = None,
    # ssl_version: int = SSL_PROTOCOL_VERSION,
    # ssl_cert_reqs: int = ssl.CERT_NONE,
    # ssl_ca_certs: str | None = None,
    # ssl_ciphers: str = "TLSv1",


if __name__ == "__main__":
    main()
