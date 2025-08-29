# basic tasks:
# create account, get balance, perform transaction

# btw, usdt is a stablecoin on eth (erc20)
# use stable coin, like usdc (without gas), usdt
# https://www.circle.com/usdc
# https://www.circle.com/developer

def test_etherscan():
    # just get balance
    # reference: https://etherscan.io/apis
    url = "https://api.etherscan.io"
    apikey = ...

def test_usdc():
    ...

def test_bitcoin():
    # https://bitcoin-rpc.publicnode.com
    # rpc: https://www.blockchain.com/explorer/api
    # reference: https://github.com/jgarzik/python-bitcoinrpc
    url = "https://bitcoin-rpc.publicnode.com"
    from bitcoinrpc.authproxy import AuthServiceProxy

    rpc = AuthServiceProxy(url)
    print(rpc.getbalance())


def test_sui():
    # requiring pysui, and sui binary install
    # rpc: https://sui-rpc.publicnode.com/
    # reference: https://pysui.readthedocs.io/en/latest
    from pysui import SuiConfig, SyncClient
    from pysui.abstracts.client_keypair import SignatureScheme

    url = "https://sui-rpc.publicnode.com/"

    # Option-2: Alternate setup configuration without keystrings
    cfg = SuiConfig.user_config(rpc_url=url)

    # One address (and keypair), at least, should be created
    # First becomes the 'active-address'
    _mnen, _address = cfg.create_new_keypair_and_address(scheme=SignatureScheme.ED25519)

    # Synchronous client
    client = SyncClient(cfg)  # has error over the httpx part, proxies -> proxy

    # coin_type = "0x2::sui::SUI"
    coin_type = "0x168da5bf1f48dafc111b0a488fa454aca95e0b5e::usdc::USDC"
    result = client.get_coin(coin_type=coin_type)
    print("Sui get coin:", result.result_data)  # nothing returned


def test_solana():
    # rpc: https://solana-rpc.publicnode.com
    # reference: https://github.com/michaelhly/solana-py
    url = "https://solana-rpc.publicnode.com"
    from solana.rpc.api import Client
    from solders.pubkey import Pubkey
    from solders.keypair import Keypair

    http_client = Client(url)
    response = http_client.get_balance(Pubkey([0] * 31 + [1]))
    print("Solana balance:", response.value)  # 14124540759

    from solana.rpc.api import Client
    from solders.system_program import TransferParams, transfer
    from solders.message import Message
    from solders.transaction import Transaction

    leading_zeros = [0] * 31
    sender, receiver = Keypair.from_seed(leading_zeros + [1]), Keypair.from_seed(
        leading_zeros + [2]
    )
    ixns = [
        transfer(
            TransferParams(
                from_pubkey=sender.pubkey(), to_pubkey=receiver.pubkey(), lamports=1000
            )
        )
    ]
    msg = Message(ixns, sender.pubkey())
    http_client.send_transaction(
        Transaction([sender], msg, http_client.get_latest_blockhash().value.blockhash)
    )


# use a multi-chain wallet
def test_moralis():
    from moralis import evm_api

    api_key = ...

    params = {"chain": "eth", "address": "0xcB1C1FdE09f811B294172696404e88E658659905"}

    # the result is more than enough. we only want the ETH balance, not other tokens
    result = evm_api.wallets.get_wallet_token_balances_price(
        api_key=api_key,
        params=params,
    )

    print(result)

    # perform transaction
    # rpc endpoint: site1.moralis-nodes.com
    # using rpc is better than evm_api, since it only has rate per minute limit, not 400 times per day
    # reference: https://docs.moralis.com/rpc-nodes/ethereum-json-rpc-api


# metamask is used for trading on eth
# metamask is now infura
def test_metamask(): ...


def test_ethereum():
    import web3
    from web3 import Web3
    api_key = ...
    eth_mainnet_http_provider = (
        "https://mainnet.infura.io/v3/%s" % api_key
    )
    w3 = Web3(
        Web3.HTTPProvider(
            eth_mainnet_http_provider,
            request_kwargs={
                "proxies": {
                    "https": ...,
                    "http": ...,
                }
            },
        )
    )
    print("Connected:", w3.is_connected())  # working with proxy.
    account = web3.Account()
    out = account.create()
    print("Created account:", out)  # type: eth_account.signers.local.LocalAccount

    import web3.eth

    eth = web3.eth.Eth(w3)
    print("ETH Balance:", eth.get_balance(out.address))

    address = "0xbe0eb53f46cd790cd13851d5eff43d12404d33e8"
    address = w3.to_checksum_address(address)
    balance_wei = eth.get_balance(address)
    print(f"Rich account balance: {w3.from_wei(balance_wei, 'ether')} ETH")

    # perform transaction


# monero is to be the most complex one to setup properly. we do not want this extra privacy layer at the price of convenience
def test_monero():
    # use monero, use mainnet
    # local hosting a node (apt install monero; monerod) would be disk consuming. use a remote node

    # find one node at:
    # https://monero.fail/

    # first you would use monero wallet cli to connect to a remote node and expose a local jsonrpc port
    # monero-wallet-cli --rpc-bind-ip=127.0.0.1 --rpc-bind-port=18081 --trusted-daemon --trusted-node=<remote_host>:<remote_port> --generate-from-jsonrpc

    # from monero.wallet import Wallet
    # from monero.backends.jsonrpc import JSONRPCWallet

    # backend = JSONRPCWallet(host="127.0.0.1", port=28088)

    # wallet = Wallet(backend=backend)  # cannot use the request "get_accounts" here.
    # print("Master Address:", wallet.address())
    # print("Master view key:", wallet.view_key())
    # print("Master Balance:", wallet.balance())

    # new_account = wallet.new_account()
    # print("New address:", new_account.address())
    # print("New balance:", new_account.balance())

    from monero.daemon import Daemon

    daemon = Daemon(host="xmr.gn.gy", port=18089)  # working
    print("Daemon info:", daemon.info())
    print("Daemon net:", daemon.net)
    # daemon.send_transaction(tx="")


def test_alchemy():
    # https://blastapi.io/public-api/ethereum (closing soon)
    # https://alchemyapi.io/
    ...


def test_public_eth_rpc():
    # cloudflare has a free eth rpc
    # https://cloudflare-eth.com/

    # https://ethereumnodes.com/
    # https://ethereum.stackexchange.com/questions/102967/free-and-public-ethereum-json-rpc-api-nodes
    # https://ethereum.org/en/developers/docs/nodes-and-clients/nodes-as-a-service/

    # url = "https://cloudflare-eth.com" # does not work for checking balance
    url = "https://ethereum-rpc.publicnode.com"

    import web3
    import web3.eth

    w3 = web3.Web3(web3.HTTPProvider(url))

    eth = web3.eth.Eth(w3)
    print("Public eth rpc connected:", w3.is_connected())  # True
    if w3.is_connected():
        address = "0xbe0eb53f46cd790cd13851d5eff43d12404d33e8"
        address = w3.to_checksum_address(address)
        balance_wei = eth.get_balance(address)  # not working
        print(f"Account balance: {w3.from_wei(balance_wei, 'ether')} ETH")


def test():
    # test_monero()
    # test_ethereum()
    # test_moralis()
    # test_alchemy()
    test_public_eth_rpc()
    # test_metamask()
    # test_sui()
    # test_solana()
    # test_bitcoin()


if __name__ == "__main__":
    test()
