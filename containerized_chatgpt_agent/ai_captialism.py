# there must be some smaller groups called "company" evolved in the process.
# the total amount of currency remained unchanged.
import uuid
from typing import Callable

# TODO: socialism, communism
# TODO: is your cybergod misbehaving because of psycological reasons
total_amount = 1e5
population_limit = 100

central_account_registry = {}
central_bank_registry = {}


def create_account(
    user_registry, base_amount=100
):  # you can create account in 'central_account_registry' and 'central_bank_registry'
    if len(user_registry) > population_limit:
        raise Exception(
            "too many accounts: %d; limit: %d" % (len(user_registry), base_amount)
        )
    for _ in range(3):
        user_id = str(uuid.uuid4())
        if user_id not in user_registry.keys():
            user_registry[user_id] = base_amount
            return user_id
    raise Exception("failed to create new account. is the world collapsed?")


def check_account(user_id, user_registry):
    balance = user_registry.get(user_id, None)
    if balance is None:
        status = "not_found"
    elif balance < 0:
        status = "overdrawn"
    else:
        status = "ok"
    result = {"status": status, "balance": balance, "account": user_id}
    return result


def pay_amount(amount: float, user_id, target_id, user_registry, target_registry):
    reason = []
    status = "unknown"
    if amount > 0:
        user_info = check_account(user_id, user_registry)
        target_info = check_account(target_id, target_registry)
        user_status = user_info["status"]
        target_status = target_info["status"]
        if user_status != "ok":
            reason.append(f"user account {user_id} has invalid state {user_status}")
        if target_status != "ok":
            reason.append(f"target account {user_id} has invalid state {target_status}")
        if reason == []:
            user_balance = user_info["balance"]
            # target_balance = target_info['balance']
            user_balance_after = user_balance - amount
            if user_balance_after > 0:
                status = "ok"
                user_registry[user_id] -= amount
                target_registry[target_id] += amount
            else:
                reason.append(
                    f"user account {user_id} has insufficient balance {user_balance} (needed: {amount})"
                )
        else:
            status = "invalid_transfer"
    else:
        status = "invalid_amount"
    result = {"status": status, "amount": amount, "reason": reason}
    return result


def put_into_bank(user_id, amount, user_registry, bank_registry):
    result = pay_amount(amount, user_id, user_id, user_registry, bank_registry)
    return result


def extract_from_bank(user_id, amount, user_registry, bank_registry):
    result = pay_amount(amount, user_id, user_id, bank_registry, user_registry)
    return result


def parse_command_to_components(command: str):
    command_components = command.strip().split(" ")
    command_components = [c.strip() for c in command_components]
    command_components = [c.lower() for c in command_components if len(c) > 0]
    return command_components


def parse_amount(components: list[str]):
    amount = components.pop(0)
    amount = float(amount)
    return amount


import inspect


def construct_command_excutor(executor):
    sig = inspect.signature(executor)
    parameter_names = list(sig.parameters.keys())

    def command_executor(components: list[str], context: dict[str, str]):
        kwargs = {
            pname: parse_amount(components) if pname == "amount" else context[pname]
            for pname in parameter_names
        }
        ret = executor(*kwargs)
        return ret

    return command_executor


def pay_result_formatter():
    ...

command_handlers: dict[str, Callable[[list[str], dict[str, str]], dict]] = dict(
    pay=construct_command_excutor(pay_amount),
    check=construct_command_excutor(check_account),
    put_into_bank=construct_command_excutor(put_into_bank),
    extract_from_bank=construct_command_excutor(extract_from_bank),
)
result_formatters: dict[str, Callable[[dict,dict], dict]] = dict(pay=pay_result_formatter)


def clerk(command: str, context: dict[str, str]):
    command_components = parse_command_to_components(command)
    ret = ...
    if len(command_components) >= 2:
        comp_0 = command_components[0]
        rest_of_components = command_components[1:]
        handler = command_handlers.get(comp_0, None)
        formatter = result_formatters.get(comp_0, None)
        if handler:
            ret = handler(rest_of_components, context)
            if formatter:
                ret = formatter(ret, context)
        else:
            ...
    else:
        ...
    return ret
