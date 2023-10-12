# there must be some smaller groups called "company" evolved in the process.
# the total amount of currency remained unchanged.
import uuid

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
