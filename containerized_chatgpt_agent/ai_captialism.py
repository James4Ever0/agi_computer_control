# there must be some smaller groups called "company" evolved in the process.
# the total amount of currency remained unchanged.
import uuid

total_amount = 1e5
population_limit = 100

account_registry = {}


def create_account(base_amount=100):
    if len(account_registry) > population_limit:
        raise Exception(
            "too many accounts: %d; limit: %d" % (len(account_registry), base_amount)
        )
    while True:
        user_id = str(uuid.uuid4())
        if user_id not in account_registry.keys():
            account_registry[user_id] = base_amount
            return user_id


def check_account(user_id):
    ...


def pay_amount(amount, user_id, target_id):
    ...


def put_into_bank(user_id, amount):
    ...


def extract_from_bank(user_id, amount):
    ...
