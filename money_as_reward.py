# in order to earn money, one may have to know how others spend it.
# it needs to both earn and spend.

# if an ai got 1 million what it will do?
# what about earning?

import time

class TaxAccountant:
    def __init__(self, init_money:float, tax_rate:float, debug:bool = True):
        self.money = init_money
        self.debug = debug
        self.tax_rate = tax_rate
        # self.state = 'active'
        self.born_time = time.time()
        self.last_tax_time = time.time()

    def pay_tax(self):
        if self.state == "suspended": return 0
        current_time = time.time()
        tax_duration = current_time-self.last_tax_time
        living_tax = tax_duration * self.tax_rate
        if self.debug:
            print('money: %f tax: %f' % (self.money, living_tax))
        paied_tax = min(self.money, living_tax)
        self.money -= paid_tax
        self.last_tax_time = current_time
        return paid_tax

    @property
    def state(self):
        if self.money > 0: return "active"
        return "suspended"

def test():
    acc = TaxAccountant(3, 1)
    for _ in range(4):
        acc.pay_tax()
        time.sleep(1)
    assert acc.state == 'suspended'

if __name__ == '__main__':
    test()

# about the tax being deducted, we need to get the value