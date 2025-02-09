# in order to earn money, one may have to know how others spend it.
# it needs to both earn and spend.

# if an ai got 1 million what it will do?
# what about earning?

import time

class FinancialReward:
    def __init__(self, init_money:float, decrease_rate:float):
        self.money=init_money
        self.decrease_rate=decrease_rate
        self.state = 'active'
        self.born_time = time.time()
    def get_step(self):
        ...