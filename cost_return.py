import numpy as np

class cost_return:

    def __init__(self, new_port, ret):
        self.old_port = {}
        self.new_port = new_port
        self.old_stock = []
        self.new_stock = []
        self.add_weight = 1
        self.ret = ret
        self.cap = [10000000]
        self.cap_change()

    def update_old_port(self): #last week new portfolio become this week's old portfolio
        self.old_port = self.new_port
        return None

    def update_new_port(self, new_port): #udpate this week's new portfolio
        self.new_port = new_port
        return None

    def update_old_stock(self): #find the stocks that are still in the portfolio this week
        self.old_stock = list(self.old_port.keys() & self.new_port.keys())
        return None

    def update_new_stock(self): #find the new stocks in the portfolio
        self.new_stock = list(set(self.new_port.keys()) - set(self.old_port.keys()))
        return None

    def update_ret(self, ret): #the correponding returns of the stocks in that week
        self.ret = ret
        return None

    def old_stock_cost(self): #only the stocks with increased weight count towards the additional weight to calculate transaction cost
        add_weight = 0
        dict_value = list(self.old_port.values())
        nav_change = np.dot(np.array(self.ret), np.array(dict_value)) + 1
        for i in self.old_stock:
            index = list(self.new_port.keys()).index(i)
            change = self.new_port.get(i) - self.old_port.get(i)*(self.ret[index]+1)/nav_change
            if change > 0:
                add_weight += change
        return add_weight

    def new_stock_cost(self): #weight of new stocks in the portfolio
        add_weight = 0
        for i in self.new_stock:
            add_weight += self.new_port.get(i)
        return add_weight

    def cap_change(self): #find the portfolio size change due to transaction cost and return
        cost = self.cap[-1] * self.add_weight * 0.001
        dict_value = self.new_port.values()
        weight = []
        for i in range(len(dict_value)):
            weight.append(float([x for x in dict_value][i]))
        profit = np.dot(np.array(self.ret), np.array(weight))
        self.cap.append((self.cap[-1] - cost) * (1 + profit))
        return None

    def update_all(self, new_port, ret): #perform the above functions
        self.update_old_port()
        self.update_new_port(new_port)
        self.update_old_stock()
        self.update_new_stock()
        self.update_ret(ret)
        self.add_weight = self.old_stock_cost() + self.new_stock_cost()
        self.cap_change()
        return None
