from __future__ import annotations
import pandas as pd
import numpy as np
import matplotlib.pylab as plt


class Trades:
    def __init__(self,
            trade_decisions:np.array,
            step:int=0):
        self._trade_decisions = trade_decisions
        self._fitness = 0.0
        self._step_used = step

    def set_fitness(self,value:float):
        self._fitness = value

    def get_trade_decisions(self)->np.array:
        return self._trade_decisions
    
    def get_fitness(self)->float:
        return self._fitness

    def get_step(self)->int:
        return self._step_used


class LabelData:
    def __init__(self,
            buy_data:np.array,
            sell_data:np.array,
            starting_funds:float=100.0):

        self._max_population = 1000
        
        self._buy_data = buy_data
        self._sell_data = sell_data
        self._funds = starting_funds
        self._population = []


    def shift_prices_timesteps(self,data:np.array,timesteps:int=2)->np.array:
        """Simulate the delay between action and execution."""
        shifted_prices = data[2:]
        shifted_prices = np.append(shifted_prices,np.repeat(np.nan,2))
        return shifted_prices[~np.isnan(shifted_prices)]
        

    def score_program(self,decisions:Trades):
        dollar_balance = self._funds
        coin_balance = 0.0
        current_balance = dollar_balance

        shifted_buy = self.shift_prices_timesteps(self._buy_data)
        shifted_sell = self.shift_prices_timesteps(self._sell_data)

        decision_set = decisions.get_trade_decisions()
        index = 0
        while index < shifted_buy.shape[0]:
            buy_price = shifted_buy[index]
            sell_price = shifted_sell[index]
            decision = decision_set[index]
            index+=1

            if decision == "BUY":
                coin_balance = (dollar_balance / buy_price)*0.999
                continue

            if decision == "SELL":
                current_balance = (coin_balance * sell_price)*0.999
                dollar_balance = current_balance
                continue


        decisions.set_fitness(current_balance - self._funds)

        
    def generate_trade_data(self,step:int):        
        """Generates trades based on the step parameters and the data provided.
        the step decides what chunk of time we should look at to evaluate a 
        buy and sell decision.
        
        Returns an array of BUY|SELL|HOLD for each timestamp."""
        shifted_buy = self.shift_prices_timesteps(self._buy_data)
        shifted_sell = self.shift_prices_timesteps(self._sell_data)

        decisions = []
        index = 0
        while index+step < shifted_sell.shape[0]:
            while len(decisions) < index+step:
                decisions.append("HOLD")

            buy_points = shifted_buy[index:index+step]
            sell_points = shifted_sell[index:index+step]
            
            best_buy = 0
            best_sell = 1
            best_diff = sell_points[best_sell] - buy_points[best_buy]
            for i in range(1,buy_points.shape[0]-2):
                sells = sell_points[i+1:-1]
                buys = np.repeat(buy_points[i],sells.shape[0])
                diffs = sells-buys

                candidate = diffs.argmax()
                if diffs[candidate] > best_diff:
                    best_diff = diffs[candidate]
                    best_buy = i
                    best_sell = i+1+candidate

            decisions[best_buy+index] = "BUY"
            decisions[best_sell+index] = "SELL"
            index = best_sell+index

        while len(decisions) < shifted_sell.shape[0]+2:
            decisions.append("HOLD")

        return np.array(decisions)


    def find_labelings(self):
        steps = np.random.randint(
            low=30,
            high=min(120*self._max_population,2000),
            size=self._max_population)
        for step in steps:
            print(F"Generating data with step {step}")
            labels = self.generate_trade_data(step=step)
            self._population.append(Trades(trade_decisions=labels,step=step))

        for pop in self._population:
            self.score_program(pop)

        best_idx = 0
        best_fitness = 0
        for idx,pop in enumerate(self._population):
            if pop.get_fitness() > best_fitness:
                best_fitness = pop.get_fitness()
                best_idx = idx
        
        best = self._population[best_idx]
        
        trades_conducted = [t for t in best.get_trade_decisions() if t in ["BUY","SELL"]]
        print(F"Number of trades conducted: {len(trades_conducted)/2}")
        print(F"Best Member Fitness(profit): {best.get_fitness()}")
        print(F"Best Member step used: {best.get_step()}")

        output = self._buy_data.reshape(self._buy_data.shape[0],1)
        output = np.append(output,
            self._sell_data.reshape(self._buy_data.shape[0],1),axis=1)
        
        decision_set = best.get_trade_decisions()
        decision_set = decision_set.reshape(self._buy_data.shape[0],1)
        output = np.append(output,decision_set,axis=1)

        df = pd.DataFrame(output,columns=["best_ask","best_bid","decision"])
        df.to_csv("report.csv")
        
        plot_against = range(df["best_ask"].values.shape[0])
        colors = []
        sizes = []
        for d in best.get_trade_decisions():
            if d == "SELL":
                colors.append("red")
                sizes.append(30.0)
            elif d == "BUY":
                colors.append("blue")
                sizes.append(30.0)
            else:
                sizes.append(0.5)
                colors.append("gray")
        plt.scatter(plot_against,self._buy_data,c=colors,s=sizes)
        plt.legend()
        plt.show()


def test():
    data = pd.read_csv("ticker.csv")
    data = data.iloc[:40000]  
    # buy at ask, sell at bid
    lb = LabelData(
        buy_data=data["best_ask"].values,
        sell_data=data["best_bid"].values)
    lb.find_labelings()


if __name__ == "__main__":
    test()