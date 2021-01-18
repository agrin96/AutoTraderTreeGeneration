from __future__ import annotations
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

from Indicators import *

def array_shift(arr:np.array,num:int)->np.array:
    new = np.empty_like(arr)
    if num >= 0:
        new[:num] = np.nan
        new[num:] = arr[:-num]
    else:
        new[num:] = np.nan
        new[:num] = arr[-num:]
    return new


class TradesSet:
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


class MovingAverageTrades:
    def __init__(self,
            buy_data:np.array,
            sell_data:np.array,
            derivative_points:int,
            kama_signal_ma:int,
            starting_funds:float=100.0):
        
        self._buy_data = buy_data
        self._sell_data = sell_data
        self._funds = starting_funds

        self._fast_kama = kaufman_adaptive_moving_average(
            data=self._buy_data,
            er_period=60*10,
            fast_ema=2,
            slow_ema=30)

        self._slow_kama = kaufman_adaptive_moving_average(
            data=self._buy_data,
            er_period=60*10,
            fast_ema=2*3,
            slow_ema=30*3)

        # The difference will be our signal. We can see how far above
        # and below the zero line we are.
        self._kama_diff = self._fast_kama - self._slow_kama

        # This is a signal line for the difference.
        self._kama_signal = simple_moving_average(
                data=self._kama_diff,
                window=kama_signal_ma)

        # Find the derivative of the kama signal to use in rate of change
        # analysis
        self._kama_derivative =\
            np.divide(
                np.subtract(
                    self._kama_diff,
                    array_shift(self._kama_diff,derivative_points)),
                derivative_points)


    def shift_prices_timesteps(self,data:np.array,timesteps:int=2)->np.array:
        """Simulate the delay between action and execution."""
        shifted_prices = data[2:]
        shifted_prices = np.append(shifted_prices,np.repeat(np.nan,2))
        return shifted_prices[~np.isnan(shifted_prices)]
        

    def score_trades(self,decisions:TradesSet):
        """Uses the decisions generated to generate a profit score by 
        executing all of the decisions and finding the final profit.
        """
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


    def generate_trade_data(self,signal_lookback:int=2)->List[str]:        
        """Generates trades based on the step parameters and the data provided.
        the step decides what chunk of time we should look at to evaluate a 
        buy and sell decision.
        
        Returns an array of BUY|SELL|HOLD for each timestamp."""
        decisions = ["HOLD"]*signal_lookback
        index = signal_lookback
        prev_decision = "SELL"

        while index < self._kama_signal.shape[0]:
            kama_signal = self._kama_signal[index]
            kam_signal_old = self._kama_signal[index-signal_lookback]
            kama_diff = self._kama_diff[index]
            kama_diff_old = self._kama_diff[index-signal_lookback]

            # Our difference is crossing its MA from the bottom
            # Our difference is crossing its MA from the top
            if kama_signal < kama_diff and kam_signal_old > kama_diff\
            or kama_signal > kama_diff and kam_signal_old < kama_diff:
                # Our short term line is below our long term so we BUY
                if kama_diff <= 0 and prev_decision == "SELL":
                    decisions.append("BUY")
                    prev_decision = "BUY"
                if kama_diff > 0 and prev_decision == "BUY":
                    decisions.append("SELL")
                    prev_decision = "SELL"
            else:
                decisions.append("HOLD")
            index += 1


        while len(decisions) < self._sell_data.shape[0]:
            decisions.append("HOLD")

        return np.array(decisions)


    def find_labelings(self,plot_low:int,plot_high:int):
        trades = TradesSet(
            trade_decisions=self.generate_trade_data(signal_lookback=3),
            step=0)
        self.score_trades(decisions=trades)

        trades_conducted = [t for t in trades.get_trade_decisions() if t in ["BUY","SELL"]]
        print(F"Number of trades conducted: {len(trades_conducted)/2}")
        print(F"Trade Performance (profit): {trades.get_fitness()}")

        output = self._buy_data.reshape(self._buy_data.shape[0],1)
        output = np.append(output,
            self._sell_data.reshape(self._buy_data.shape[0],1),axis=1)
        
        decision_set = trades.get_trade_decisions()
        decision_set = decision_set.reshape(self._buy_data.shape[0],1)
        output = np.append(output,decision_set,axis=1)

        df = pd.DataFrame(output,columns=["best_ask","best_bid","decision"])
        df.to_csv("moving_average_report.csv")
        
        plot_against = range(df["best_ask"].values.shape[0])
        colors = []
        sizes = []
        for d in trades.get_trade_decisions():
            if d == "SELL":
                colors.append("red")
                sizes.append(30.0)
            elif d == "BUY":
                colors.append("blue")
                sizes.append(30.0)
            else:
                sizes.append(0.5)
                colors.append("gray")
        fig,axs = plt.subplots(3)

        axs[0].scatter(plot_against[:plot_high-plot_low],self._buy_data[plot_low:plot_high],c=colors[plot_low:plot_high],s=sizes[plot_low:plot_high])
        axs[0].plot(self._fast_kama[plot_low:plot_high],color="purple",lineStyle="-",label="fast_kama")
        axs[0].plot(self._slow_kama[plot_low:plot_high],color="green",lineStyle="-",label="slow_kama")
        axs[0].legend()

        axs[1].plot(self._kama_diff[plot_low:plot_high],color="green",lineStyle="-",label="Kama diff")
        axs[1].plot(self._kama_signal[plot_low:plot_high],color="cyan",lineStyle="-",label="kama diff SMA")
        axs[1].plot(np.zeros_like(self._kama_signal)[plot_low:plot_high],color="black",lineStyle="-",label="signal")
        axs[1].legend()

        ddx_ma = simple_moving_average(self._kama_derivative,60*3)
        # axs[2].plot(self._kama_derivative[plot_low:plot_high],color="red",lineStyle="-",label="d/dx Kama diff signal")
        axs[2].plot(ddx_ma[plot_low:plot_high],color="green",lineStyle="-",label="d/dx Kama diff signal sma")
        axs[2].plot(np.zeros_like(self._kama_signal)[plot_low:plot_high],color="black",lineStyle="-",label="signal")
        axs[2].legend()
        plt.show()


def test():
    data = pd.read_csv("ticker.csv")    
    # buy at ask, sell at bid
    lb = MovingAverageTrades(
        buy_data=data["best_ask"].values,
        sell_data=data["best_bid"].values,
        kama_signal_ma=60*6,
        derivative_points=2)
    lb.find_labelings(plot_low=0000,plot_high=10000)


if __name__ == "__main__":
    test()