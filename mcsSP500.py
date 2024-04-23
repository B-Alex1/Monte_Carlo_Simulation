"""
Monte Carlo Simulation of Amazon Stock Prices
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def price_calculation(S0, mean, sigma, w, t):
	"""
	S0: initial stock price
	mean: annualized drift (mean) of the stock returns
	sigma: annualized volatility of the stock returns
	w: Brownian motion value
	t: date (current_day / total_days)
	
	return: simulated stock price
	"""
	return S0 * np.exp((mean - 0.5 * sigma**2) * t + sigma * np.sqrt(t) * w)

def monte_carlo_simulation(S0, mean, sigma, T, N):
	"""
	S0: initial stock price
	mean: annualized drift (mean) of the stock returns
	sigma: annualized volatility of the stock returns
	T: period of simulation
	N: number of simulations
	
    	return: double array of simulated stock prices
	"""
	dt = 1/252 #Delta time set at one day (assuming there are 252 trading days per year)
	num_days = int(T/dt)
	
	#Brownian motion
	#Double array of size [N (number of simulations)][num_days]. Used to assign Brownian value for each day of each simulation
	W = np.random.standard_normal(size=(N, num_days))
	W = np.cumsum(W, axis=1) * np.sqrt(dt)
	
	#Setting initial Brownian motion of each simulation to 0 so that the inital stockprice is the same as the current one
	W[:, 0] = 0
	
	#Initializing double array of prices
	prices = np.zeros_like(W)
	
	#Setting initial price of each simulation to the most recent historical price
	prices[:, 0] = S0
	
	#Running simulation
	for i in range(N):
		for j in range(1, num_days):
			prices[i, j] = price_calculation(S0, mean, sigma, W[i, j], j*dt)
	return prices
	
def plot_stock_simulation(stock_data, simulated_prices, title, T):
	"""
	stock_data: Historical stock data
	simulated_prices: Result of the Monte Carlo simulations
	title: Title of the graph
	T: period of simulation
	
	Plot the data into 3 seperate graphs:
	1- Historical data of the stock price
	2- Historical data of the stock price followed by the simulated data
	3- Simulated data of the stock price
	"""
	#Creation of the new time index from today until the same date next year
	new_index = stock_data.index[len(stock_data.index)-(T*252):] + pd.DateOffset(years=T)

	plt.figure(figsize=(15, 6))

	#Plotting historical stock data
	plt.subplot(3, 1, 1)  # Subplot 1
	plt.plot(stock_data.index, stock_data.values, color='blue')
	plt.title('Historical Stock Data')
	plt.xlabel('Date')
	plt.ylabel('Stock Price in USD')

	#Plotting historical and simulated stock prices combined
	plt.subplot(3, 1, 2)  # Subplot 2
	plt.plot(stock_data.index, stock_data.values, label='Historical Stock Price', color='blue')
	plt.plot(new_index, simulated_prices.T, lw=0.5)
	plt.title('Simulated Stock Prices')
	plt.xlabel('Date')
	plt.ylabel('Stock Price in USD')
	plt.legend()

	#Plotting only simulated stock prices
	plt.subplot(3, 1, 3)  # Subplot 3
	plt.plot(new_index, simulated_prices.T, lw=0.5)
	plt.title('Only Simulated Stock Prices')
	plt.xlabel('Date')
	plt.ylabel('Stock Price in USD')

	plt.suptitle(title)
	plt.tight_layout()
	plt.show()

#Defining stock symbol and data collection time period
stock_symbol = '^GSPC'
start_date = datetime.now() - timedelta(days=20*365) #20 years worth of data
end_date = datetime.now()

#Retrieving historical stock data
stock_data = yf.download(stock_symbol, start=start_date, end=end_date)['Adj Close']
stock_data.to_csv("SP500_data.csv")
print("Data exported successfully.")

#Calculating parameters for Monte Carlo simulation based on the stock data
returns = np.log(stock_data / stock_data.shift(1)) #Log returns
mean = returns.mean() * 252 #Mean
sigma = returns.std() * np.sqrt(252) #Volatility
S0 = stock_data.iloc[-1] #Most recent stock price
T = 1 #Period for simulation (in years)

#Running Monte Carlo simulation
N = 100  # number of simulations
simulated_prices = monte_carlo_simulation(S0, mean, sigma, T, N)

#Plotting data
plot_stock_simulation(stock_data, simulated_prices, 'Historical and Simulated Stock Prices', T)
