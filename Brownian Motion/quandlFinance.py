
# data from Quandl

import quandl
import numpy as np
import matplotlib.pyplot as plt

start = "2016-01-01"
end = "2016-12-31"

df = quandl.get("WIKI/AMZN", start_date = start, end_date = end)

adj_close = df['Adj. Close']
time = np.linspace(1, len(adj_close), len(adj_close))

plt.plot(time, adj_close)

def daily_return(adj_close):
    returns = []
    for i in xrange(0, len(adj_close)-1):
        today = adj_close[i+1]
        yesterday = adj_close[i]
        daily_return = (today - yesterday)/today
        returns.append(daily_return)
    return returns

returns = daily_return(adj_close)

mu = np.mean(returns)*252.
sig = np.std(returns)*np.sqrt(252.)

print mu, sig

    


