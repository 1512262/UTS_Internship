import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d
from scipy.signal import argrelextrema


def EMA(x,period):
	alpha = 2.0/(period+1)
	denominator = (1-(1-alpha)**period)/alpha  
	numerator =  np.zeros((1,period))
	for i in range(0,period):
		numerator[0,i] = (1-alpha)**i
	EMA_filter = np.ravel(numerator/denominator)
	
	return np.correlate(x,EMA_filter,"same")

def SMA(x,period):
	denominator = period
	numerator =  np.zeros((1,period))
	for i in range(0,period):
		numerator[0,i] = 1
	SMA_filter = np.ravel(numerator/denominator)
	
	return np.correlate(x,SMA_filter,"same")

def main():
	path = "D:/OneDrive - khoavanhoc.edu.vn/UTS/Data/"
	data = pd.read_csv("A year.csv")
	x = data['Date'].values
	Close = data['Close'].values
	N= 40
	x = x[N:5*N]
	Close =Close[N:5*N]
	fig, ax = plt.subplots()
	
	# High = data['High'].values
	# Low = data['Low'].values
	# dim = np.shape(Close)[0]


	price = ax.plot(x,Close,'red',label = 'price')
	TODAY = ax.plot(x[-1],Close[-1],'bo',label = 'Today')
	p270517 = ax.plot(x[-5],Close[-5],'go',label = '27/05/2017')

	# smooth_Close = gaussian_filter1d(Close,sigma=1)


	# smooth_price = ax.plot(x,smooth_Close,'green',label = 'smooth price')

	# pos_max = np.ravel(argrelextrema(smooth_Close, np.greater))
	# pos_min = np.ravel(argrelextrema(smooth_Close, np.less))

	# pos_max = np.ravel(argrelextrema(Close, np.greater))
	# pos_min = np.ravel(argrelextrema(Close, np.less))

	# idx_min= np.logical_and(pos_min>50,pos_min< dim -50)
	# idx_max= np.logical_and(pos_max>50,pos_max< dim -50)

	# ax.plot(x[pos_min], Close[pos_min],'bo',label='local minimum')
	# ax.plot(x[pos_max], Close[pos_max],'ro',color='black', label='local maximum')

	# pos= np.sort(np.concatenate([pos_max,pos_min]))
	# idx_pos = np.logical_and(pos>50,pos< dim -50)
	

	# x_local = x[pos]
	# y_local = Close[pos]

	# extreme = ax.plot(x_local,y_local,'bo',label = 'extreme point')
	# trending_line = ax.plot(x_local,y_local,'r--',label = 'trending line', color='black')

	# #EMA
	# ema20 = ax.plot(x[50:-50],EMA(Close,20)[50:-50],'g--',label = 'EMA(20)')
	# ema50 = ax.plot(x[50:-50],EMA(Close,50)[50:-50],'b--',label = 'EMA(50)')

	# idx = np.argwhere(np.diff(np.sign(EMA(Close,20) - EMA(Close,50))) != 0).reshape(-1) + 0

	# idx_root = np.logical_and(idx>50,idx< dim -50)
	# ax.plot(x[idx[idx_root]],EMA(Close,50)[idx[idx_root]],'ro',label = 'intersection point of 2 EMA lines')

	#SMA
	# sma20 = ax.plot(x[24:-24],SMA(Close,12)[24:-24],'r--',label = 'SMA(12)',color = 'blue')
	# sma50 = ax.plot(x[24:-24],SMA(Close,24)[24:-24],'r--',label = 'SMA(24)',color = 'yellow')


	ax.legend()

	# plt.plot(x,High,'green')
	# plt.plot(x,Low,'blue')
	# plt.plot(x,Close,'yellow')
	plt.xlabel("Time")
	plt.ylabel("Value")
	plt.show()


if __name__ == '__main__':
	main()