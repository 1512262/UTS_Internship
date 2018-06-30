import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.patches as plp
from scipy.ndimage.filters import gaussian_filter1d
from scipy.signal import argrelextrema
import time

my_filter = np.array([-1,1]) 
def SMA(x,period):
	denominator = period
	numerator =  np.zeros((1,period))
	for i in range(0,period):
		numerator[0,i] = 1
	SMA_filter = np.ravel(numerator/denominator)
	
	return np.correlate(x,SMA_filter,"same")

def Fibo(Time,Close):
	dim = np.shape(Time)[0]
	mid_filter = np.array([-1,2])


	t = Time[1:].reshape((dim-1,1))
	c = Close[1:].reshape((dim-1,1))
	coor = np.c_[t,c]

	vec_x = np.abs(np.correlate(Time,my_filter,"same")).reshape((dim,1))
	vec_y = np.abs(np.correlate(Close,my_filter,"same")).reshape((dim,1))
	t_o = np.correlate(Time,mid_filter,"same")
	t_o = t_o[1:]
	vec = np.c_[vec_x,vec_y]
	radius = np.linalg.norm(vec,axis=1)[1:] 

	return coor, radius, t_o
def filter1(Time,Close,threshold):
	 # filter này là cái sau trừ cái trước
	res_1 = np.abs(np.correlate(Close,my_filter,"same"))
	pos_t = res_1>threshold

	Close_t = Close[pos_t]
	Time_t = Time[pos_t]
	return Time_t,Close_t

def filter2(Time,Close):
	res_2 = np.correlate(Close,my_filter,"same")
	res_2_next = np.append(res_2[1:],0)
	pos = res_2*res_2_next < 0

	return Time[pos],Close[pos]
def main():
	#Getting Data
	path = "D:/OneDrive - khoavanhoc.edu.vn/UTS/Data/"
	data = pd.read_csv(path+"Gdax_ETHBTC_d.csv")
	Time = data['Date'].values
	Close = data['Close'].values
	length = np.shape(Time)[0]
	fig, ax = plt.subplots()

	#Calculating SMA
	SMA_12 = SMA(Close,12)[24:]
	SMA_24 = SMA(Close,24)[24:]


	epsilon = 0.00002
	SMA_pos = np.abs(SMA_24-SMA_12)<epsilon

	Time_meet= Time[24:][SMA_pos]

	#Taking a part of data for calculation and visualzation
	N= 90
	for i in range (length//N):
		print(i)
		Time_t = Time[i*N:(i+1)*N]
		Close_t =Close[i*N:(i+1)*N]	
		maxTime,minTime = np.max(Time_t),np.min(Time_t)


		price = ax.plot(Time_t,Close_t,'red',label = 'price')
		smooth_Close = gaussian_filter1d(Close_t,sigma=1)

		pos_max = np.ravel(argrelextrema(smooth_Close, np.greater))
		pos_min = np.ravel(argrelextrema(smooth_Close, np.less))

		pos= np.sort(np.concatenate([pos_max,pos_min]))

		threshold = 0.001
		Time_t = Time_t[pos]
		Close_t =Close_t[pos]

		Time_t,Close_t= filter1(Time_t,Close_t,threshold)
		Time_t,Close_t=filter2(Time_t,Close_t)

		extreme = ax.plot(Time_t,Close_t,'bo',label = 'extreme point')
		trending_line = ax.plot(Time_t,Close_t,'--',label = 'trending line', color='black')

		xlines = []
		for value in Time_meet:
			if(value>=minTime and value<=maxTime):
				xline = plt.axvline(x=value,color ='y')
				xlines.append(xline)
			elif value>maxTime:
				break 


		ax.legend()

		plt.xlabel("Time")
		plt.ylabel("Value")
		name = "./Picture/"+str(int((time.time()*100)))+".png"
		fig.savefig(name)
		fig.clf()
		

if __name__ == '__main__':
	main()