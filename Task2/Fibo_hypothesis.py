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

def main():
	path = "D:/OneDrive - khoavanhoc.edu.vn/UTS/Data/"
	data = pd.read_csv(path+"Gdax_ETHBTC_d.csv")
	Time = data['Date'].values
	Close = data['Close'].values
	length = np.shape(Time)[0]

	#Calculating SMA
	SMA_12 = SMA(Close,12)[24:]
	SMA_24 = SMA(Close,24)[24:]


	epsilon = 0.00008
	SMA_pos = np.abs(SMA_24-SMA_12)<epsilon

	Time_meet= Time[24:][SMA_pos]
	
	N= 30
	# Time_t = Time[0*N:3*N]
	# Close_t =Close[0*N:3*N]

	# Time_t = Time[3*N:6*N]
	# Close_t =Close[3*N:6*N]

	# Time_t = Time[6*N:9*N]
	# Close_t =Close[6*N:9*N]

	# Time_t = Time[9*N:12*N]
	# Close_t =Close[9*N:12*N]

	# Time_t = Time[12*N:15*N]
	# Close_t =Close[12*N:15*N]

	# Time_t = Time[15*N:18*N]
	# Close_t =Close[15*N:18*N]

	# Time_t = Time[18*N:21*N]
	# Close_t =Close[18*N:21*N]

	Time_t = Time[21*N:24*N]
	Close_t =Close[21*N:24*N]
	maxTime,minTime = np.max(Time_t),np.min(Time_t)
	fig, ax = plt.subplots()
	
	
	
	price = ax.plot(Time_t,Close_t,'red',label = 'price')
	smooth_Close = gaussian_filter1d(Close_t,sigma=1)  #Xài Gaussian Filter. Cái này em đã nói rồi

	pos_max = np.ravel(argrelextrema(smooth_Close, np.greater))
	pos_min = np.ravel(argrelextrema(smooth_Close, np.less))


	pos= np.sort(np.concatenate([pos_max,pos_min]))
	
	threshold = 0.002
	Time_t = Time_t[pos]
	Close_t =Close_t[pos]

	Time_t,Close_t= filter1(Time_t,Close_t,threshold)
	Time_t,Close_t= filter2(Time_t,Close_t)

	start = np.array([Time_t[0],Close_t[0]])

	mid_filter = np.array([-1,2])

	
	coor, radius, t_o = Fibo(Time_t,Close_t)

	extreme = ax.plot(Time_t,Close_t,'bo',label = 'extreme point')
	trending_line = ax.plot(Time_t,Close_t,'--',label = 'trending line', color='black')



	
	size = np.shape(radius)[0]
	
	count = 0 
	for value in Time_meet:
		if(value>=minTime and value<=maxTime):
			if(count==0):
				xline = ax.axvline(x=value,color ='y',label='x = (SMA12==SMA24)')
			if(count==1):
				xline = ax.axvline(x=value,color ='y')
			count+=1
		elif value>maxTime:
			break 
	

	plt.xlabel("Time")
	plt.ylabel("Value")

	for i in range(size):
		

		# fc1000 = plp.Circle((coor[i,0],coor[i,1]),radius[i],color='g',fill=False)
		# fc764 = plp.Circle((coor[i,0],coor[i,1]),0.764*radius[i],color='g',fill=False)
		# fc618 = plp.Circle((coor[i,0],coor[i,1]),0.618*radius[i],color='g',fill=False)
		fc500 = plp.Circle((coor[i,0],coor[i,1]),0.5*radius[i],color='g',fill=False)
		fc382 = plp.Circle((coor[i,0],coor[i,1]),0.382*radius[i],color='g',fill=False)
		fc236 = plp.Circle((coor[i,0],coor[i,1]),0.236*radius[i],color='g',fill=False)
	
		if(i==0):
			sign = (coor[0,1]-start[1])/np.abs((coor[0,1]-start[1]))
		else:
			sign = (coor[i,1]-coor[i-1,1])/np.abs((coor[i,1]-coor[i-1,1]))

		
		fl1000 = plt.axhline(y=coor[i,1]-sign*radius[i])
		fl764 = plt.axhline(y=coor[i,1]-sign*0.764*radius[i])
		fl618 = plt.axhline(y=coor[i,1]-sign*0.618*radius[i])
		fl500 = plt.axhline(y=coor[i,1]-sign*0.5*radius[i])
		fl382 = plt.axhline(y=coor[i,1]-sign*0.382*radius[i])
		fl236 = plt.axhline(y=coor[i,1]-sign*0.236*radius[i])
		fl000 = plt.axhline(y=coor[i,1])

		length = t_o[i] -coor[i,0]

		flv1618 = plt.axvline(x=coor[i,0]+1.618*length)
		flv1000 = plt.axvline(x=coor[i,0]+length)
		flv1764 = plt.axvline(x=coor[i,0]+1.764*length)
		flv1500 = plt.axvline(x=coor[i,0]+1.5*length)
		flv1382 = plt.axvline(x=coor[i,0]+1.382*length)
		flv1236 = plt.axvline(x=coor[i,0]+1.236*length)

		
		
		# ax.add_artist(fc1000)
		# ax.add_artist(fc764)
		# ax.add_artist(fc618)
		ax.add_artist(fc500)
		ax.add_artist(fc382)
		ax.add_artist(fc236)
		
		ax.legend()
		name = "./Fibo_gird_Fibo_circle/"+str(int((time.time()*100)))+".png"
		fig.savefig(name)

		flv1618.remove()
		flv1000.remove()
		flv1764.remove()
		flv1500.remove()
		flv1382.remove()
		flv1236.remove()

		# fc1000.remove()
		# fc764.remove()
		# fc618.remove()
		fc500.remove()
		fc382.remove()
		fc236.remove()

		fl1000.remove()
		fl764.remove()
		fl618.remove()
		fl500.remove()
		fl382.remove()
		fl236.remove()
		fl000.remove()

	

	
	

		

if __name__ == '__main__':
	main()