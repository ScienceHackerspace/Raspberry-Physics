import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import leastsq

#open the file of your choice

fopen=open('foupend3.tsv','r')

#get the initial time

init_line=fopen.readline()
init_time=float(init_line.split()[0])

#tabulate the time values in seconds

time_data=[]
fopen.seek(0)
for jj in fopen:
    line=jj.split()
    time_data.append((float(line[0])-init_time)/1000)

def tabulate_data(col,f):
    result=[]
    f.seek(0)
    for ii in f:
        if len(ii) < 85:
            break
        line=ii.split()
        result.append(float(line[col]))
    return result

def average_data(data_set,time_set,avg_range):
    new_time=[]
    new_data=[]
    
    for k in range(avg_range,len(data_set)-avg_range):
        cur_data=[]
        cur_time=[]
        for m in range(k-avg_range,k+avg_range+1):
            cur_data.append(data_set[m])
            cur_time.append(time_set[m])
        data_avg=sum(cur_data)/(2*avg_range+1)
        time_avg=sum(cur_time)/(2*avg_range+1)
        new_time.append(time_avg)
        new_data.append(data_avg)
    new_time=np.array(new_time)
    new_data=np.array(new_data)
    return (new_time, new_data)

time_data=np.array(time_data)
z_data=tabulate_data(6,fopen)
z_data=np.array(z_data)/396.25

start_time=float(raw_input('Beginning time? '))
end_time=float(raw_input('End time? '))

#find indicies closest to requested times

start_array=abs(time_data-start_time) 
start_index=np.argmin(start_array) #this will find the index of the entry with the smallest difference from the requested time
end_array=abs(time_data-end_time)
end_index=np.argmin(end_array) 

#truncate the zdata and time data using above indicies

raw_z_data=z_data[start_index:end_index]
raw_time_data=time_data[start_index:end_index]

std_amp = 3*np.std(z_data)/np.sqrt(2) #use the root mean square method to estimate the amplitude

[time_data,z_data]=average_data(raw_z_data,raw_time_data,0)

max_accel=max(z_data)
min_accel=min(z_data)

start = time_data[0]
stop = time_data[-1]
num = len(time_data)

sp = (np.fft.fft(z_data)) #this generates an array containing the prevelance of certain frequencies
freq = np.fft.fftfreq(time_data.shape[-1])*num/(stop-start) #this contains the values of the frequencies

max_len=min([len(freq),len(sp)])

fig = plt.figure() #open a new figure
ax1 = fig.add_subplot(2,1,1) #add a subplot in the upper half
ax1.plot(freq[1:max_len], (sp.real[1:max_len]))

mainfreq=np.abs(freq[np.argmax(sp.real[1:])])
data_average=sum(z_data)/len(z_data)

plt.xlim([0,mainfreq*4]) #change the displayed domain to make it look nice

optimize_func = lambda x: x[0]*np.sin(x[1]*time_data+x[2])*np.exp(-x[3]*time_data) + x[4] - z_data
est_std,est_ang_freq, est_phase,est_b2m, est_mean = leastsq(optimize_func, [std_amp, 2*np.pi*mainfreq,0,0, data_average])[0]

data_fit = est_std*np.sin(est_ang_freq*time_data+est_phase)*np.exp(-est_b2m*time_data) + est_mean
guess_fit = std_amp*np.sin(2*np.pi*mainfreq*time_data+1.7) + data_average
        
fig.suptitle(format('y=%.5f*e^(-%.5f*t)*sin(%.5f*t+%.5f)+%.5f'%(est_std,est_b2m,est_ang_freq,est_phase,est_mean)))

print mainfreq
print est_ang_freq/(2*np.pi)

ax3=fig.add_subplot(2,1,2)
#ax3.plot(raw_time_data,raw_z_data,'b')
ax3.plot(time_data,z_data,'y',label='raw_data')
ax3.plot(time_data,data_fit,'c',label='fitted curve')
#ax3.plot(time_data,guess_fit,'r')
#ax3.legend()
plt.xlim([start_time,end_time])
plt.show()
