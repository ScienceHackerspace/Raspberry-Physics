# graph plot of logged data

import matplotlib.pyplot as plt
%matplotlib inline

import numpy as np
import math

filenum=raw_input("Num of file?: ")
filename="logdata" + filenum + ".txt"
#selection=input("Magnetic Field-0,Accel-1,RotationalAccel-2: ")
selection=1
avg=int(raw_input("Averaging range?: "))
peak_allow=float(raw_input("Maximum Noise Allowance (decimal)?: "))
n=int(raw_input("Number of Times to Average? "))
#dim_choice=int(raw_input("x,y,z? (1,2,3): "))

x_col=int((3*float(selection)+1))
y_col=int((3*float(selection)+2))
z_col=int((3*float(selection)+3))

fopen=open(filename,"r+")
init_line=fopen.readline()
init_time=float(init_line.split()[0])
time_data=[]
fopen.seek(0)
for jj in fopen:
    line=jj.split()
    time_data.append((float(line[0])-init_time)/1000)

def tabulate_data(col,f):
    result=[]
    f.seek(0)
    for ii in f:
        line=ii.split()
        result.append(float(line[col]))
    return result

def peak_remover(data_list,allowance):
    no_peaks=[]
    for i in range(2,len(data_list)-2):
        i_data_set=[]
        for j in range(i-2,i+3):
            i_data_set.append(data_list[j])
        i_avg=sum(i_data_set)/5
        if i_avg==0 or abs((data_list[i]-i_avg)/i_avg)<allowance:
            no_peaks.append((sum(i_data_set)-data_list[i])/4)
        else:
            no_peaks.append(data_list[i])
    return no_peaks
            
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
    return (new_time, new_data)

def find_normal(x_data,y_data,z_data,time_set):
    print 'Your data spans from '+str(time_set[0])+' to '+str(time_set[-1])
    lower_bound=float(raw_input('Please pick starting sample time: '))
    upper_bound=float(raw_input('Please pick end sample time: '))
    time_set=np.array(time_set)
    
    start_array=abs(time_set-lower_bound)
    start_index=np.argmin(start_array)
    end_array=abs(time_set-upper_bound)
    end_index=np.argmin(end_array)
    
    xtot=0
    ytot=0
    ztot=0
    for k in range(start_index,end_index):
        xtot=xtot+x_data[k]
        ytot=ytot+y_data[k]
        ztot=ztot+z_data[k]
    xavg=xtot/(end_index-start_index)
    yavg=ytot/(end_index-start_index)
    zavg=ztot/(end_index-start_index)
    #
    #Check to see if end-start is the right number
    
    normal=[yavg*zavg,-xavg*zavg,0]
    mag_r=math.sqrt(xavg**2+yavg**2+zavg**2)
    costheta=zavg/mag_r
    theta=math.acos(costheta)
    return (normal,theta)

def rotate_about(vector,axis,theta):
    axis=np.array(axis)
    vector=np.array(vector)
    mag_axis=math.sqrt((axis[0]**2)+(axis[1]**2)+(axis[2]**2))
    axis=axis/mag_axis
    dot=np.dot(vector,axis)
    result=[axis[0]*dot*(1-np.cos(theta))+vector[0]*np.cos(theta)+(-axis[2]*vector[1]+axis[1]*vector[2])*np.sin(theta),
            axis[1]*dot*(1-np.cos(theta))+vector[1]*np.cos(theta)+(axis[2]*vector[0]-axis[0]*vector[2])*np.sin(theta),
            axis[2]*dot*(1-np.cos(theta))+vector[2]*np.cos(theta)+(-axis[1]*vector[0]+axis[0]*vector[1])*np.sin(theta)]     
    return result

def area_under(data_set,time_set,start_lim,end_lim):
    data_set=np.array(data_set)
    time_set=np.array(time_set)
    
    start_array=abs(time_set-start_lim)
    start_index=np.argmin(start_array)
    end_array=abs(time_set-end_lim)
    end_index=np.argmin(end_array)
    
    area_array=[0]
    area=0
    for i in range(start_index,end_index-1):
        width_i=time_set[i+1] - time_set[i]
        height_i=(data_set[i+1]+data_set[i])/2
        area_i=width_i*height_i
        area=area+area_i
        area_array.append(area)
    corresp_time=time_set[start_index:end_index]
    print 'Using the closest points, the area from '+str(time_set[start_index])+' to '+str(time_set[end_index])+' is:'
    print area
    return (area_array,corresp_time)

raw_x_data=tabulate_data(x_col,fopen)
raw_y_data=tabulate_data(y_col,fopen)
raw_z_data=tabulate_data(z_col,fopen)

(normal_vec,theta_rad)=find_normal(raw_x_data,raw_y_data,raw_z_data,time_data)

rotated_x=[]
rotated_y=[]
rotated_z=[]
for i in range(0,len(raw_x_data)):
    x_vec_i=[raw_x_data[i],0,0]
    y_vec_i=[0,raw_y_data[i],0]
    z_vec_i=[0,0,raw_z_data[i]]
    
    x_vec_i=rotate_about(x_vec_i,normal_vec,theta_rad)
    y_vec_i=rotate_about(y_vec_i,normal_vec,theta_rad)
    z_vec_i=rotate_about(z_vec_i,normal_vec,theta_rad)
    
    rotated_x.append(x_vec_i[0]+y_vec_i[0]+z_vec_i[0])
    rotated_y.append(x_vec_i[1]+y_vec_i[1]+z_vec_i[1])
    rotated_z.append(x_vec_i[2]+y_vec_i[2]+z_vec_i[2])

grav_mult=sum(rotated_z)/(len(rotated_z)*9.81)
rotated_x=np.array(rotated_x)/grav_mult
rotated_y=np.array(rotated_y)/grav_mult
rotated_z=np.array(rotated_z)/grav_mult
#317.654

no_peak_x=peak_remover(rotated_x,peak_allow)
no_peak_y=peak_remover(rotated_y,peak_allow)
no_peak_z=peak_remover(rotated_z,peak_allow)

for b in range(0,n):
    [xtime,x_data]=average_data(no_peak_x,time_data,avg)
    [ytime,y_data]=average_data(no_peak_y,time_data,avg)
    [ztime,z_data]=average_data(no_peak_z,time_data,avg)

integrate_l1=float(input('First limit? '))
integrate_l2=float(input('Second limit? '))

[xvelocity,vtime]=area_under(x_data,xtime,integrate_l1,integrate_l2)
[xposition,ptime]=area_under(xvelocity,vtime,integrate_l1,integrate_l2)

final_position=[xposition[-1], xposition[-1]]
final_position_time=[integrate_l1,integrate_l2]

fig = plt.figure()
result = fig.add_subplot(111)
plt.plot(xtime,x_data,'c',vtime,xvelocity,'m',ptime,xposition,'g', final_position_time,final_position,'r-')#,time_data,rotated_z,'r')
string='logdata'+str(filenum)+'; averaged '+str(n)+' times with range '+str(avg)+'; peak allowance: '+str(peak_allow)
fig.suptitle(string)
plt.xlim([0.9*integrate_l1,1.1*integrate_l2])
