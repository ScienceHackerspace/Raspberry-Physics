#initiate data collection
import paramiko
init_pc_time=time.time()

ssh=paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

ssh.connect('192.168.43.67',port=22,username='pi',password='raspberry')

execute=ssh.exec_command("sudo python /home/pi/minimu9code/loggerstart2.py")

[a,b,c]=ssh.exec_command("date '+%s%N'")

pc_time1=time.time()-init_pc_time
pi_time1=b.readline()
time.sleep(1.5)


execute=ssh.exec_command("sudo python /home/pi/minimu9code/loggerstart2.py")

[d,e,f]=ssh.exec_command("date '+%s%N'")

pc_time2=time.time()-init_pc_time
pi_time2=e.readline()

print pc_time1
print pi_time1

print pc_time2
print pi_time2
