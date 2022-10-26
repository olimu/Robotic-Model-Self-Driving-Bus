import socket
import motorsmod as mm
import time
import sys
import numpy as np
import argparse
import motorsmod
from pidClass import PID
import qwiic_icm20948

debug = 1
#HOST_IP = '192.168.1.14' # server IP 3b
HOST_IP = '172.26.161.196' # server IP 3b
HOST_PORT = 4000

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind((HOST_IP, HOST_PORT))
if debug == 1:
  print("Server Started")
sock.listen(5)
conn, addr = sock.accept()

i = 0
# remember to get timeNew = time.time() to get latest time
def step(iterations, target, timeNew):
  pidYaw = PID(target, Kpy, Kiy, Kdy, 1)
  for i in range(iterations):
    timeOld = timeNew
    timeNew = time.time()
    dt = timeNew - timeOld
    measureCorrIMU()
    yawRateAvg = np.mean(yawRateData[-N_avg:])
    _, yaw, yawControlVal = pidYaw.update(yawRateAvg, dt, 0) # do not reset
    origy = yawControlVal
    yawControlVal = pidYaw.limit(0.45)
    print(round(timeNew - time0,6), round(dt,6), i, target, round(yawRateAvg,2), round(yaw,2), round(origy,2), round(yawControlVal,2))
    motorsmod.Spin(-yawControlVal)
  return(yawRateAvg, timeNew, dt)

def measureRawIMU():
  if (IMU.getAgmt()):
    yawRateData.append(IMU.gzRaw/(32768/250.0))

def measureCorrIMU():
  if (IMU.getAgmt()):
    yawRateData.append(IMU.gzRaw/(32768/250.0) - yawRateBias)
    
# calculates the initial yaw rate gyro static bias
def calculateGyroBias():
  global yawRateBias
  for i in range(N_bias):
    measureRawIMU()
  yawRateBias = np.mean(yawRateData[-N_bias:])
  # add N_avg more data so that average calc use the Bias corrected data
  for i in range(N_avg):
    measureCorrIMU()
    
motorsmod.allStop()
time.sleep(4)

yawRateData = []
N_avg = 17
N_bias = 50
time0 = time.time()
timeNew = time.time()

yaw = 0

Kpy = 0.045
Kiy = 0.0025
Kdy = 0.005

STDEV_THRESHOLD = 0.07

yawRateBias = 0

parser = argparse.ArgumentParser(description='Test ICM20948 IMU')
# tm: set default to I2C in next line
parser.add_argument('--spi', action='store_true', help='Use SPI interface instead of I2C')
args = parser.parse_args()

IMU = qwiic_icm20948.QwiicIcm20948()
if IMU.connected == False:
    print("The Qwiic ICM20948 device isn't connected to the system. \
          Please check your connection", file=sys.stderr)
    exit
IMU.begin()

calculateGyroBias() # calc init static bias; update global called yawRateBias
    
timeOld = timeNew
timeNew = time.time()

with conn:
  print('Connected by', addr)
  while True:
    data = conn.recv(1024)
    data = data.decode('utf-8')
    i = i + 1
    if debug == 1:
      print("Message from", str(addr), i)
      print("Message", data)

    if(data == "g"):
      #mm.forwardDrive(0.2)
      #time.sleep(0.2) # was 0.3, then 0.2, then commented out
      #mm.allStop()
      calculateGyroBias()
      timeNew = time.time()
      yawRateAvg, timeNew, dt = step(400, -90, timeNew)
      print('turn')
      mm.allStop()
      mm.forwardDrive(0.1)
      time.sleep(0.8) # was 0.3, then 0.7, then 1.0, then 0.7 again, then 0.8
      mm.allStop()
    if(data == "t"):
      print('turn 90 w/o gyro')
      mm.forwardDrive(0.2) # same as 'g'
      time.sleep(1.4) # more than 'g'
      mm.spinRight(0.7)
      time.sleep(0.9)
      mm.forwardDrive(0.1) # same as 'g'
      time.sleep(0.8) # same as 'g'
      mm.allStop()
    if(data == "n"):
      continue
    if(data == "w"):
      print('forward')
      mm.forwardDrive(0.2)
      time.sleep(0.17)
      mm.allStop()
    if(data == "s"):
      print('reverse')
      mm.reverseDrive(0.15)
      time.sleep(0.17)
      mm.allStop()
    if(data == "a"):
      print('left')
      mm.spinLeft(0.7) # orig 0.4
      time.sleep(0.09) #0.11
      mm.allStop()
    if(data == "d"):
      print('right')
      mm.spinRight(0.7) # orig 0.4
      time.sleep(0.09) #0.11
      mm.allStop()
    if(data == "q"):
      print('quit')
      mm.allStop()

    senddata = "OK" + ' ' + data
    if debug == 1:
      print("Sending", senddata)
    conn.send(senddata.encode('utf-8'))

sock.close()
