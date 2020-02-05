import time
import serial
import argparse

import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from numpy import array, asarray

import matplotlib.pyplot as plt

def parse_args():
    # Parse command line arguments
    ap = argparse.ArgumentParser(description="1-wire data receiver")
    ap.add_argument("-i", "--input", required=True,
                    help="input channel")
    ap.add_argument("-s", "--speed", required=True,
                    help="speed rate")
    ap.add_argument("-c", "--count", required=True,
                    help="speed rate")

    return vars(ap.parse_args())

def maketuple(variables, names):
  return tuple(variables[n] for n in names)

temp = 0
y = 0
def getTemp(response):
    tempList = response.split('\t')
    temp = float(tempList[1])
    y = float(tempList[2])

    if (len(tempList) == 3):
       if type(float(tempList[1])) is float:
          temp = float(tempList[1])
       else:
          print('Data is not in float and hence marked to zero')
          tempList[1] = 0.0
          temp = float(tempList[1])

       if type(float(tempList[2])) is float:
          y = float(tempList[2])
       else:
           print('Data is not in float and hence marked to zero')
           tempList[2] = 0.0
           y = float(tempList[2])

    yprTuple = maketuple(vars(), 'temp y'.split())
    outputYpr = "{} - {}".format(temp, y)
    print(outputYpr)
    return yprTuple

my_filter = KalmanFilter(dim_x=2, dim_z=1)
my_filter.x = np.array([[2.],
                [0.]])       # initial state (location and velocity)
my_filter.F = np.array([[1.,1.],
                [0.,1.]])    # state transition matrix

my_filter.H = np.array([[1.,0.]])    # Measurement function
my_filter.P *= 1000.                 # covariance matrix
my_filter.R = 5                      # state uncertainty
dt = 0.1
my_filter.Q = Q_discrete_white_noise(2, dt, .1)  # process uncertainty



def main(args):
    ser = serial.Serial(
    #   port='COM16',
       port=args["input"],
       baudrate=int(args["speed"]),
    )

    if ser.is_open == True:
       ser.close()

    ser.open()
    i = 0
    xs_scaled = []
    xs = []
    ct = 20
    while True:
       while ser.inWaiting() == 0:
           pass
       response = ser.readline()
       response = str(response, encoding="utf-8")
       print("read data: " + response)
       #if response.startswith('Send'):
       #   ser.write(str.encode('r'))
       gyroTuple =(0,0)
       if response.startswith('Grove'):
          gyroTuple = getTemp(response)



       my_filter.predict()
       my_filter.update(gyroTuple[0])
       xs_scaled.append(gyroTuple[0])
       # do something with the output
       x = my_filter.x
       print(x)
       xs.append(x)
       if i < int(args["count"]):
          i = i + 1
       else:
          break
       time.sleep(0.0001)

    xs_scaled = asarray(xs_scaled)
    xs = asarray(xs)
    plt.subplot(211)
    plt.plot(xs_scaled[:], label='measurements')
    plt.plot(xs[:, 0], label='estimated by KF')
    plt.legend(loc=4)

    plt.subplot(212)
    plt.plot(xs_scaled[:], label='measurements')
    plt.plot(xs[:, 1], label='estimation error')
    plt.legend(loc=4)
    plt.show()


if __name__ == '__main__':
    args = parse_args()
    main(args)