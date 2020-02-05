import time
import serial
import argparse

from numpy import array, asarray
import matplotlib.pyplot as plt

def parse_args():
    # Parse command line arguments
    ap = argparse.ArgumentParser(description="Glove HighTemp data receiver")
    ap.add_argument("-i", "--input", required=True,
                    help="input channel")
    ap.add_argument("-s", "--speed", required=True,
                    help="speed rate")
    ap.add_argument("-c", "--count", required=True,
                    help="sample number")
    ap.add_argument("-n", "--name", required=True,
                    help="sensor name")
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

SENSOR_TYPE='Grove'
def main(args):
    ser = serial.Serial(
       port=args["input"],
       baudrate=int(args["speed"]),
    )

    if ser.is_open == True:
       ser.close()

    ser.open()
    i = 0
    diff = 0
    prev_temp = 0
    xs_scaled = []
    xs = []
    while True:
       while ser.inWaiting() == 0:
           pass
       response = ser.readline()
       response = str(response, encoding="utf-8")
       print("read data: " + response)
       tempTuple = (0, 0)

       sensor_name = args["name"],
       if response.startswith(sensor_name):
           tempTuple = getTemp(response)

       xs_scaled.append(tempTuple[0])
       diff = tempTuple[0] - prev_temp
       xs.append(diff)
       prev_temp = tempTuple[0]
       if i < int(args["count"]):
           i = i + 1
       else:
           break
       time.sleep(0.0001)

    xs_scaled = asarray(xs_scaled)
    xs = asarray(xs)
    plt.subplot(211)
    plt.plot(xs_scaled[:], label='measurements')
    plt.plot(xs[:], label='difference')
    plt.legend(loc=4)
    plt.show()


if __name__ == '__main__':
    args = parse_args()
    main(args)
