import glob
import time
from time import sleep
import RPi.GPIO as GPIO

try:
    # interval between measurements
    sleeptime = 1

    # The One-Wire input pin is declared and the integrated pull-up resistor is enabled.
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(4, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    base_dir = '/sys/bus/w1/devices/'

    start_time = time.monotonic()
    duration = 10  # seconds
except OSError:
    print("Failed to initialize device.")

while (time.monotonic() - start_time) < duration:
    try:
        device_folder = glob.glob(base_dir + '28*')[0]
        break
    except IndexError:
        sleep(0.5)
        continue
device_file = device_folder + '/w1_slave'


# read the current measurement blindly
def blind_read_temperature():
    f = open(device_file, 'r')
    lines = f.readlines()
    f.close()
    return lines


# In the Raspberry Pi, detected One-Wire slaves are assigned to their own subfolder in the directory /sys/bus/w1/devices/. In this folder, there is a file called w1_slave where the data transmitted over the One-Wire bus is stored. In this function, this data is analyzed, the temperature is read out, and output.
def read_temperature():
    lines = blind_read_temperature()
    while lines[0].strip()[-3:] != 'YES':
        time.sleep(0.2)
        lines = blind_read_temperature()
    equals_pos = lines[1].find('t=')
    if equals_pos != -1:
        temp_string = lines[1][equals_pos + 2:]
        temp_c = float(temp_string) / 1000.0
        return temp_c


if __name__ == "__main__":
    try:
        while (time.monotonic() - start_time) < duration:
            print("---------------------------------------")
            print("Temperature:", blind_read_temperature(), "°C")
            time.sleep(sleeptime)

    except KeyboardInterrupt:
        GPIO.cleanup()
