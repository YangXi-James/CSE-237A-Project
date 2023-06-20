import time
import board
import busio
import adafruit_sgp30

try:
    i2c = busio.I2C(board.SCL, board.SDA, frequency=100000)
    # Create library object on our I2C port
    sgp30 = adafruit_sgp30.Adafruit_SGP30(i2c)
    print("SGP30 serial #", [hex(i) for i in sgp30.serial])
    sgp30.set_iaq_baseline(0x8973, 0x8AAE)
    sgp30.set_iaq_relative_humidity(celsius=22.1, relative_humidity=44)

    duration = 10  # seconds
except ValueError:
    print("Initializing sensor failed. Will use random values for testing...")


def read_air_quality_local():
    start_time = time.monotonic()
    elapsed_sec = 0

    while (time.monotonic() - start_time) < duration:
        print("eCO2 = %d ppm \t TVOC = %d ppb" % (sgp30.eCO2, sgp30.TVOC))
        time.sleep(1)
        elapsed_sec += 1
        if elapsed_sec > 10:
            elapsed_sec = 0
            print(
                "**** Baseline values: eCO2 = 0x%x, TVOC = 0x%x"
                % (sgp30.baseline_eCO2, sgp30.baseline_TVOC)
            )


def read_air_quality():
    return sgp30.eCO2, sgp30.TVOC


if __name__ == "__main__":
    read_air_quality_local()
