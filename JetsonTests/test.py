#!/usr/bin/python
# -*- coding:UTF-8 -*-

import RPi.GPIO as GPIO
import time

RelayA = [21, 20, 26]
RelayB = [16, 19, 13]

Pin21 = 21 

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

GPIO.setup(RelayA, GPIO.OUT, initial=GPIO.LOW)
time.sleep(2)
try:
	while True:
		print("turn off")
		GPIO.output(Pin21, GPIO.LOW)
		time.sleep(2)
		print("turn on")
		GPIO.output(Pin21, GPIO.HIGH)		
		time.sleep(2)

except:
    GPIO.cleanup()


# urn on
