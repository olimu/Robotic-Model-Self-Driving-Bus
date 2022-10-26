"""
This code will control Raspberry Pi GPIO PWM on four GPIO
pins. The code test ran with one TB6612N H-Bridge driver module connected.

Inspired by Website:www.bluetin.io
Date: Dec 14, 2021
"""

from gpiozero import PWMOutputDevice
from gpiozero import DigitalOutputDevice
import time

#///////////////// Define Motor Driver GPIO Pins /////////////////

# switch is front
# battery is back
# TB6612 GPIO pins
# Motor A, Right Side GPIO CONSTANTS
PWM_DRIVE_RIGHT = 21 #40   # PWMA - H-Bridge enable pin
FORWARD_RIGHT_PIN = 26 #37 # IN1 - Forward Drive
REVERSE_RIGHT_PIN = 19 #35 # IN2 - Reverse Drive
# Motor B, Left Side GPIO CONSTANTS
FORWARD_LEFT_PIN = 24 # 18 #13 #33 # IN1 - Forward Drive
REVERSE_LEFT_PIN = 6 #31 # IN2 - Reverse Drive
PWM_DRIVE_LEFT = 5 #29   # PWMB - H-Bridge enable pin

# Initialise objects for H-Bridge GPIO PWM pins
# Set initial duty cycle to 0 and frequency to 1000
driveLeft = PWMOutputDevice(PWM_DRIVE_LEFT, True, 0, 1000)
driveRight = PWMOutputDevice(PWM_DRIVE_RIGHT, True, 0, 1000)

# Initialise objects for H-Bridge digital GPIO pins
forwardLeft = DigitalOutputDevice(FORWARD_LEFT_PIN)
reverseLeft = DigitalOutputDevice(REVERSE_LEFT_PIN)
forwardRight = DigitalOutputDevice(FORWARD_RIGHT_PIN)
reverseRight = DigitalOutputDevice(REVERSE_RIGHT_PIN)

def allStop():
    forwardLeft.value = True
    reverseLeft.value = True
    forwardRight.value = True
    reverseRight.value = True
    driveLeft.value = 0
    driveRight.value = 0

def forwardDrive(speed, silent = 'False'):
    if not silent:
        print('forwardDrive: ', speed)
    forwardLeft.value = False
    reverseLeft.value = True
    forwardRight.value = False
    reverseRight.value = True
    driveLeft.value = speed
    driveRight.value = speed

def reverseDrive(speed, silent = 'False'):
    if not silent:
        print('reverseDrive: ', speed)
    forwardLeft.value = True
    reverseLeft.value = False
    forwardRight.value = True
    reverseRight.value = False
    driveLeft.value = speed
    driveRight.value = speed

# used for pid4.py
def Drive(speed, silent = 'False'):
    if not silent:
        print('Drive: (neg means reverse', speed)
    if speed < 0:
        forwardLeft.value = True
        reverseLeft.value = False
        forwardRight.value = True
        reverseRight.value = False
        speed = -speed
    else:
        forwardLeft.value = False
        reverseLeft.value = True
        forwardRight.value = False
        reverseRight.value = True
    driveLeft.value = speed
    driveRight.value = speed
        
def spinLeft(speed, silent = 'False'):
    if not silent:
        print('spinLeft: ', speed)
    forwardLeft.value = True
    reverseLeft.value = False
    forwardRight.value = False
    reverseRight.value = True
    driveLeft.value = speed
    driveRight.value = speed

def spinRight(speed, silent = 'False'):
    if not silent:
        print('spinRight: ', speed)
    forwardLeft.value = False
    reverseLeft.value = True
    forwardRight.value = True
    reverseRight.value = False
    driveLeft.value = speed
    driveRight.value = speed

# used for pid4.py
def Spin(speed, silent = 'False'):
    if not silent:
        print('Spin: (negative is Left', speed)
    if speed < 0:
        forwardLeft.value = True
        reverseLeft.value = False
        forwardRight.value = False
        reverseRight.value = True
        speed = -speed
    else:
        forwardLeft.value = False
        reverseLeft.value = True
        forwardRight.value = True
        reverseRight.value = False
    driveLeft.value = speed
    driveRight.value = speed
        
def turnOff():
    driveLeft.off()
    driveRight.off()
    forwardLeft.off()
    reverseLeft.off()
    forwardRight.off()
    reverseRight.off()

