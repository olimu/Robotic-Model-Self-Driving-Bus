# autonomousvehicle2022
2022 Science Fair Project - Comparing Modular and Integrated Autonomous Vehicle Systems: Inspired by the Human Brain

My school was delayed by 2 weeks this year because few drivers were willing to take the risk of contracting COVID from school kids. 
Outside of school, minors can’t call ride-sharing services because of age restrictions placed after kids were kidnapped by drivers. 
Additionally, many kids are abducted while waiting for transportation, even at school bus stops.  
I wanted to engineer an autonomous vehicle capable of reducing this need for human drivers, and providing minors with a new safe mode of transportation.

My robot consists of two Raspberry Pi, one which uses two cameras for driving, detecting people and stop signs.
The second Raspberry Pi receives instructions from the drive Pi, and uses a gyroscope sensor to command a motor driver to drive or turn the robot.

Python Scripts for Perception Pi
- drive_dual_final.py

Python Scripts and modules for Drive Pi
- server_tcp.py
- pidClass.py
- motorsmod.py

Colab notebooks for training Neural Networks
- StopMobilenetFullTune.ipynb
- PeopleMobilenetFullTune.ipynb
- PeopleandStopMobilenetFullTune.ipynb

Results
- raw data google sheet
