# Robotic Model Self Driving Bus
2022 Science Fair Project - Comparing Modular and Integrated Autonomous Vehicle Systems: Inspired by the Human Brain

My school was delayed by 2 weeks this year because few drivers were willing to take the risk of contracting COVID from school kids. 
Outside of school, minors can’t call ride-sharing services because of age restrictions placed after kids were kidnapped by drivers. 
Additionally, many kids are abducted while waiting for transportation, even at school bus stops.  
I wanted to engineer an autonomous vehicle capable of reducing this need for human drivers, and providing minors with a new safe mode of transportation.

My robot consists of two Raspberry Pi, one which uses two cameras for driving, detecting people and stop signs.
The second Raspberry Pi receives instructions from the drive Pi, and uses a gyroscope sensor to command a motor driver to drive or turn the robot.

code
- drive programs
  - drive.py (drives the robot, staying between lines, runs people and stop sign neural networks)
  - motors.py (sets up motors to drive stright, turn left, turn right, drive backwards)
  - pid.py (sets of PID controller which allows for accurate turns)
  - pi_communications.py (sets up wifi server and client to send data between the 2 rasberry Pis)

- neural networks (Colab Notebooks)
  - SignDetectingNetwork.ipynb
  - PeopleDetectingNetwork.ipynb
  - PeopleandSignDetectingNetwork.ipynb


video results
  - videos of robot
  - worksheet with data from robot runs


SelfDrivingBus:VehiclePoster.pdf (poster from PRSEF competition explaining project)
SelfDrivingBus:VehicleSlides.pdf (slides from PJAS competition explaining project)
