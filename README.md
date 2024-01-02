# Robotic Model Self Driving Bus
**Comparing Modular and Integrated Autonomous Vehicle Systems: Inspired by the Human Brain**

My school was delayed by 2 weeks this year because few drivers were willing to take the risk of contracting COVID from school kids. Outside of school, minors can’t call ride-sharing services because of age restrictions placed after kids were kidnapped by drivers. Additionally, many kids are abducted while waiting for transportation, even at school bus stops. I wanted to engineer an autonomous vehicle capable of reducing this need for human drivers, and providing minors with a new safe mode of transportation.

My robot consists of a Perception Raspberry Pi  which processes camera data to sense lane lines, people, and stop signs. The Drive Pi receives instructions from the Perception Pi, using the motor driver and gyroscope to steer the robot.

**code folder**
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


**SelfDrivingBus:VehiclePoster.pdf:** poster from the Pittsburgh Regional Science and Engineering Fair where I won sponsor awards from FedEx, Institute of Electrical and Electronics Engineers (Honorable Mention), and U.S. Air Force

**SelfDrivingBus:VehicleSlides.pdf:** slides from the Pennsylvania Junior Academy of Science competition where I won 1st place and the Director's Award at the Region 7 competition and 1st place with a perfect score at the State competition in the Computer Science category
