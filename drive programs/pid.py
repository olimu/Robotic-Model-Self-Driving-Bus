
class PID:
    '''
    Class for managing PID
    '''

    def __init__(self, Target, Kp, Ki, Kd, numint) -> None:
        '''
        Initialize class

        param Target: PID target value
        param Kp: PID proportional constant
        param Ki: PID integration constant 
        param Kd: PIT derivative constant
        param numint: integration order (must be 1 or 2)
        '''

        self.sensor = 0
        self.Target = Target
        self.Error = 0
        self.ErrorArea = 0
        self.ErrorSlope = 0
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.ControlVal = 0
        try:
            if not ((numint == 1) or (numint == 2)):
                raise ValueError
        except ValueError:
            print('numint must be either 1 or 2')
            
        self.numint = numint
        self.int1 = 0 # used for 2nd order integrations

    def update(self, sensorData, dt, reset):
        '''
        Perform PID update

        param sensorData: accel or gyro data
        param dt: integration delta time between this reading and previous reading
        '''

        if self.numint == 1:
            self.sensor = self.sensor + sensorData*dt
        if self.numint == 2:
            if reset == 1:
                self.int1 = 0
            else:
                self.int1 = self.int1 + sensorData*dt
            self.sensor = self.sensor + self.int1*dt + 0.5*sensorData*dt*dt
        self.ErrorOld = self.Error
        self.Error = self.Target - self.sensor
        self.ErrorArea = self.ErrorArea + self.Error * dt
        self.ErrorSlope = (self.Error - self.ErrorOld)/dt

        self.ControlVal = self.Kp * self.Error + self.Ki * self.ErrorArea + self.Kd * self.ErrorSlope
        return self.int1, self.sensor, self.ControlVal

    def limit(self, val):
        '''
        Limit PID Control

        param val: maximum value (must be 1), expected to always be positive
        '''
        try: 
            if val > 1:
                raise ValueError
            if val < 0:
                raise ValueError
        except ValueError:
            print('PWM values range between 0 and 1')
            
        if self.ControlVal > val:
            self.ControlVal = val
        if self.ControlVal < -val:
            self.ControlVal = -val

        return self.ControlVal
    
