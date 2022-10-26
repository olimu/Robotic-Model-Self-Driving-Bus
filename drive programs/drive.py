import time
import sys
import imutils
from imutils.video import VideoStream
import numpy as np
import cv2
import socket
import matplotlib.pyplot as plt
live = True
if live:
  from tflite_runtime.interpreter import Interpreter

def logprint(text, fh):
  '''Prints text to stdout and log it to fh.'''
  print(text)
  print(text, file = fh)

def set_input_tensor(interpreter, image):
  '''Sets the input tensor.'''
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image

def get_output_tensor(interpreter, index):
  '''Returns the output tensor at the given index.'''
  output_details = interpreter.get_output_details()[index]
  tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
  return tensor

def canny_and_hough(img, vertices, mask, threshold):
  '''Uses typical cv2 Canny, mask and Hough routines to determine line segments in camera field of view.'''
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  blur_gray = cv2.GaussianBlur(gray, (3, 3), 0)
  edges = cv2.Canny(blur_gray, 60, 120) # parameters with LED light in bedroom
  if edges is None:
    print('[ERROR]: canny failed')
    quit()
  mask.fill(0)
  cv2.fillPoly(mask, vertices, 255)
  canny_with_mask = cv2.addWeighted(edges, 0.5, mask, 0.5, 0.0)
  canny_with_mask_color = cv2.cvtColor(canny_with_mask, cv2.COLOR_GRAY2BGR)
  edges = cv2.bitwise_and(edges, edges, mask=mask)
  line_segments = cv2.HoughLinesP(edges, rho=1.0, theta=np.pi/180, threshold=threshold, minLineLength=10, maxLineGap=20)
  return line_segments, canny_with_mask_color

def analyze_lane(img, vertices, mask, basic_line_img, all_line_img, XSIZE, YSIZE):
  '''Analyzes front cam image for lane markers (using Hough lines) and selects lines to guide robot to stay in lane.'''
  basic_line_img.fill(0)
  all_line_img.fill(0)
  line_segments, canny_with_mask_color = canny_and_hough(img, vertices, mask, 75) # 75 was a good threshold for staying in the driving lane
  if line_segments is None:
    return basic_line_img, all_line_img, canny_with_mask_color, 0, 0, 0, 0, False
  boundary = 1/3
  left_area, right_area = XSIZE * boundary, XSIZE * 2*boundary
  left_segments = []
  right_segments = []
  horiz_segments = []
  angle_left, angle_right, x1_left, x2_right = 0, 0, 0, 0
  for line in line_segments:
    for x1, y1, x2, y2 in line:
      if ((x1 == x2) or (y1 == y2)): # vertical line OR horizontal line
        continue
      fit = np.polyfit((x1, x2), (y1, y2), 1) # fit to a line
      cv2.line(all_line_img, (x1, y1), (x2, y2), (255,0,0), thickness=5)
      cv2.circle(all_line_img, (x1, y1), 10, [255,0,0], -1)
      if x1 < left_area and fit[0] < 0: # x1 on left and negative slope
        left_segments.append((x1, y1, x2, y2))
      if x1 > right_area and y2 > YSIZE * 0.6 and fit[0] >= 0: # x1 on right, y2 on bot half, and positive slope
        right_segments.append((x1, y1, x2, y2))
      if (fit[0] > -0.1 and fit[0] < 0.1 and x1 > right_area): # small slope and x1 on right
        horiz_segments.append((x1, y1, x2, y2))
  if len(left_segments) > 0:
    left_segments = sorted(left_segments, key=lambda left_segment: left_segment[1]) # sort left segments by y1
    x1, y1, x2, y2 = left_segments[len(left_segments) - 1] # max y1 is left line
    angle_left = round(np.arctan2(y1-y2, x2-x1)*180/np.pi, 2) # CHANGED: arctan(y, x)
    x1_left = x1
    cv2.line(basic_line_img, (x1, y1), (x2, y2), (0,0,255), thickness=5)
  if len(right_segments) > 0:
    right_segments = sorted(right_segments, key=lambda right_segment: right_segment[3]) # sort right segments by y2
    x1, y1, x2, y2 = right_segments[len(right_segments) - 1] # max y2 is right line
    angle_right = round(np.arctan2(y2-y1, x2-x1)*180/np.pi, 2) # CHANGED: arctan(y, x)
    x2_right = x2
    cv2.line(basic_line_img, (x1, y1), (x2, y2), (0,255,0), thickness=5)
  horizontal_detected = False
  if len(horiz_segments) > 0:
    x1, y1, x2, y2 = horiz_segments[len(horiz_segments) - 1]
    cv2.line(all_line_img, (x1, y1), (x2, y2), (255,255,255), thickness=5)
    horizontal_detected = True
  return basic_line_img, all_line_img, canny_with_mask_color, angle_left, angle_right, x1_left, x2_right, horizontal_detected

def analyze_intersection(img, vertices, mask):
  '''Analyzes side image to detect intersection.'''
  side_line_segments, side_canny_with_mask_color = canny_and_hough(img, vertices, mask, 25) # 25 was a good threshold for side image
  side_overlay = cv2.addWeighted(img, 0.5, side_canny_with_mask_color, 0.5, 0.0)
  if side_line_segments is None:
    intersection_detected = True
  else:
    intersection_detected = False
    for line in side_line_segments:
      for x1, y1, x2, y2 in line:
        cv2.line(side_overlay, (x1, y1), (x2, y2), (0,0,255), thickness=5)
  return intersection_detected, side_overlay

def analyze_neural_networks(img, config_indivNN, stop_interpreter, people_interpreter, stopandpeople_interpreter):
  '''Uses the individual or joint neural networks to detect stop signs and people.'''
  if config_indivNN:
    set_input_tensor(stop_interpreter, img)
    stop_interpreter.invoke()
    boxes = get_output_tensor(stop_interpreter, 0)
    stop_detected_percentage = boxes[1]
    set_input_tensor(people_interpreter, img)
    people_interpreter.invoke()
    boxes = get_output_tensor(people_interpreter, 0)
    people_detected_percentage = boxes[1]
  else: # code below for config_indivNN False hasn't been tested
    set_input_tensor(stopandpeople_interpreter, img)
    stopandpeople_interpreter.invoke()
    boxes = get_output_tensor(stopandpeople_interpreter, 0)
    people_detected_percentage = boxes[1]
    stop_detected_percentage = boxes[2]
  if stop_detected_percentage > 0.5:
    stop_detected = True
  else:
    stop_detected = False
  if people_detected_percentage > 0.1:
    people_detected = True
  else:
    people_detected = False
  return people_detected, people_detected_percentage, stop_detected, stop_detected_percentage
  
def init_send_to_robot():
  '''Initializes socket comms to Rpi3 to control robot.'''
#  HOST_IP = '192.168.1.13' # client on Rpi 4
#  HOST_IP = '10.0.0.82' # client on Rpi 4
  HOST_IP = '172.26.161.199' # client on Rpi 4
  HOST_PORT = 4005
#  server = ('192.168.1.14', 4000) # server on Rpi 3
#  server = ('10.0.0.1', 4000) # server on Rpi 3
  server = ('172.26.161.196', 4000) # server on Rpi 3
  sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  sock.bind((HOST_IP, HOST_PORT))
  sock.connect(server)
  return sock
  
def send_to_robot(msg, sock, debug):
  '''Sends message to Rpi3 to control robot.'''
  sock.send(msg.encode('utf-8'))
  data = sock.recv(1024)
  data = data.decode('utf-8')
  if debug == 10:
    print("[DEBUG]: Sent and Received", msg, data)

def drive_robot(people_detected, people_detected_percentage, intersection_detected, config_interactive_turn, num_intersections, config_sidecam, angle_left, angle_right, x1_left, x2_right):
  '''Uses people, intersection and lane data to drive robot. '''
  if people_detected:
    textdrive = 'people detected ' + str(people_detected_percentage)
    textdrive_demo = 'People detected'
    msg = 'p'
    return msg, textdrive, textdrive_demo
  if intersection_detected:
    if config_interactive_turn:
      textdrive = 'interactive turn '
      textdrive_demo = '' # figure out
      val = input('use gyro turn or go straight [t/s] ?')
    else:
      # turn only on 2nd potential intersection detected
      if num_intersections == 2: 
        textdrive = 'programmed turn num_intersections = ' + str(num_intersections)
        textdrive_demo = '' # figure out
        val = 't'
      else:
        textdrive = 'programmed straight num_intersections = ' + str(num_intersections)
        textdrive_demo = '' # figure out
        val = 's'
    if val == 't':
      textdrive = textdrive + ' turn selected' + ' gyro? ' + str(config_sidecam)
      textdrive_demo = '' # figure out 
      if config_sidecam:
        msg = 'g' # use gyro if config_sidecam is True
      else:
        msg = 't' # use regular 90 deg turn (ie do not use gyro) since config_sidcam is False
    else:
      textdrive = textdrive + ' straight selected'
      textdrive_demo = '' # figure out
      msg = 'w'
    return msg, textdrive, textdrive_demo
  if ((angle_left > 10 and angle_left < 80) or (angle_right > 10 and angle_right < 80)):
    # left and right edges are not horizontal far edges
    if x1_left > 63: # too far left or cannot see right
      textdrive = 'turning right due to x1_left: ' + str(x1_left)
      textdrive_demo = 'turning right'
      msg = 'd'
    elif x2_right > 0 and x2_right < 595: #537
      textdrive = 'turning left due to x2_right: ' + str(x2_right)
      textdrive_demo = 'turning left'
      msg = 'a'
    elif angle_left > 0 and angle_left < 25: 
      textdrive = 'turning right due to angle_left: ' + str(angle_left)
      textdrive_demo = 'turning right'
      msg = 'd'
    elif angle_right > 0 and angle_right < 65: 
      textdrive = 'turning left due to angle_right: ' + str(angle_right)
      textdrive_demo = 'turning left'
      msg = 'a'
    else:
      textdrive = 'straight'
      textdrive_demo = textdrive
      msg = 'w'
  else:
    # left and right edges are probably horizontal far edge
    textdrive = 'straight w/ large angles or small angle_left'
    textdrive_demo = 'straight'
    msg = 'w'
  return msg, textdrive, textdrive_demo

def get_image(cam):
  '''get an image from the camera'''
  img = cam.read()
  if img is None:
    print('[INFO]: End of captured video file')
    quit()
    if np.all((img == 0)):
      print('[INFO]: img is black... quitting')
      quit()
  return img


#### main code
if __name__ == '__main__':
  # configuration variables
  config = 'ihis'
  if ((config == 'mhms') or (config == 'mhis')):
    config_sidecam = True # True is modular hw (also turns on gyro)
  else:
    config_sidecam = False # False is integrated hw (also turns off gyro)
  if ((config == 'mhms') or (config == 'ihms')):
    config_indivNN = True # modular sw
  else:
    config_indivNN = False # integrated sw
  debug = 0 # debug options are
            # 0 no debug
            # 6: show images each loop for front image analysis
            # 7: only works with config_sidecam == True for side image analysis
            # 10: communication debug only
  config_interactive_turn = False # do not make to false as it detects turn in first image
  if live:
    config_drive_robot = True # or can be True, set manually
  else:
    config_drive_robot = False # has to be False when not live

  # time variables
  now = time.time()
  start = now
    
  # initialize image size
  XSIZE = 640
  YSIZE = 480

  # initialize various mask vertices to define region of interest
  bot = 0.2
  s_bot = 0.3
  #vertices = np.array([[(0, 0), (XSIZE, 0), (XSIZE, YSIZE), (0, YSIZE)]], np.int32)
  f_vertices = np.array([[(0, YSIZE*bot), (XSIZE, YSIZE*bot), (XSIZE, YSIZE), (0, YSIZE)]], np.int32)
  s_vertices = np.array([[(0, YSIZE*s_bot), (XSIZE, YSIZE*s_bot), (XSIZE, YSIZE), (0, YSIZE)]], np.int32)
  nn_vertices = np.array([[(0, YSIZE*bot), (XSIZE, YSIZE*bot), (XSIZE, YSIZE), (0, YSIZE)]], np.int32)

  # initialize log file
  timestr = time.strftime("%m%d-%H%M%S")
#  if live:
  filename = 'log_drive_dual2' + timestr + '.txt'
#  else:
#    filename = '/dev/null'
  logfile = open(filename, 'w+', encoding='utf-8')

  # initialize socket communications with Rpi3 to drive robot
  if config_drive_robot:
    # initialize socket communications with Rpi 3
    logprint('[INFO]: Initializing Socket Comm Link', logfile)
    sock = init_send_to_robot()
  else:
    logprint('[INFO]: config_drive_robot is False -- will not send Comms to Rpi 3 server_tcp', logfile)

  if live:
    # initialize the interpreter
    logprint('[INFO]: Initializing Interpreter(s)', logfile)
    if config_indivNN:
      stop_interpreter = Interpreter('stop_mobilenet4_full.tflite') # try 4 80 epochs
      stop_interpreter.allocate_tensors()
      _, nn_input_height, nn_input_width, _ = stop_interpreter.get_input_details()[0]['shape']
      people_interpreter = Interpreter('people_mobilenet4_full.tflite') # try 4 80 epochs
      people_interpreter.allocate_tensors()
      _, nn_input_height, nn_input_width, _ = people_interpreter.get_input_details()[0]['shape']
      stopandpeople_interpreter = 'undefined'
    else:
      stopandpeople_interpreter = Interpreter('peoplestop_dec_mobilenet4.tflite') # try 4 80 epochs
      stopandpeople_interpreter.allocate_tensors()
      _, nn_input_height, nn_input_width, _ = stopandpeople_interpreter.get_input_details()[0]['shape']
      stop_interpreter = people_interpreter = 'undefined'
  
    # initialize the cameras
    logprint('[INFO]: Initializing Cameras', logfile)
    fcam = imutils.video.VideoStream(usePiCamera=True).start() # front camera
    scam = imutils.video.VideoStream(src=0).start() # side camera
    # allow the cameras to warmup
    time.sleep(2.0)
  else: # not live
    # nothing to do do initialize the interpreter
    # initialize the readfile
    # Google Drive
    # prefix = '/content/drive/My Drive/ScienceFair20-21/'
    # fcam = cv2.VideoCapture(prefix + 'rawtest.mp4')
    # scam = cv2.VideoCapture(prefix + 'sidetest.mp4')
    # On Pi
    prefix = '/home/pi/runrobot/videos/demo_video-21-Dec-19-evening/'
    fcam = imutils.video.FileVideoStream(prefix + 'raw.mp4').start()
    scam = imutils.video.FileVideoStream(prefix + 'side.mp4').start()
  
  logprint('[INFO]: Initializing VideoWriters', logfile)
  overlay_vid = cv2.VideoWriter('overlay.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 15, (XSIZE, YSIZE))
  canny_vid = cv2.VideoWriter('canny.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 15, (XSIZE, YSIZE))
  raw_vid = cv2.VideoWriter('raw.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 15, (XSIZE, YSIZE))
  merge_vid = cv2.VideoWriter('merge.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 15, (XSIZE, YSIZE))
  side_vid = cv2.VideoWriter('side.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 15, (XSIZE, YSIZE))

  ### Miscelleneous initializations

  # i and current_iter are loop iterations
  i = 0
  current_iter = 0

  # counters to aid in number of iterations beyond stop detection
  # prevents early stop
  stop_iter = 0
  num_times_stop_detected = 0
  people_detected = False
  num_intersections = 0
  prev_intersection_detected = False
  num_no_right_line = 0
  last_leg_counter = 0

  # allocate memory
  basic_line_img = np.zeros((YSIZE, XSIZE, 3), dtype=np.uint8)
  all_line_img = np.zeros((YSIZE, XSIZE, 3), dtype=np.uint8)
  mask = np.zeros((YSIZE, XSIZE), dtype=np.uint8)

  fimg = get_image(fcam)
#  if config_sidecam:
#    simg = get_image(scam)
#    plt.imshow(simg)
#    plt.show()
#    logprint('[INFO]: check side camera image', logfile)
  if debug > 0:
    ax1 = plt.subplot(2,2,1)
    ax2 = plt.subplot(2,2,2)
    ax3 = plt.subplot(2,2,3)
    ax4 = plt.subplot(2,2,4)
    im1 = ax1.imshow(imutils.opencv2matplotlib(fimg))
    im2 = ax2.imshow(imutils.opencv2matplotlib(all_line_img))
    im3 = ax3.imshow(imutils.opencv2matplotlib(basic_line_img))
    im4 = ax4.imshow(imutils.opencv2matplotlib(mask))
    plt.ion()

  logprint('[INFO]: Entering Main Loop', logfile)
  while True:
    # to aid in capturing training images
    #plt.imshow(fimg)
    #plt.show()

    # process front image to keep robot in lane
    fimg = imutils.resize(fimg, width = XSIZE, height = YSIZE)
    raw_vid.write(fimg)
    basic_line_img, all_line_img, canny_with_mask_color, angle_left, angle_right, x1_left, x2_right, horizontal_detected = analyze_lane(fimg, f_vertices, mask, basic_line_img, all_line_img, XSIZE, YSIZE)
    if angle_left < 0 or angle_left > 90 or angle_right < 0 or angle_right > 90 or x1_left < 0 or x1_left > XSIZE or x2_right < 0 or x2_right > XSIZE:
      print('[ERROR]: ', angle_left, angle_right, x1_left, x2_right)
      quit()

    if debug == 6:
      im1.set_data(imutils.opencv2matplotlib(fimg))
      im2.set_data(imutils.opencv2matplotlib(all_line_img))
      im3.set_data(imutils.opencv2matplotlib(basic_line_img))
      im4.set_data(imutils.opencv2matplotlib(canny_with_mask_color))
      plt.pause(0.2)

    # process side image to detect intersections
    simg = get_image(scam)
    side_vid.write(simg)
    simg = imutils.resize(simg, width = XSIZE, height = YSIZE)
    # rotate simg (side image) by 180
    simg = cv2.flip(simg, 0)
    simg = cv2.flip(simg, 1)
    #plt.imshow(cv2.hconcat([fimg, simg]))
    #plt.show()

    if config_sidecam:
      intersection_detected, side_overlay = analyze_intersection(simg, s_vertices, mask)
      if debug == 7:
        im1.set_data(imutils.opencv2matplotlib(fimg))
        im2.set_data(imutils.opencv2matplotlib(all_line_img))
        im3.set_data(imutils.opencv2matplotlib(simg))
        im4.set_data(imutils.opencv2matplotlib(side_overlay))
        plt.pause(0.2)

      if intersection_detected:
        textintersection_demo = 'Intersection'
        if not prev_intersection_detected:
          num_intersections = num_intersections + 1
        prev_intersection_detected = True
      else:
        textintersection_demo = ''
        prev_intersection_detected = False
    else:
      #### print(num_no_right_line, horizontal_detected)
      intersection_detected = False
      textintersection_demo = ''
      if angle_right == 0 and x2_right == 0:
        num_no_right_line = num_no_right_line + 1
      else:
        num_no_right_line = 0
      if horizontal_detected and num_no_right_line > 10:
        #### print('detected horizontal')
        intersection_detected = True
        num_intersections = num_intersections + 1
        textintersection_demo = 'Intersection'
        num_no_right_line = 0

    if live:
      nnimg = cv2.resize(fimg, (nn_input_width, nn_input_height))
      people_detected, people_detected_percentage, stop_detected, stop_detected_percentage = analyze_neural_networks(nnimg, config_indivNN, stop_interpreter, people_interpreter, stopandpeople_interpreter)
      people_detected_percentage = round(people_detected_percentage, 4)
      stop_detected_percentage = round(stop_detected_percentage, 4)
      textstop = str(current_iter) + ' st ' + str(stop_detected_percentage)
      textpeople = str(current_iter) + ' pe ' + str(people_detected_percentage)
      if stop_detected:
        textstop = textstop + ' stop sign detected '
        textstop_demo = 'Stop Sign'
        stop_iter = current_iter
        num_times_stop_detected = num_times_stop_detected + 1
      else:
        textstop_demo = ''
        if num_times_stop_detected > 2 and current_iter - stop_iter == 10:
          logprint('stopping for stop sign ' + str(current_iter), logfile)
          time.sleep(10)
          stop_iter = 0
          num_times_stop_detected = 0
      if people_detected:
        textpeople = textpeople + ' people detected '
        textpeople_demo = 'People'
      else:
        textpeople_demo = ''
    else:
      textstop = textpeople = textstop_demo = textpeople_demo = 'not live'
      stop_detected_percentage = people_detected_percentage = 0

    # capture front frames from the cameras after NN for less blur
    next_fimg = get_image(fcam)

    # loop controls
    if i == 1500:
      # stop the robot
      msg = 'p'
      if config_drive_robot:
        send_to_robot(msg, sock, debug)
        val = input('continue [y/n] ?')
        if val == 'y':
          i = 0
          continue
        else:
          break

    msg, textdrive, textdrive_demo = drive_robot(people_detected, people_detected_percentage, intersection_detected, config_interactive_turn, num_intersections, config_sidecam, angle_left, angle_right, x1_left, x2_right)
    if num_intersections > 1:
      last_leg_counter = last_leg_counter + 1
    if last_leg_counter == 16:
      logprint("[INFO]: Arrived at Destination!!!", logfile)
      quit()

    text = str(current_iter) + ' ' + str(angle_left) + ' ' + str(angle_right) + ' ' + str(x1_left) + ' ' + str(x2_right) + ' ' + str(num_intersections) + ' ' + str(people_detected_percentage) + ' ' + str(stop_detected_percentage) + ' ' + msg
    logprint(text + '  ' + textdrive, logfile)

    cv2.putText(all_line_img, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText(all_line_img, str(current_iter) + ' ' + textdrive, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText(all_line_img, textstop, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText(all_line_img, textpeople, (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    overlay_with_canny = cv2.addWeighted(fimg, 1.0, canny_with_mask_color, 0.2, 0.0)
    overlay_with_canny = cv2.addWeighted(overlay_with_canny, 0.5, all_line_img, 0.5, 0.0)
    overlay_with_canny = cv2.addWeighted(overlay_with_canny, 0.5, basic_line_img, 0.5, 0.0)
    #plt.imshow(cv2.hconcat([fimg, canny_with_mask_color, all_line_img, basic_line_img, overlay_with_canny, simg]))
    #plt.imshow(cv2.hconcat([fimg, all_line_img, simg]))
    #plt.show()
    now = time.time()
  
    cv2.putText(basic_line_img, str(round(time.time()-start, 1)) + 's', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText(basic_line_img, textdrive_demo, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText(basic_line_img, textintersection_demo, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText(basic_line_img, textstop_demo, (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText(basic_line_img, textpeople_demo, (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    merge_img = cv2.addWeighted(fimg, 0.5, basic_line_img, 0.5, 0.0)
    overlay_vid.write(overlay_with_canny)
    canny_vid.write(canny_with_mask_color)
    merge_vid.write(merge_img)  
  
    if config_drive_robot:
      send_to_robot(msg, sock, debug)
#  if live:
#    time.sleep(0.5)
    i = i + 1
    current_iter = current_iter + 1
    fimg = next_fimg
  # end of while True loop
  plt.show()
  raw_vid.release()
  canny_vid.release()
  overlay_vid.release()
  merge_vid.release()
  side_vid.release()
  if config_drive_robot:
    sock.close()
  quit()




