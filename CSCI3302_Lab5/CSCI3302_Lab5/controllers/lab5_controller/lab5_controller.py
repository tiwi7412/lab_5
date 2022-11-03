"""lab5 controller."""
from runpy import _TempModule
from controller import Robot, Motor, Camera, RangeFinder, Lidar, Keyboard
import math
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import convolve2d # Uncomment if you want to use something else for finding the configuration space
import sys

MAX_SPEED = 7.0  # [rad/s]
MAX_SPEED_MS = 0.633 # [m/s]
AXLE_LENGTH = 0.4044 # m
MOTOR_LEFT = 10
MOTOR_RIGHT = 11
N_PARTS = 12
WHEEL_RADIUS = 0.0904
AXLE_RADIUS = AXLE_LENGTH/2

LIDAR_ANGLE_BINS = 667
LIDAR_SENSOR_MAX_RANGE = 2.75 # Meters
LIDAR_ANGLE_RANGE = math.radians(240)


##### vvv [Begin] Do Not Modify vvv #####

# create the Robot instance.
robot = Robot()
# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

# The Tiago robot has multiple motors, each identified by their names below
part_names = ("head_2_joint", "head_1_joint", "torso_lift_joint", "arm_1_joint",
              "arm_2_joint",  "arm_3_joint",  "arm_4_joint",      "arm_5_joint",
              "arm_6_joint",  "arm_7_joint",  "wheel_left_joint", "wheel_right_joint")

# All motors except the wheels are controlled by position control. The wheels
# are controlled by a velocity controller. We therefore set their position to infinite.
target_pos = (0.0, 0.0, 0.09, 0.07, 1.02, -3.16, 1.27, 1.32, 0.0, 1.41, 'inf', 'inf')
robot_parts=[]

for i in range(N_PARTS):
    robot_parts.append(robot.getDevice(part_names[i]))
    robot_parts[i].setPosition(float(target_pos[i]))
    robot_parts[i].setVelocity(robot_parts[i].getMaxVelocity() / 2.0)

# The Tiago robot has a couple more sensors than the e-Puck
# Some of them are mentioned below. We will use its LiDAR for Lab 5

# range = robot.getDevice('range-finder')
# range.enable(timestep)
# camera = robot.getDevice('camera')
# camera.enable(timestep)
# camera.recognitionEnable(timestep)
lidar = robot.getDevice('Hokuyo URG-04LX-UG01')
lidar.enable(timestep)
lidar.enablePointCloud()

# We are using a GPS and compass to disentangle mapping and localization
gps = robot.getDevice("gps")
gps.enable(timestep)
compass = robot.getDevice("compass")
compass.enable(timestep)

# We are using a keyboard to remote control the robot
keyboard = robot.getKeyboard()
keyboard.enable(timestep)

# The display is used to display the map. We are using 360x360 pixels to
# map the 12x12m2 apartment
display = robot.getDevice("display")

# Odometry
pose_x     = 0
pose_y     = 0
pose_theta = 0

vL = 0
vR = 0

lidar_sensor_readings = [] # List to hold sensor readings
lidar_offsets = np.linspace(-LIDAR_ANGLE_RANGE/2., LIDAR_ANGLE_RANGE/2., LIDAR_ANGLE_BINS)
lidar_offsets = lidar_offsets[83:len(lidar_offsets)-83] # Only keep lidar readings not blocked by robot chassis
map = None
##### ^^^ [End] Do Not Modify ^^^ #####

##################### IMPORTANT #####################
# Set the mode here. Please change to 'autonomous' before submission
#mode = 'manual' # Part 1.1: manual mode
#mode = 'planner'
mode = 'autonomous'

def get_neighbors(vertex, map): #can at most send a list of 4 pairs of coordinates
    x = vertex[0]
    y = vertex[1]
    neighbors = []
    if x > 0:
        neighbors.append([x-1,y])
    if x < len(map) - 1: #0 - 359
        neighbors.append([x+1,y])
    if y > 0:
        neighbors.append([x,y-1])
    if y < len(map) - 1:
        neighbors.append([x,y+1])
    return neighbors
    
def get_neighbors_bigger(vertex, map):
    x = vertex[0]
    y = vertex[1]
    neighbors = []
    if x > 0:
        neighbors.append([x-1,y])
        if y > 0:
            neighbors.append([x-1,y-1])
    if x < len(map) - 1: #0 - 359
        neighbors.append([x+1,y])
        if y < len(map) - 1:
            neighbors.append([x+1,y+1])
    if y > 0:
        neighbors.append([x,y-1])
        if x < len(map) - 1:
            neighbors.append([x+1,y-1])
    if y < len(map) - 1:
        neighbors.append([x,y+1])
        if x > 0:
            neighbors.append([x-1,y+1])
    return neighbors

def get_travel_cost(source, dest, map):
    if source == dest:
        return 0
    elif map[dest[0]][dest[1]] > 0.2:
        return 1e5
    return 1
        


###################
#
# Planner
#
###################
if mode == 'planner':
    # Part 2.3: Provide start and end in world coordinate frame and convert it to map's frame
    start_w = (-8.4357, -4.6653) # (Pose_X, Pose_Z) in meters
    end_w = (-7, -10.0) # in meters

    # Convert the start_w and end_w from the webots coordinate frame into the map frame
    start = (round(start_w[0]*-30), round(start_w[1]*-30))# (x, y) in 360x360 map
    end = (round(end_w[0]*-30), round(end_w[1]*-30)) # (x, y) in 360x360 map

    # Part 2.3: Implement A* or Dijkstra's Algorithm to find a path
    def path_planner(map, start, end):
   
        #param map: A 2D numpy array of size 360x360 representing the world's cspace with 0 as free space and 1 as obstacle
        #param start: A tuple of indices representing the start cell in the map
        #param end: A tuple of indices representing the end cell in the map
        #return: A list of tuples as a path from the given start to the given end in the given maze
     
        dist = {}
        prev = {}
        for i in range(len(map)):
            for j in range(len(map[0])):
                dist[(i, j)] = math.inf
                prev[(i, j)] = None

        dist[(start[0], start[1])] = 0

        q_cost = dist.copy()

        while len(q_cost) > 0:
            u = min(q_cost, key=q_cost.get)
            _ = q_cost.pop(u)
            for neighbor in get_neighbors(u, map):
                new_dist = dist[u] + get_travel_cost(u, neighbor, map)
                if new_dist < dist[(neighbor[0], neighbor[1])]:
                    dist[(neighbor[0], neighbor[1])] = new_dist
                    q_cost[(neighbor[0], neighbor[1])] = new_dist
                    prev[(neighbor[0], neighbor[1])] = u

        #find the shortest path from start to end using while loop
        #return value
        path = [end]
        parent_vertex = prev[(end[0], end[1])]

        while parent_vertex is not None:
            path.append(parent_vertex)
            parent_vertex = prev[(parent_vertex[0], parent_vertex[1])]

        path.reverse()
        return path

    # Part 2.1: Load map (map.npy) from disk and visualize it    



    lidar_map = np.load("map.npy")
    print("lidar_map loaded")

    #Play with this number to find something suitable, the number corresponds to the # of pixels you want to cover
    kernel_size = 5
    
    for i in range(len(lidar_map)):
         for j in range(len(lidar_map[i])):
                # filter out small values on the map
                 if lidar_map[i][j] < 1:
                     lidar_map[i][j] = 0

                 #draw squares on each pixel
                 if lidar_map[i][j] != 0:
                     lidar_map[i][j] = 0
                     rectangle = plt.Rectangle((i - 1,j - 1), 8, 8, fc='yellow')           
                     plt.gca().add_patch(rectangle)
                     kernel = np.ones((kernel_size, kernel_size))
                     convolved_map = convolve2d(lidar_map, kernel, mode = 'same')
                     #now threshold this map
                    
    plt.imshow(lidar_map)
    plt.show()
    

    #test
 
    # Part 2.2: Compute an approximation of the “configuration space”

    
    # Part 2.3 continuation: Call path_planner
    lidar_map = np.load("map.npy")
    lidar_map_copy = lidar_map.copy()
    for i in range(len(lidar_map)):
         for j in range(len(lidar_map[i])):
             if lidar_map[i][j] > 0.2:
                 lidar_map_copy[i][j] = 1
                 for neighbor in get_neighbors_bigger((i,j), lidar_map):
                     lidar_map_copy[neighbor[0]][neighbor[1]] = 1 
    path_full = path_planner(lidar_map_copy, start, end) #returns a list of coords #need to change end_w = (10.0, 7.0) # Pose_X, Pose_Z in meters and start to robot start pos
    
    # Part 2.4: Turn paths into waypoints and save on disk as path.npy and visualize it
    waypoints = []
    for i in range(len(path_full)):
        wp = [round(path_full[i][0]/-30, 2), round(path_full[i][1]/-30, 2)] #list index, coord
        if i == 0:
            waypoints.append(wp)
        elif i > 0 and waypoints[-1] != wp:
            waypoints.append(wp)
    np.save('path.npy',waypoints)
    print("waypoints file saved")
    print(waypoints)
        
        #for k in range(len(waypoints)):
        #    waypoint_map = (waypoints[k][0]*(-30),waypoints[k][1]*(-30))
        #    if i == waypoint_map[0] and j == waypoint_map[1]:
        #        rectangle = plt.Rectangle((i - 1,j - 1), 2, 2, fc='green')           
        #        plt.gca().add_patch(rectangle)
        #        kernel = np.ones((kernel_size, kernel_size))
        #        convolved_map = convolve2d(lidar_map, kernel, mode = 'same')
        #plt.imshow(lidar_map)
        #plt.show()

######################
#
# Map Initialization
#
######################

# Part 1.2: Map Initialization

# Initialize your map data structure here as a 2D floating point array
map = np.empty((360,360)) 
waypoints = []

if mode == 'autonomous':
    # Part 3.1: Load path from disk and visualize it
    waypoints = np.load("path.npy")
    waypoints = [[-8.43, -4.67], [-8.00, -4.8], [-7.8, -5], [-7.6, -5.2], [-7.2, -5.4], [-7.0, -5.4], [-6.8, -5.4], [-6.6, -5.4], [-6.5, -5.4], [-6.4, -5.4], [-6.4, -5.6], [-6.4, -5.8], [-6.3, -6], [-6.3, -6.2], [-6.3, -6.4], [-6.3, -7], [-6.3, -7.5], [-6.3, -8], [-6.3, -8.5], [-6.3, -8.75], [-6.5, -9], [-6.7, -9.2], [-6.9, -9.4], [-7, -10]]
    target_pose = waypoints[1]#waypoints[10]
    point_count = 0
    prev_DE = 0
    direction = 1
    for_a_few = 0
    STOP_TURNING = 0
state = 0 # use this to iterate through your path

while robot.step(timestep) != -1 and mode != 'planner':

    ###################
    #
    # Mapping
    #
    ###################

    ################ v [Begin] Do not modify v ##################
    # Ground truth pose
    pose_y = gps.getValues()[1]
    pose_x = gps.getValues()[0]

    n = compass.getValues()
    rad = ((math.atan2(n[0], -n[2])))#-1.5708)
    pose_theta = rad
    
    lidar_sensor_readings = lidar.getRangeImage()
    lidar_sensor_readings = lidar_sensor_readings[83:len(lidar_sensor_readings)-83]
    
    for i, rho in enumerate(lidar_sensor_readings):
        alpha = lidar_offsets[i]

        if rho > LIDAR_SENSOR_MAX_RANGE:
            continue

        # The Webots coordinate system doesn't match the robot-centric axes we're used to
        rx = -math.cos(alpha)*rho + 0.202
        ry = math.sin(alpha)*rho -0.004


        # Convert detection from robot coordinates into world coordinates
        wx =  math.cos(pose_theta)*rx - math.sin(pose_theta)*ry + pose_x
        wy =  +(math.sin(pose_theta)*rx + math.cos(pose_theta)*ry) + pose_y


    
        ################ ^ [End] Do not modify ^ ##################

        #print("Rho: %f Alpha: %f rx: %f ry: %f wx: %f wy: %f" % (rho,alpha,rx,ry,wx,wy))

        if rho < LIDAR_SENSOR_MAX_RANGE:
            # Part 1.3: visualize map gray values.

            # You will eventually REPLACE the following 3 lines with a more robust version of the map
            # with a grayscale drawing containing more levels than just 0 and 1.

            # print("wx: " + str(wx * 30))
            # print("wy: " + str(wy * 30))

            x_coor = int(wx*30)
            y_coor = 360-int(wy*30)

            if x_coor > 359:
                x_coor = 359
            if y_coor > 359:
                y_coor = 359
            
            
            if map[x_coor][y_coor] < 1:
                map[x_coor][y_coor] += 5e-3 

            #map_temp[int(wx*30)][360-int(wy*30)] = min(map_temp[int(wx*30)][360-int(wy*30)] + 5e-3, 1)
            
            g = map[x_coor][y_coor]

            color = int( (g * 256**2 + g * 256 + g) * 255)

            #print(color)
            #we need to reject all values less that .5
            if map[x_coor][y_coor] > 0.5:
                #map = map * 1
                display.setColor(int(color))
                display.drawPixel(x_coor,y_coor)


            # display.setColor(color)
            # if map[int(wx*30)][360-int(wy*30)] >= .5:
            #     display.drawPixel(int(wx*30),360-int(wy*30))
            

    # Draw the robot's current pose on the 360x360 display
    display.setColor(int(0xFF0000))
    
    #print(pose_x,pose_y,pose_theta)
    display.drawPixel(int(pose_x*30),360-int(pose_y*30))



    ###################
    #
    # Controller
    #
    ###################
    #print("first")
    if mode == 'manual':
        key = keyboard.getKey()
        while(keyboard.getKey() != -1): pass
        if key == keyboard.LEFT :
            vL = -MAX_SPEED
            vR = MAX_SPEED
        elif key == keyboard.RIGHT:
            vL = MAX_SPEED
            vR = -MAX_SPEED
        elif key == keyboard.UP:
            vL = MAX_SPEED
            vR = MAX_SPEED
        elif key == keyboard.DOWN:
            vL = -MAX_SPEED
            vR = -MAX_SPEED
        elif key == ord(' '):
            vL = 0
            vR = 0
        elif key == ord('S'):
            # Part 1.4: Filter map and save to filesystem
            
            np.save('map.npy',map)
            print("Map file saved")
        elif key == ord('L'):
            # You will not use this portion in Part 1 but here's an example for loading saved a numpy array
            map = np.load("map.npy")
            map_config = copy.copy(map)
            print("Map loaded")
                        
        else: # slow down
            vL *= .75
            vR *= .75
    else: # not manual mode
        # Part 3.2: Feedback controller
        #STEP 1: Calculate the error
        #print(point_count)
        
        # if (math.sqrt((pose_x - target_pose[0])**2+(pose_y - target_pose[1])**2)) < 0.05 and point_count < len(waypoints) - 1:
        #     print("Point + 1")
        #     point_count+=1
        #     target_pose = waypoints[point_count]
        # elif point_count >= len(waypoints):
        #     print("end of path!")

        
        if math.sqrt((pose_x - target_pose[0])**2+(pose_y - target_pose[1])**2) < 0.1 and point_count < len(waypoints) - 1:
            point_count+=1
            target_pose = waypoints[point_count]
            print("new target pose is: ", target_pose)
            print("step: ", point_count, " of ", len(waypoints))
        elif point_count >= len(waypoints):
            print("end of path!")

        distance_error = math.sqrt((pose_x - target_pose[0])**2+(pose_y - target_pose[1])**2)
        if target_pose[0] < pose_x:   
            bearing_error = math.atan2((target_pose[1] - pose_y),(target_pose[0] - pose_x)) - pose_theta + math.pi
            theta_should_be = math.atan2((target_pose[1] - pose_y),(target_pose[0] - pose_x)) + math.pi
        else:
            bearing_error = math.atan2((target_pose[1] - pose_y),(target_pose[0] - pose_x)) - pose_theta
            theta_should_be = math.atan2((target_pose[1] - pose_y),(target_pose[0] - pose_x))
        heading_error = bearing_error

        
        print("target_pose: ", target_pose)
        print("Current pose X: ", pose_x, " Y: ", pose_y, " Theta: ", pose_theta)
        print("distance_error: ", distance_error)
        print("heading_error: ", heading_error)
        print("theta should be: ", theta_should_be)
        #if target_pose[1] - pose_y < 0.1 and target_pose[1] - pose_y < 0.1: 
        #STEP 2: Controller
        if distance_error > 0.015:
            distance_constant = .2
            if distance_error > 0.05:
                phi_l = (distance_error*distance_constant - (bearing_error*AXLE_LENGTH)/2)/AXLE_RADIUS
                phi_r = (distance_error*distance_constant + (bearing_error*AXLE_LENGTH)/2)/AXLE_RADIUS
            else:
                phi_l = (distance_error - (heading_error*AXLE_LENGTH)/2)/AXLE_RADIUS
                phi_r = (distance_error + (heading_error*AXLE_LENGTH)/2)/AXLE_RADIUS
                print("point found")
                point_count+=1
                robot_parts[MOTOR_LEFT].setVelocity(0)
                robot_parts[MOTOR_RIGHT].setVelocity(0)
        #STEP 3: Compute wheelspeeds
            if phi_l < phi_r:
                vL = (MAX_SPEED/4) * (phi_l/phi_r)
                vR = (MAX_SPEED/4)
            elif phi_l > phi_r:
                vL = (MAX_SPEED/4)
                vR = (MAX_SPEED/4) * (phi_r/phi_l)
            else:
                vL = MAX_SPEED/2
                vR = MAX_SPEED/2
            #if prev_DE - distance_error < 0 and for_a_few <= 0:
            #    direction = -direction
            #    for_a_few = 100
            vL = vL * direction
            vR = vR * direction
            prev_DE = distance_error
            for_a_few -= 1 #so it gives the robot a second to change directions
        # (Keep the wheel speeds a bit less than the actual platform MAX_SPEED to minimize jerk)


    # Odometry code. Don't change vL or vR speeds after this line.
    # We are using GPS and compass for this lab to get a better pose but this is how you'll do the odometry
    #pose_x += (vL+vR)/2/MAX_SPEED*MAX_SPEED_MS*timestep/1000.0*math.cos(pose_theta)
    #pose_y -= (vL+vR)/2/MAX_SPEED*MAX_SPEED_MS*timestep/1000.0*math.sin(pose_theta)
    #pose_theta += (vR-vL)/AXLE_LENGTH/MAX_SPEED*MAX_SPEED_MS*timestep/1000.0

    # print("X: %f Z: %f Theta: %f" % (pose_x, pose_y, pose_theta))

    # Actuator commands
    robot_parts[MOTOR_LEFT].setVelocity(vL)
    robot_parts[MOTOR_RIGHT].setVelocity(vR)
