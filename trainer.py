import rospy
from mavros_msgs.msg import State, ExtendedState
from mavros_msgs.srv import CommandBool, SetMode
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion
import math
import numpy as np
from geometry_msgs.msg import TwistStamped
import pdb
import random
import time 
from tf.transformations import euler_from_quaternion
from tf.transformations import quaternion_from_euler
import message_filters
import yaml
import pdb

from os.path import join, exists
from os import makedirs
from network import A2CContinuous
#initialize network

arming_srv = rospy.ServiceProxy("/mavros/cmd/arming", CommandBool)
mode_srv = rospy.ServiceProxy("/mavros/set_mode", SetMode)

result = mode_srv(custom_mode="OFFBOARD")
result = arming_srv(value=True)
yaml_file = 'params.yaml'

settings = {}
stream = open(yaml_file, "r")
docs = yaml.load_all(stream)
for doc in docs:
  for k,v in doc.items():
    settings[k] = v
pdb.set_trace()

agent = A2CContinuous(settings)
action_executed = TwistStamped()

def trainer(vel, pos):
    state = np.zeros((1,13))
    print "this is good boy"
    state[0,0]  = pos.pose.position.x
    state[0,1]  = pos.pose.position.y
    state[0,2]  = pos.pose.position.z
    state[0,3]  = pos.pose.orientation.x
    state[0,4]  = pos.pose.orientation.y
    state[0,5]  = pos.pose.orientation.z
    state[0,6]  = pos.pose.orientation.w
    state[0,7]  = vel.twist.linear.x
    state[0,8]  = vel.twist.linear.y
    state[0,9]  = vel.twist.linear.z
    state[0,10] = vel.twist.angular.x
    state[0,11] = vel.twist.angular.y
    state[0,12] = vel.twist.angular.z
    pdb.set_trace()
    print state
    state = np.reshape(state,(13,))
    pdb.set_trace()
    action = agent.choose_action(state)[0]
    pdb.set_trace()
    action_executed.twist.linear.x =  action
    vel_pub.publish(action_executed)
    pdb.set_trace() 
    
    #publish these states and observe the reward 
    #TODO write code for publishing the state (start with vel x and vel y)
    #TODO write code for computing reward (for now simple (x2-x1)^2 + (y2-y1)^2)
    #the x2 and y2 are cordinates for the target x and target y. Since we want 
    #maximization of reward, we put a negative sign in front of it 
    agent.learner(reward, states, action_executed)

    
    #print vel
    #print "-----------------------------"
    #print pos
    
rospy.init_node('f450_velocity_controller', anonymous=True)
rospy.set_param("/mavros/conn/heartbeat_rate", '3.0');
vel_pub = rospy.Publisher('/mavros/setpoint_velocity/cmd_vel', TwistStamped)
vel_sub = message_filters.Subscriber('/mavros/local_position/velocity', TwistStamped)
pos_sub = message_filters.Subscriber('/mavros/local_position/pose', PoseStamped)
ts = message_filters.TimeSynchronizer([vel_sub, pos_sub], 10)
ts.registerCallback(trainer) 
attPub = rospy.Publisher('mavros/setpoint_attitude/attitude', PoseStamped, queue_size=10)
arming_srv = rospy.ServiceProxy("/mavros/cmd/arming", CommandBool)
mode_srv = rospy.ServiceProxy("/mavros/set_mode", SetMode)
v_p = rospy.Publisher('/mavros/setpoint_attitude/cmd_vel', TwistStamped, queue_size=10)


rospy.spin()