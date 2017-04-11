import rospy
from mavros_msgs.msg import State, ExtendedState
from mavros_msgs.srv import CommandBool, SetMode
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion
import math
import numpy
from geometry_msgs.msg import TwistStamped
from VelocityController import VelocityController
from LandingController import LandingController
import pdb
import random
import time 
from tf.transformations import euler_from_quaternion
from tf.transformations import quaternion_from_euler


class QuadController:
	cur_vel = TwistStamped()
    des_vel = TwistStamped()
    sim_ctr = 1
    des_pose = PoseStamped()
    cur_pose = PoseStamped()
    isReadyToFly = True

	def __init__(self):
        rospy.init_node('f450_velocity_controller', anonymous=True)
        rospy.set_param("/mavros/conn/heartbeat_rate", '3.0');
        vel_pub = rospy.Publisher('/mavros/setpoint_velocity/cmd_vel', TwistStamped, queue_size=10)
        vel_sub = rospy.Subscriber('/mavros/local_position/velocity', TwistStamped, callback=self.vel_cb)
        pos_sub = rospy.Subscriber('/mavros/local_position/pose', PoseStamped, callback=self.pos_cb)
		arming_srv = rospy.ServiceProxy("/mavros/cmd/arming", CommandBool)
		mode_srv = rospy.ServiceProxy("/mavros/set_mode", SetMode)

        isReadyToFly = True
        target = Pose()
        landed = False

    def copy_vel(self, vel):
        copied_vel = TwistStamped()
        copied_vel.header= vel.header
        return copied_vel

    def vel_cb(self, msg):
        # print msg
        self.cur_vel = msg

    def pos_cb(self, msg):
        # print msg
        self.cur_pose = msg

    def state_cb(self,msg):
        if(msg.mode=='OFFBOARD'):
            self.isReadyToFly = True

    def ext_state_cb(self, msg):
        if(msg.landed_state == 1):
            self.landed = True


