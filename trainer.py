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


class reinforcement_hover:
    

    def __init__(self):
        yaml_file = 'params.yaml'
        settings = {}
        stream = open(yaml_file, "r")
        docs = yaml.load_all(stream)
        for doc in docs:
            for k,v in doc.items():
                settings[k] = v
        self.counter = 0
        self.agent = A2CContinuous(settings)
        self.action_executed = TwistStamped()
        self.target = PoseStamped()
        self.target.pose.position.x = 20
        self.target.pose.position.y = 15
        self.target.pose.position.z = 5

        rospy.set_param("/mavros/conn/heartbeat_rate", '3.0');
        self.vel_pub = rospy.Publisher('/mavros/setpoint_velocity/cmd_vel', TwistStamped)
        vel_sub = message_filters.Subscriber('/mavros/local_position/velocity', TwistStamped)
        pos_sub = message_filters.Subscriber('/mavros/local_position/pose', PoseStamped)
        ts = message_filters.TimeSynchronizer([vel_sub, pos_sub], 10)
        ts.registerCallback(self.trainer) 
        attPub = rospy.Publisher('mavros/setpoint_attitude/attitude', PoseStamped, queue_size=10)
        arming_srv = rospy.ServiceProxy("/mavros/cmd/arming", CommandBool)
        mode_srv = rospy.ServiceProxy("/mavros/set_mode", SetMode)
        v_p = rospy.Publisher('/mavros/setpoint_attitude/cmd_vel', TwistStamped, queue_size=10)
        self.action_executed = TwistStamped()
        print "Setting Offboard Mode"
        result = mode_srv(custom_mode="OFFBOARD")
        print result
        print "Arming"
        result = arming_srv(value=True)

    def trainer(self,vel, pos):
        self.counter += 1
        state = np.zeros((1,13))
        print self.counter
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
        self.reward = self.calc_reward(state[0,0], state[0,1], self.target, state[0,7], state[0,8],state[0,9])
        #self.old_pose.pose.position.x = state[0,0]
        #self.old_pose.pose.position.y = state[0,1]

        print self.reward
        state = np.reshape(state,(13,))
        
        action = self.agent.choose_action(state)
        #pdb.set_trace()

        self.action_executed.twist.linear.x =  np.random.normal(loc = action[0],scale = action[1])
        self.action_executed.twist.linear.y =  np.random.normal(loc = action[0],scale = action[1])
        #self.action_executed.twist.linear.z =  np.random.normal(loc = action[0],scale = action[1])
        #self.action_executed.twist.angular.x =  np.random.normal(loc = action[0],scale = action[1])
        #self.action_executed.twist.angular.y =  np.random.normal(loc = action[0],scale = action[1])
        #self.action_executed.twist.angular.z =  np.random.normal(loc = action[0],scale = action[1])

        self.vel_pub.publish(self.action_executed)
        self.agent.learner(state,self.reward, action[2])

    def calc_reward(self,new_x,new_y,old_pose,velocity_x, velocity_y, velocity_z):
            reward = np.square((new_x - old_pose.pose.position.x)) + np.square((new_y - old_pose.pose.position.y)) 
            vel_reward = np.square(velocity_x) + np.square(velocity_y) + np.square(velocity_z)
            total_reward = 0.6*reward + 0.4*vel_reward
            return (-total_reward)


def main():
        rospy.init_node('f450_velocity_controller', anonymous=True)
        t_ = reinforcement_hover()
        rospy.spin()


if __name__=='__main__':
        main()  


