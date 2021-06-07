#!/usr/bin/env python

import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, Range
from geometry_msgs.msg import Pose, Twist, Vector3
from tf.transformations import euler_from_quaternion
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import rospkg
import cv2


class ThymioController:
    TURNING = 1
    FORWARD = 2
    BACKWARD = 3

    def __init__(self):
        """Initialization."""

        # initialize the node
        rospy.init_node(
            'data_collect_controller'  # name of the node
        )

        self.name = rospy.get_param('~robot_name')
        self.state = ThymioController.FORWARD
        self.hz = 10

        # self.set_mstate = rospy.ServiceProxy( '/gazebo/set_model_state', SetModelState)

        rospy.sleep(10)

        self.directions = ['left', 'center_left', 'center', 'center_right', 'right']
        self.proximity = [0.2, 0.2, 0.2, 0.2, 0.2]

        # create velocity publisher
        self.velocity_publisher = rospy.Publisher(
            self.name + '/cmd_vel',  # name of the topic
            Twist,  # message type
            queue_size=self.hz  # queue size
        )

        # create proximity subscribers
        for direction in self.directions:
            rospy.Subscriber(
                f'{self.name}/proximity/{direction}',  # name of the topic
                Range,  # message type
                self.update_proximity,  # function that handles incoming messages,
                direction # which sensor
            )

        # create image subscriber
        self.bridge = CvBridge()
        self.write_img_counter_1 = 0
        self.write_img_counter_0 = 0
        self.img_counter = 0
        self.img_save_path = rospkg.RosPack().get_path('learning_to_drive')+'/images/'
        self.img_subscriber = rospy.Subscriber(
            self.name + '/camera/image_raw',  # name of the topic
            Image,  # message type
            self.image_callback  # function that hanldes incoming messages
        )

        # tell ros to call stop when the program is terminated
        rospy.on_shutdown(self.stop)

        # initialize pose to (X=0, Y=0, theta=0)
        self.pose = Pose()

        # initialize linear and angular velocities to 0
        self.velocity = Twist()

        # set node update frequency in Hz
        self.rate = rospy.Rate(self.hz)


    def human_readable_pose2d(self, pose):
        """Converts pose message to a human readable pose tuple."""

        # create a quaternion from the pose
        quaternion = (
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w
        )

        # convert quaternion rotation to euler rotation
        roll, pitch, yaw = euler_from_quaternion(quaternion)

        result = (
            pose.position.x,  # x position
            pose.position.y,  # y position
            yaw  # theta angle
        )

        return result

    def update_proximity(self, data, direction):
        """Updates robot proximity sensor readings"""

        index = self.directions.index(direction)

        # round to milimeters and scale to be in [0, 1]
        self.proximity[index] = round(data.range, 2) / 0.12

    def image_callback(self, data):
        """Reads and saves camera image with calculated labels"""

        try:
            ## get image
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

            ## get label
            # compute angleness
            angleness = np.dot(self.proximity, np.array([1, 2, 0, -2, -1])) / 3

            # compute centerness
            centerness = np.dot(self.proximity, np.array([-1, -1, 4, -1, -1])) / 4

            print(f'Angleness = {angleness}, Centerness = {centerness}')

            # obstacle or not?
            obstacle = 1 if centerness != 0 else 0

            ## write image with label
            if obstacle:
                img_name = f'{self.img_save_path}/{obstacle}_{np.round(angleness, 3)}_{np.round(centerness, 3)}_{self.img_counter}.jpg'
                cv2.imwrite(img_name, cv_image)
                self.write_img_counter_1 += 1
            elif (self.img_counter % 3) == 0:
                img_name = f'{self.img_save_path}/{obstacle}_{np.round(angleness, 3)}_{np.round(centerness, 3)}_{self.img_counter}.jpg'
                cv2.imwrite(img_name, cv_image)
                self.write_img_counter_0 += 1

                print(self.write_img_counter_1, self.write_img_counter_0)
                
            self.img_counter += 1

        except CvBridgeError as e:
            print(e)

        

    def get_control(self):

        # if robot is stuck or about to be stuck, 
        # turn so that robot can start moving again

        # compute angleness
        angular_vel = 0
        angleness = np.dot(self.proximity, np.array([1, 2, 0, -2, -1])) / 3

        # compute centerness
        centerness = np.dot(self.proximity, np.array([-1, -1, 4, -1, -1])) / 4

        if self.state == ThymioController.FORWARD:
            self.turning_steps_completed = 0
            if abs(centerness) > 0.3:
                self.state = ThymioController.TURNING
                self.turning_steps = self.hz - 1
                self.turning_steps_completed = 1
                angular_vel = np.pi * np.sign(angleness)

                vel_msg = Twist(
                    linear=Vector3(
                    -0.1,
                    .0,
                    .0,
                    ),
                    angular=Vector3(
                    .0,
                    .0,
                    angular_vel
                    ))
            else:
                vel_msg = Twist(
                linear=Vector3(
                    .2,  # moves forward .2 m/s
                    .0,
                    .0,
                ),
                angular=Vector3(
                    .0,
                    .0,
                    angleness * np.pi * 2
                ))
        elif self.state == ThymioController.TURNING:
            self.turning_steps -= 1
            self.turning_steps_completed += 1

            angular_vel = np.pi * np.sign(angleness)

            vel_msg = Twist(
                linear=Vector3(
                -0.1,
                .0,
                .0,
                ),
                angular=Vector3(
                .0,
                .0,
                angular_vel
                ))

            if (self.turning_steps == 0) and centerness <= 0.3:
                self.state = ThymioController.FORWARD
            elif self.turning_steps_completed > self.hz:
            	self.state = ThymioController.BACKWARD
            	self.backward_steps = self.hz

    	
        if self.state == ThymioController.BACKWARD:
            self.backward_steps -= 1
            vel_msg = Twist(
            linear=Vector3(
            -.1,
            .0,
            .0,
            ),
            angular=Vector3(
            .0,
            .0,
            .0
            ))

            if self.backward_steps == 0:
                self.state = ThymioController.FORWARD

        return vel_msg


    def run(self):
        """Controls the Thymio."""

        while not rospy.is_shutdown():
            # publish velocity message
            velocity = self.get_control()
            self.velocity_publisher.publish(velocity)

            # sleep until next step
            self.rate.sleep()

    def stop(self):
        """Stops the robot."""

        self.velocity_publisher.publish(
            Twist()  # set velocities to 0
        )

        self.rate.sleep()


if __name__ == '__main__':
    controller = ThymioController()

    try:
        controller.run()
    except rospy.ROSInterruptException as e:
        pass
