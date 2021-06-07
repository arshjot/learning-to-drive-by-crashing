#!/usr/bin/env python  

from random import uniform
import rospy
import rospkg
import tf

import numpy as np
from gazebo_msgs.srv import *
from geometry_msgs.msg import *
from sensor_msgs.msg import *


if __name__ == '__main__':
    print("Waiting for gazebo services...")
    rospy.init_node("spawn_products_in_bins")
    rospy.wait_for_service("gazebo/delete_model")
    rospy.wait_for_service("gazebo/spawn_sdf_model")
    print("Got it.")
    delete_model = rospy.ServiceProxy("gazebo/delete_model", DeleteModel)
    spawn_model = rospy.ServiceProxy("gazebo/spawn_sdf_model", SpawnModel)

    model_path = rospkg.RosPack().get_path('learning_to_drive')+'/launch/models/'
    model_xml = ''

    q = tf.transformations.quaternion_from_euler(0.0, 0.0, 0.0)
    orient = Quaternion(*q)

    # spawn cones
    with open(model_path + '/cone_2.sdf', 'r') as xml_file:
        cone_model_xml = xml_file.read().replace('\n', '')
         
    # spawn beer cans
    with open(model_path + '/beer.sdf', 'r') as xml_file:
        can_model_xml = xml_file.read().replace('\n', '')

    # spawn boxes of different colors
    with open(model_path + '/black_box_2.sdf', 'r') as xml_file:
        blackbox_model_xml = xml_file.read().replace('\n', '')

    with open(model_path + '/blue_box_1.sdf', 'r') as xml_file:
        bluebox_model_xml = xml_file.read().replace('\n', '')
    
    with open(model_path + '/grey_box_2.sdf', 'r') as xml_file:
        greybox_model_xml = xml_file.read().replace('\n', '')

    with open(model_path + '/purple_box_1.sdf', 'r') as xml_file:
        purplebox_model_xml = xml_file.read().replace('\n', '')

    with open(model_path + '/red_box_2.sdf', 'r') as xml_file:
        redbox_model_xml = xml_file.read().replace('\n', '')
                
    with open(model_path + '/white_box_1.sdf', 'r') as xml_file:
        whitebox_model_xml = xml_file.read().replace('\n', '')

    with open(model_path + '/yellow_box_2.sdf', 'r') as xml_file:
        yellowbox_model_xml = xml_file.read().replace('\n', '')

    with open(model_path + '/turquoise_box_1.sdf', 'r') as xml_file:
        turquoisebox_model_xml = xml_file.read().replace('\n', '')



     # Spawn the models

    for i in range(0,40):
        # spawn cones
        bin_x = np.random.uniform(-4, 4)
        bin_y = np.random.uniform(-4, 4)
        item_name   =   "cone_{0}".format(i)
        item_pose   =   Pose(Point(x=bin_x, y=bin_y,    z=0.25),   orient)
        spawn_model(item_name, cone_model_xml, "", item_pose, "world")

    for i in range(0,40):
         # spawn beer cans
        bin_x = np.random.uniform(-4, 4)
        bin_y = np.random.uniform(-4, 4)
        item_name   =   "beer_{0}".format(i)
        item_pose   =   Pose(Point(x=bin_x, y=bin_y,    z=0.25),   orient)
        spawn_model(item_name, can_model_xml, "", item_pose, "world")

    for i in range(0,200):
        # spawn a box with a random color
        color = np.random.randint(0,8)

        # black
        if color==0:
            bin_x = np.random.uniform(-4, 4)
            bin_y = np.random.uniform(-4, 4)
            item_name   =   "box_{0}".format(i)
            # print("Spawning model:%s", item_name)
            item_pose   =   Pose(Point(x=bin_x, y=bin_y,    z=0.0375),   orient)
            if (abs(bin_x) > 0.15) and (abs(bin_y) > 0.15):
	            spawn_model(item_name, blackbox_model_xml, "", item_pose, "world")

        # blue
        if color==1:
            bin_x = np.random.uniform(-4, 4)
            bin_y = np.random.uniform(-4, 4)
            item_name   =   "box_{0}".format(i)
            # print("Spawning model:%s", item_name)
            item_pose   =   Pose(Point(x=bin_x, y=bin_y,    z=0.05),   orient)
            if (abs(bin_x) > 0.15) and (abs(bin_y) > 0.15):
	            spawn_model(item_name, bluebox_model_xml, "", item_pose, "world")
        
        # grey
        if color==2:
            bin_x = np.random.uniform(-4, 4)
            bin_y = np.random.uniform(-4, 4)
            item_name   =   "box_{0}".format(i)
            # print("Spawning model:%s", item_name)
            item_pose   =   Pose(Point(x=bin_x, y=bin_y,    z=0.0375),   orient)
            if (abs(bin_x) > 0.15) and (abs(bin_y) > 0.15):
	            spawn_model(item_name, greybox_model_xml, "", item_pose, "world")

        # purple
        if color==3:
            bin_x = np.random.uniform(-4, 4)
            bin_y = np.random.uniform(-4, 4)
            item_name   =   "box_{0}".format(i)
            # print("Spawning model:%s", item_name)
            item_pose   =   Pose(Point(x=bin_x, y=bin_y,    z=0.05),   orient)
            if (abs(bin_x) > 0.15) and (abs(bin_y) > 0.15):
	            spawn_model(item_name, purplebox_model_xml, "", item_pose, "world")

        # red
        if color==4:
            bin_x = np.random.uniform(-4, 4)
            bin_y = np.random.uniform(-4, 4)
            item_name   =   "box_{0}".format(i)
            # print("Spawning model:%s", item_name)
            item_pose   =   Pose(Point(x=bin_x, y=bin_y,    z=0.0375),   orient)
            if (abs(bin_x) > 0.15) and (abs(bin_y) > 0.15):
	            spawn_model(item_name, redbox_model_xml, "", item_pose, "world")
        
        # white
        if color==5:
            bin_x = np.random.uniform(-4, 4)
            bin_y = np.random.uniform(-4, 4)
            item_name   =   "box_{0}".format(i)
            # print("Spawning model:%s", item_name)
            item_pose   =   Pose(Point(x=bin_x, y=bin_y,    z=0.05),   orient)
            if (abs(bin_x) > 0.15) and (abs(bin_y) > 0.15):
	            spawn_model(item_name, whitebox_model_xml, "", item_pose, "world")
        
        # yellow
        if color==6:
            bin_x = np.random.uniform(-4, 4)
            bin_y = np.random.uniform(-4, 4)
            item_name   =   "box_{0}".format(i)
            # print("Spawning model:%s", item_name)
            item_pose   =   Pose(Point(x=bin_x, y=bin_y,    z=0.0375),   orient)
            if (abs(bin_x) > 0.15) and (abs(bin_y) > 0.15):
	            spawn_model(item_name, yellowbox_model_xml, "", item_pose, "world")
        
        # turquoise
        if color==7:
            bin_x = np.random.uniform(-4, 4)
            bin_y = np.random.uniform(-4, 4)
            item_name   =   "box_{0}".format(i)
            # print("Spawning model:%s", item_name)
            item_pose   =   Pose(Point(x=bin_x, y=bin_y,    z=0.05),   orient)
            if (abs(bin_x) > 0.15) and (abs(bin_y) > 0.15):
	            spawn_model(item_name, turquoisebox_model_xml, "", item_pose, "world")
