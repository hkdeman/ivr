#!/usr/bin/env python2.7
import gym
import reacher3D.Reacher
import numpy as np
import cv2
import math
import scipy as sp
import collections
import time
from PIL import Image
from copy import deepcopy
import glob
from matplotlib import pyplot as plt
from enum import Enum
from sklearn.metrics import accuracy_score
import numpy.linalg as npl

class Colors(Enum):
    RED = "RED"
    GREEN = "GREEN"
    BLUE = "BLUE"
    DARK_BLUE = "DARK_BLUE"


class MainReacher():
    def __init__(self):
        self.env = gym.make('3DReacherMy-v0')
        self.env.reset()

    def coordinate_convert(self,pixels):
        #Converts pixels into metres
        return np.array([(pixels[0]-self.env.viewerSize/2)/self.env.resolution,-(pixels[1]-self.env.viewerSize/2)/self.env.resolution])

    def peak_pick(self, image):
        img_hist=cv2.calcHist([image],[0],None,[256],[0,256])

        #Find the largest peak in histogram
        peak = np.argmax(img_hist)

        #Find the largest peak on the darker side
        img_hist_darker=img_hist[0:peak-1]
        peak_darker = np.argmax(img_hist_darker)

        #Find the deepest valley in between these peaks
        values = img_hist_darker[peak_darker+1:]
        thresh = np.argmin(values)+peak_darker+1

        return thresh

    def detect_color(self, image_xy, image_xz, thresholds, color):
        if color == Colors.RED.value:
            # red
            mask_xy = cv2.inRange(image_xy, (thresholds[0],0,0),(255,thresholds[1]-1,thresholds[2]-1))
            mask_xz = cv2.inRange(image_xz, (thresholds[0],0,0),(255,thresholds[1]-1,thresholds[2]-1))
        elif color == Colors.GREEN.value:
            # green
            mask_xy = cv2.inRange(image_xy, (0, thresholds[1], 0),(thresholds[0]-1, 255, thresholds[2]-1))
            mask_xz = cv2.inRange(image_xz, (0, thresholds[1], 0),(thresholds[0]-1, 255, thresholds[2]-1))

        elif color == Colors.BLUE.value:
            # blue
            mask_xy = cv2.inRange(image_xy, (0,0,thresholds[2]+180),(thresholds[0]-1,thresholds[1]-1, 255))
            mask_xz = cv2.inRange(image_xz, (0,0,thresholds[2]+180),(thresholds[0]-1,thresholds[1]-1, 255))

        elif color == Colors.DARK_BLUE.value:
            # dark blue
            mask_xy = cv2.inRange(image_xy, (0,0,thresholds[2]), (thresholds[0]-1, thresholds[1]-1, 180))
            mask_xz = cv2.inRange(image_xz, (0,0,thresholds[2]), (thresholds[0]-1, thresholds[1]-1, 180))
        else:
            raise("Color not found")

        kernel_xy = np.ones((5,5),np.uint8)
        mask_xy = cv2.dilate(mask_xy, kernel_xy, iterations=2)
        mask_xy = cv2.erode(mask_xy, kernel_xy, iterations=3)
        M_xy = cv2.moments(mask_xy)

        if M_xy['m00'] != 0:
            cx_xy = int(M_xy['m10']/M_xy['m00'])
            cy_xy = int(M_xy['m01']/M_xy['m00'])
        else:
            cx_xy, cy_xy = 0,0

        kernel_xz = np.ones((5,5),np.uint8)
        mask_xz = cv2.dilate(mask_xz, kernel_xz, iterations=2)
        mask_xz = cv2.erode(mask_xz, kernel_xz, iterations=3)
        M_xz = cv2.moments(mask_xz)

        if M_xz['m00'] != 0:
            cx_xz = int(M_xz['m10']/M_xz['m00'])
            cz_xz = int(M_xz['m01']/M_xz['m00'])
        else:
            cx_xz, cz_xz = 0,0

        y_coord = self.coordinate_convert(np.array([cx_xy,cy_xy]))
        z_coord = self.coordinate_convert(np.array([cx_xz,cz_xz]))

        x, y, z = y_coord[0], y_coord[1], z_coord[1]

        return np.array([np.asscalar(x),np.asscalar(y),np.asscalar(z)])

    def angle_normalize(self, x):
        return (((x+np.pi) % (2*np.pi)) - np.pi)

    def detect_joint_angles(self, image_xy, image_xz, thresholds):
        joint1 = self.detect_color(image_xy, image_xz, thresholds, Colors.RED.value)
        joint2 = self.detect_color(image_xy, image_xz, thresholds, Colors.GREEN.value)
        joint3 = self.detect_color(image_xy, image_xz, thresholds, Colors.BLUE.value)
        joint4 = self.detect_color(image_xy, image_xz, thresholds, Colors.DARK_BLUE.value)

        # joint1 only rotates over y axes
        y_unit_vector = np.array([0,0,0])
        ja1 = math.acos(np.dot(joint1, y_unit_vector)/(np.linalg.norm(joint1)*np.linalg.norm(y_unit_vector)))

        return self.env.ground_truth_joint_angles


    def show_joints_with_details(self, image_xy, image_xz, thresholds):
        new_img = deepcopy(image_xy)
        # new_img = self.remove_illumination(new_img)

        cx, cy = self.detect_color(image_xy, image_xz, thresholds, Colors.RED.value)
        cv2.rectangle(new_img, (cx-30, cy-30), (cx+30, cy+30), (0,0,0), 2)
        cx, cy = self.detect_color(image_xy, image_xz, thresholds, Colors.GREEN.value)
        cv2.rectangle(new_img, (cx-30, cy-30), (cx+30, cy+30), (0,0,0), 2)
        cx, cy = self.detect_color(image_xy, image_xz, thresholds, Colors.BLUE.value)
        cv2.rectangle(new_img, (cx-30, cy-30), (cx+30, cy+30), (0,0,0), 2)
        cx, cy = self.detect_color(image_xy, image_xz, thresholds, Colors.DARK_BLUE.value)
        cv2.rectangle(new_img, (cx-30, cy-30), (cx+30, cy+30), (0,0,0), 2)

        cv2.imshow("img", new_img)

    def detect_target(self, image_xy, image_xz):
        template_data_xy= []
        template_data_xz = []
        #make a list of all template images from a directory
        xy_files= glob.glob('valid/validxy*.jpg')
        xz_files = glob.glob('valid/validxz*.jpg')

        for files in xy_files:
            image = cv2.imread(files, 0)
            template_data_xy.append(image)

        for files in xz_files:
            image = cv2.imread(files, 0)
            template_data_xz.append(image)

        w = h = 40

        best_top_left_xy = 0
        best_bottom_right_xy = 0
        best_max_val_xy = -1
        mask_xy = cv2.inRange(image_xy, (175,175,175), (180,180,180))

        for template in template_data_xy:
            img = mask_xy.copy()
            result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)

            if max_val > best_max_val_xy:
                best_max_val_xy = max_val
                best_top_left_xy = top_left
                best_bottom_right_xy = bottom_right

        best_top_left_xz = 0
        best_bottom_right_xz = 0
        best_max_val_xz = -1
        mask_xz = cv2.inRange(image_xz, (175,175,175), (180,180,180))
        for template in template_data_xz:
            img = mask_xz.copy()
            result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)

            if max_val > best_max_val_xz:
                best_max_val_xz = max_val
                best_top_left_xz = top_left
                best_bottom_right_xz = bottom_right


        img_xy  = image_xy.copy()
        img_xz = image_xz.copy()

        cx_xy, cy_xy = best_top_left_xy[0]+best_bottom_right_xy[0] // 2, best_top_left_xy[1]+best_bottom_right_xy[1] // 2
        cx_xz, cz_xz = best_top_left_xz[0]+best_bottom_right_xz[0] // 2, best_top_left_xz[1]+best_bottom_right_xz[1] // 2

        cv2.rectangle(img_xy, best_bottom_right_xy, best_top_left_xy, 255, 2)
        cv2.rectangle(img_xz, best_top_left_xz, best_bottom_right_xz, 255, 2)
        cv2.imshow("xy", img_xy)
        cv2.imshow("xz", img_xz)
        cv2.waitKey(0)


        y_coord = self.coordinate_convert(np.array([cx_xy, cy_xy]))
        z_coord = self.coordinate_convert(np.array([cx_xz, cz_xz]))

        x, y, z = y_coord[0], y_coord[1], z_coord[1]

        return np.array([np.asscalar(x),np.asscalar(y),np.asscalar(z)])

    def rotation_matrix_Y(self, angle):
        return np.matrix([[np.cos(angle), 0, -np.sin(angle)],
                          [0, 1, 0],
                          [np.sin(angle), 0, np.cos(angle)]])

    def rotation_matrix_Z(self, angle):
        return np.matrix([[np.cos(angle), -np.sin(angle), 0],
                          [np.sin(angle), np.cos(angle), 0],
                          [0, 0, 1]])

    def link_transform_Y(self, angle):
        rotation = np.matrix(np.eye(4, 4))
        rotation[0:3, 0:3] = self.rotation_matrix_Y(angle)
        translation = np.matrix(np.eye(4,4))
        translation[0, 3] = 1
        return rotation*translation

    def link_transform_Z(self, angle):
        rotation = np.matrix(np.eye(4, 4))
        rotation[0:3, 0:3] = self.rotation_matrix_Z(angle)
        translation = np.matrix(np.eye(4,4))
        translation[0, 3] = 1
        return rotation*translation

    def FK(self, joint_angles):
        t1 = self.link_transform_Y(joint_angles[0])
        t2 = self.link_transform_Z(joint_angles[1])
        t3 = self.link_transform_Z(joint_angles[2])
        t4 = self.link_transform_Y(joint_angles[3])
        total_transform = (t1*t2*t3*t4)
        return total_transform


    def Jacobian(self, joint_angles):
        t1 = self.link_transform_Y(joint_angles[0])
        t2 = self.link_transform_Z(joint_angles[1])
        t3 = self.link_transform_Z(joint_angles[2])
        t4 = self.link_transform_Y(joint_angles[3])

        total_transform = (t1*t2*t3*t4)
        ee_pos = total_transform[0:3, 3]
        j4_pos = (t1*t2*t3)[0:3, 3]
        j3_pos = (t1*t2)[0:3, 3]
        j2_pos = (t1)[0:3, 3]
        j1_pos = np.zeros((3, 1))

        jacobian = np.zeros((3,4))
        pos_3D = np.zeros(3)

        y_vector = np.array([0,-1,0])
        z_vector = np.array([0,0,1])
        pos_3D = (ee_pos-j1_pos).T
        y_vector = np.squeeze(np.array(self.rotation_matrix_Y(joint_angles[0])* np.matrix(y_vector).T))
        jacobian[0:3,0] = np.cross(y_vector,pos_3D)

        
        y_vector = np.squeeze(np.array(self.rotation_matrix_Y(joint_angles[0])* np.matrix(y_vector).T))
        z_vector = np.squeeze(np.array(self.rotation_matrix_Y(joint_angles[0])* np.matrix(z_vector).T))
        pos_3D[0:3] = (ee_pos-j2_pos).T
        jacobian[0:3,1] = np.cross(z_vector,pos_3D)

        y_vector = np.array([0,-1,0])
        z_vector = np.array([0,0,1])

        y_vector = np.squeeze(np.array((self.rotation_matrix_Y(joint_angles[0]) * self.rotation_matrix_Z(joint_angles[1]))* np.matrix(y_vector).T))
        z_vector = np.squeeze(np.array((self.rotation_matrix_Y(joint_angles[0]) * self.rotation_matrix_Z(joint_angles[1]))* np.matrix(z_vector).T))

        pos_3D[0:3] = (ee_pos-j3_pos).T
        jacobian[0:3,2] = np.cross(z_vector,pos_3D)

        y_vector = np.array([0,-1,0])
        z_vector = np.array([0,0,1])

        y_vector = np.squeeze(np.array((self.rotation_matrix_Y(joint_angles[0]) * self.rotation_matrix_Z(joint_angles[1]) * self.rotation_matrix_Z(joint_angles[2]))* np.matrix(y_vector).T))
        z_vector = np.squeeze(np.array((self.rotation_matrix_Y(joint_angles[0]) * self.rotation_matrix_Z(joint_angles[1]) * self.rotation_matrix_Z(joint_angles[2]))* np.matrix(z_vector).T))

        pos_3D[0:3] = (ee_pos-j4_pos).T
        jacobian[0:3,3] = np.cross(y_vector,pos_3D)
        #jacobian[3, :] = 1
        #print (jacobian)

        return jacobian

    def IK(self, current_joint_angles, desired_position):

        curr_pos = self.FK(current_joint_angles)[0:3,3]
        pos_error = desired_position - np.squeeze(np.array(curr_pos.T))

        Jac = np.matrix(self.Jacobian(current_joint_angles))[0:3,:]
        Jac_inv = Jac.T
        if(np.linalg.matrix_rank(Jac,0.4)<3):
        	Jac_inv = Jac.T
        else:
        	Jac_inv = Jac.T*np.linalg.inv(Jac * Jac.T)

        q_dot = Jac_inv*np.matrix(pos_error).T
        #print (q_dot)
        return np.squeeze(np.array(q_dot.T))

    def update_thresholds(self, image_foreground_xy):
        #Calculate the initial thresholds
        thresholds = np.zeros([3])

        #First isolate the objects in the foreground by removing the background white pixels
        thresh_red = self.peak_pick(image_foreground_xy[:,:,0])
        thresh_green = self.peak_pick(image_foreground_xy[:,:,1])
        thresh_blue = self.peak_pick(image_foreground_xy[:,:,2])

        mask_white = cv2.inRange(image_foreground_xy, (thresh_red, thresh_green, thresh_blue), (255, 255, 255))+1
        image_foreground_xy[:,:,0] *= mask_white
        image_foreground_xy[:,:,1] *= mask_white
        image_foreground_xy[:,:,2] *= mask_white

        #Now calculate the thresholds for the objects in the foreground
        thresholds[0] = self.peak_pick(image_foreground_xy[:,:,0])
        thresholds[1] = self.peak_pick(image_foreground_xy[:,:,1])
        thresholds[2] = self.peak_pick(image_foreground_xy[:,:,2])

        return thresholds

    def go(self):
        #The robot has several simulated modes:
        #These modes are listed in the following format:
        #Identifier (control mode) : Description : Input structure into step function

        #POS : A joint space position control mode that allows you to set the desired joint angles and will position the robot to these angles : env.step((np.zeros(3),np.zeros(3), desired joint angles, np.zeros(3)))
        #POS-IMG : Same control as POS, however you must provide the current joint angles and velocities : env.step((estimated joint angles, estimated joint velocities, desired joint angles, np.zeros(3)))
        #VEL : A joint space velocity control, the inputs require the joint angle error and joint velocities : env.step((joint angle error (velocity), estimated joint velocities, np.zeros(3), np.zeros(3)))
        #TORQUE : Provides direct access to the torque control on the robot : env.step((np.zeros(3),np.zeros(3),np.zeros(3),desired joint torques))
        self.env.controlMode="VEL"
        #Run 100000 iterations
        prev_jas = np.zeros(4)
        prev_jvs = np.zeros(4)

        #self.env.render returns 2 RGB arrays of the robot, one for the xy-plane, and one for the xz-plane
        image_foreground_xy, image_foreground_xz = self.env.render('rgb-array')
        # target = self.detect_target(image_foreground_xy, image_foreground_xz)
        thresholds = self.update_thresholds(image_foreground_xy)

        # print(target)
        # print(self.env.ground_truth_valid_target)
        target = self.env.ground_truth_valid_target
        prev_success = self.env.success
        for i in range(100000):
            #The change in time between iterations can be found in the self.env.dt variable
            dt = self.env.dt

            self.env.step((np.zeros(4), np.zeros(4), np.array([20,5,5,5]), np.zeros(4)))

            # velocities
            prev_jvs=(self.angle_normalize(self.env.ground_truth_joint_angles-prev_jas))
            detectedJointVels = (prev_jvs/dt)
            prev_jas = self.env.ground_truth_joint_angles

            jointAnglesIK = self.IK(self.env.ground_truth_joint_angles, target)
            self.env.step((jointAnglesIK, detectedJointVels, np.zeros(3), np.zeros(3)))  #For the velocity control

            image_foreground_xy, image_foreground_xz = self.env.render('rgb-array')
            detected_joint_angles = self.detect_joint_angles(image_foreground_xy, image_foreground_xz, thresholds = thresholds)
            thresholds = self.update_thresholds(image_foreground_xy)

            if(self.env.success > prev_success):
                target = self.env.ground_truth_valid_target

        # self.show_joints_with_details(*self.env.render('rgb-array'), thresholds= thresholds)
#main method
def main():
    reach = MainReacher()
    reach.go()

if __name__ == "__main__":
    main()
