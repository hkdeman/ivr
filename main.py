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
            mask_xy = cv2.inRange(image_xy, (0,0,thresholds[2]+220),(thresholds[0]-1,thresholds[1]-1, 255))
            mask_xz = cv2.inRange(image_xz, (0,0,thresholds[2]+220),(thresholds[0]-1,thresholds[1]-1, 255))

        elif color == Colors.DARK_BLUE.value:
            # dark blue
            mask_xy = cv2.inRange(image_xy, (0,0,thresholds[2]), (thresholds[0]-1, thresholds[1]-1, 220))
            mask_xz = cv2.inRange(image_xz, (0,0,thresholds[2]), (thresholds[0]-1, thresholds[1]-1, 220))


        kernel_xy = np.ones((5,5),np.uint8)
        mask_xy = cv2.dilate(mask_xy, kernel_xy, iterations=2)
        mask_xy = cv2.erode(mask_xy, kernel_xy, iterations=3)
        M_xy = cv2.moments(mask_xy)

        if M_xy['m00'] != 0:
            cx_xy = int(M_xy['m10']/M_xy['m00'])
            cy_xy = int(M_xy['m01']/M_xy['m00'])
        else:
            cx_xy, cy_xy = 0,0

        return cx_xy, cy_xy
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

    def detect_dark_blue(self, image, thresholds):
        mask = cv2.inRange(image, (0,0,thresholds[2]), (thresholds[0]-1, thresholds[1]-1, 220))
        kernel = np.ones((5,5),np.uint8)
        mask = cv2.dilate(mask,kernel, iterations=2)
        mask = cv2.erode(mask, kernel, iterations=3)
        M = cv2.moments(mask)

        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
        else:
            cx,cy = 0,0

        # return np.array([cx, cy]) #, self.coordinate_convert(np.array([cx,cy]))

    def detect_red(self, image, thresholds):
        mask = cv2.inRange(image, (thresholds[0],0,0),(255,thresholds[1]-1,thresholds[2]-1))
        kernel = np.ones((5,5),np.uint8)
        mask = cv2.dilate(mask,kernel, iterations=2)
        mask = cv2.erode(mask, kernel, iterations=3)
        M = cv2.moments(mask)

        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
        else:
            cx,cy = 0,0
        # return np.array([cx, cy]) #, self.coordinate_convert(np.array([cx,cy]))

    def detect_green(self, image, thresholds):
        mask = cv2.inRange(image, (0, thresholds[1], 0),(thresholds[0]-1, 255, thresholds[2]-1))
        kernel = np.ones((5,5),np.uint8)
        mask = cv2.dilate(mask,kernel, iterations=2)
        mask = cv2.erode(mask, kernel, iterations=3)
        M = cv2.moments(mask)

        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
        else:
            cx,cy = 0,0

        # return np.array([cx, cy]) #, self.coordinate_convert(np.array([cx,cy]))


    def angle_normalize(self, x):
        return (((x+np.pi) % (2*np.pi)) - np.pi)

    # def detect_goal(self, imagexy, imagexz):
    #     xy_valid_images = []
    #     for filename in glob.glob("./valid/imagexy*.jpg"):
    #         img = cv2.imread(filename)
    #         valid_images.append(img)
    #
    #     xz_valid_images = []
    #     for filename in glob.glob("./valid/imagexz*.jpg"):
    #         img = cv2.imread(filename)
    #         valid_images.append(img)
    #
    #     goal_coords = None
    #     trap_coords = None
    #     method = cv2.cv2.TM_CCOEFF_NORMED
    #     for template in valid_images:
    #         w, h = template.shape[::-1]
    #         new_img = image.copy()
    #         res = cv2.matchTemplate(new_img, template, method)
    #         min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    #         top_left = max_loc
    #         bottom_right = (top_left[0] + w, top_left[1] + h)
    #         cv2.rectangle(new_img,top_left, bottom_right, 255, 2)
    #         img = Image.fromarray(new_img, 'RGB')
    #         img.show()

    def detect_joint_angles(self, image, thresholds, XY=True, debug=False):
        joint1 = self.detect_red(image, thresholds)
        joint2 = self.detect_green(image, thresholds)
        joint3 = self.detect_blue(image, thresholds)
        joint4 = self.detect_dark_blue(image, thresholds)

        ja1 = math.atan2(joint1[1], joint1[0])
        ja2 = math.atan2(joint2[1] - joint1[1], joint2[0] - joint1[0]) - ja1
        ja2 = self.angle_normalize(ja2)
        ja3 = math.atan2(joint3[1] - joint2[1], joint3[0] - joint2[0]) - ja1 - ja2
        ja3 = self.angle_normalize(ja3)
        ja4 = math.atan2(joint4[1] - joint3[1], joint4[0] - joint3[0]) - ja1 - ja2 - ja3
        ja4 = self.angle_normalize(ja4)

        if debug:
            print(("XY:" if XY else "XZ:")+ str(np.array([ja1, ja2, ja3, ja4])))
        return np.array([ja1, ja2, ja3, ja4])

    # def remove_illumination(self, img):
    #     sum_img = np.sum(img, axis=2)*0.99
    #     sum_img[sum_img==0] = 1e-5
    #     average_img = img / sum_img[:, :, np.newaxis]
    #     return average_img

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
        cv2.waitKey(0)

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
        for template in template_data_xy[:5]:
            img = mask_xy.copy()
            result = cv2.matchTemplate(mask_xy, template, cv2.TM_CCOEFF)
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
        for template in template_data_xz[:5]:
            img = mask_xz.copy()
            result = cv2.matchTemplate(mask_xz, template, cv2.TM_CCOEFF)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)

            if max_val > best_max_val_xz:
                best_max_val_xz = max_val
                best_top_left_xz = top_left
                best_bottom_right_xz = bottom_right


        # img_xy  = image_xy.copy()
        # img_xz = image_xz.copy()
        # cv2.rectangle(img_xy, best_top_left_xy, best_bottom_right_xy, 255, 2)
        # cv2.rectangle(img_xz, best_top_left_xz, best_bottom_right_xz, 255, 2)
        # cv2.imshow("xy", img_xy)
        # cv2.imshow("xz", img_xz)
        # cv2.waitKey(0)

        return np.array([(best_top_left_xy, best_bottom_right_xy), (best_top_left_xz, best_bottom_right_xz)])

    def Jacobian_analytic(self,joint_angles):
        #Forward Kinematics using the analytic equation
        #Each link is 1m long
        #Use trigonometry from FK_analytic and differentiate it with respect to time.
        jacobian = np.zeros((3,3))
        jacobian[0,0] = -np.sin(joint_angles[0])*1-np.sin(joint_angles[0]+joint_angles[1])*1-np.sin(joint_angles[0]+joint_angles[1]+joint_angles[2])*1

        jacobian[0,1] = -np.sin(joint_angles[0]+joint_angles[1])*1-np.sin(joint_angles[0]+joint_angles[1]+joint_angles[2])*1

        jacobian[0,2] = -np.sin(joint_angles[0]+joint_angles[1]+joint_angles[2])*1

        jacobian[1,0] = np.cos(joint_angles[0])*1 + np.cos(joint_angles[0]+joint_angles[1])*1 + np.cos(joint_angles[0]+joint_angles[1]+joint_angles[2])*1

        jacobian[1,1] = np.cos(joint_angles[0]+joint_angles[1])*1 + np.cos(joint_angles[0]+joint_angles[1]+joint_angles[2])*1

        jacobian[1,2] = np.cos(joint_angles[0]+joint_angles[1]+joint_angles[2])*1

        jacobian[2,0] = 1
        jacobian[2,1] = 1
        jacobian[2,2] = 1
        return jacobian

    def go(self):
        #The robot has several simulated modes:
        #These modes are listed in the following format:
        #Identifier (control mode) : Description : Input structure into step function

        #POS : A joint space position control mode that allows you to set the desired joint angles and will position the robot to these angles : env.step((np.zeros(3),np.zeros(3), desired joint angles, np.zeros(3)))
        #POS-IMG : Same control as POS, however you must provide the current joint angles and velocities : env.step((estimated joint angles, estimated joint velocities, desired joint angles, np.zeros(3)))
        #VEL : A joint space velocity control, the inputs require the joint angle error and joint velocities : env.step((joint angle error (velocity), estimated joint velocities, np.zeros(3), np.zeros(3)))
        #TORQUE : Provides direct access to the torque control on the robot : env.step((np.zeros(3),np.zeros(3),np.zeros(3),desired joint torques))
        self.env.controlMode="POS"
        #Run 100000 iterations
        previous_joint_angles = np.zeros(4)
        prev_jvs = collections.deque(np.zeros(4),1)

        #self.env.render returns 2 RGB arrays of the robot, one for the xy-plane, and one for the xz-plane
        image_foreground_xy, image_foreground_xz = self.env.render('rgb-array')

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


        # for i in range(100):
        #     #The change in time between iterations can be found in the self.env.dt variable
        #     dt = self.env.dt
        #
        #     print "Real JA" + str(self.env.ground_truth_joint_angles)
        #
        #     arrxy, arrxz = self.env.render('rgb-array')
        #     detected_joint_angles = self.detect_joint_angles(arrxy, thresholds)
        #     self.env.step((np.zeros(4), np.zeros(4), detected_joint_angles, np.zeros(4)))
        #     last_img = arrxy
        #     if i ==5 :
        #         break

        self.show_joints_with_details(*self.env.render('rgb-array'), thresholds= thresholds)
#main method
def main():
    reach = MainReacher()
    reach.go()

if __name__ == "__main__":
    main()
