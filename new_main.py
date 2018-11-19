#!/usr/bin/env python2.7
import gym
import reacher3D.Reacher
import numpy as np
import numpy.linalg as npl
import cv2
import math
import collections
import time
import os
import glob


class MainReacher():
    def __init__(self):
        self.env = gym.make('3DReacherMy-v0')
        self.env.reset()

    def detect_l1(self,imageXY,imageXZ):
	#Detecting the first link
        masky = cv2.inRange(imageXY, (101,101,101),(104,104,104))
        kernel = np.ones((5,5),np.uint8)
        masky = cv2.dilate(masky,kernel,iterations=3)
        My = cv2.moments(masky)
        cx = int(My['m10']/My['m00'])
        cy = int(My['m01']/My['m00'])

        maskz = cv2.inRange(imageXZ, (101,101,101),(104,104,104))
        maskz = cv2.dilate(maskz,kernel,iterations=3)
        Mz = cv2.moments(maskz)
        cx = int(Mz['m10']/Mz['m00'])
        cz = int(Mz['m01']/Mz['m00'])

        yCor = self.coordinate_convert(np.array([cx,cy]))
        zCor = self.coordinate_convert(np.array([cx,cz]))
        x = yCor[0]
        y = yCor[1]
        z = zCor[1]

        retArray = np.array([np.asscalar(x),np.asscalar(y),np.asscalar(z)])
        return retArray

    def detect_l2(self,imageXY,imageXZ):
	#Detecting the second link
        masky = cv2.inRange(imageXY, (50,50,50),(52,52,52))
        kernel = np.ones((5,5),np.uint8)
        masky = cv2.dilate(masky,kernel,iterations=3)
        My = cv2.moments(masky)
        cx = int(My['m10']/My['m00'])
        cy = int(My['m01']/My['m00'])

        maskz = cv2.inRange(imageXZ, (50,50,50),(52,52,52))
        maskz = cv2.dilate(maskz,kernel,iterations=3)
        Mz = cv2.moments(maskz)
        cx = int(Mz['m10']/Mz['m00'])
        cz = int(Mz['m01']/Mz['m00'])

        yCor = self.coordinate_convert(np.array([cx,cy]))
        zCor = self.coordinate_convert(np.array([cx,cz]))
        x = yCor[0]
        y = yCor[1]
        z = zCor[1]

        retArray = np.array([np.asscalar(x),np.asscalar(y),np.asscalar(z)])
        return retArray

    def detect_l3(self,imageXY,imageXZ):
	#Detecting the third link
        masky = cv2.inRange(imageXY, (0,0,0),(2,2,2))
        kernel = np.ones((5,5),np.uint8)
        masky = cv2.dilate(masky,kernel,iterations=3)
        My = cv2.moments(masky)
        cx = int(My['m10']/My['m00'])
        cy = int(My['m01']/My['m00'])

        maskz = cv2.inRange(imageXZ, (0,0,0),(2,2,2))
        maskz = cv2.dilate(maskz,kernel,iterations=3)
        Mz = cv2.moments(maskz)
        cx = int(Mz['m10']/Mz['m00'])
        cz = int(Mz['m01']/Mz['m00'])

        yCor = self.coordinate_convert(np.array([cx,cy]))
        zCor = self.coordinate_convert(np.array([cx,cz]))
        x = yCor[0]
        y = yCor[1]
        z = zCor[1]

        retArray = np.array([np.asscalar(x),np.asscalar(y),np.asscalar(z)])
        return retArray

    def detect_l4(self,imageXY,imageXZ):
	#Detecting the fourth link
        masky = cv2.inRange(imageXY, (125,125,125),(128,128,128))
        kernel = np.ones((5,5),np.uint8)
        masky = cv2.dilate(masky,kernel,iterations=3)
        My = cv2.moments(masky)
        cx = int(My['m10']/My['m00'])
        cy = int(My['m01']/My['m00'])

        maskz = cv2.inRange(imageXZ, (125,125,125),(128,128,128))
        maskz = cv2.dilate(maskz,kernel,iterations=3)
        Mz = cv2.moments(maskz)
        cx = int(Mz['m10']/Mz['m00'])
        cz = int(Mz['m01']/Mz['m00'])

        yCor = self.coordinate_convert(np.array([cx,cy]))
        zCor = self.coordinate_convert(np.array([cx,cz]))
        x = yCor[0]
        y = yCor[1]
        z = zCor[1]

        retArray = np.array([np.asscalar(x),np.asscalar(y),np.asscalar(z)])
        return retArray

    def get_l_locs(self,imageXY,imageXZ):
        l1loc = self.detect_l1(imageXY,imageXZ)
        l2loc = self.detect_l2(imageXY,imageXZ)
        l3loc = self.detect_l3(imageXY,imageXZ)
        l4loc = self.detect_l4(imageXY,imageXZ)

        print "Link 1: " + str(l1loc)
        print "Link 2: " + str(l2loc)
        print "Link 3: " + str(l3loc)
        print "Link 4: " + str(l4loc)


    def detect_blue(self,imageXY,imageXZ):
        masky = cv2.inRange(imageXY, (0,0,245),(0,0,255))
        kernel = np.ones((5,5),np.uint8)
        masky = cv2.dilate(masky,kernel,iterations=3)
        My = cv2.moments(masky)
        cx = int(My['m10']/ (My['m00'] if My['m00'] != 0 else 1e-6))
        cy = int(My['m01']/ (My['m00'] if My['m00'] != 0 else 1e-6))

        maskz = cv2.inRange(imageXZ, (0,0,245),(0,0,255))
        maskz = cv2.dilate(maskz,kernel,iterations=3)
        Mz = cv2.moments(maskz)
        cx = int(Mz['m10']/ (Mz['m00'] if Mz['m00'] != 0 else 1e-6))
        cz = int(Mz['m01']/ (Mz['m00'] if Mz['m00'] != 0 else 1e-6))

        yCor = self.coordinate_convert(np.array([cx,cy]))
        zCor = self.coordinate_convert(np.array([cx,cz]))
        x = yCor[0]
        y = yCor[1]
        z = zCor[1]

        retArray = np.array([np.asscalar(x),np.asscalar(y),np.asscalar(z)])
        return retArray

    def detect_green(self,imageXY, imageXZ):
        masky = cv2.inRange(imageXY, (0,245,0),(0,255,0))
        kernel = np.ones((5,5),np.uint8)
        masky = cv2.dilate(masky,kernel,iterations=3)
        My = cv2.moments(masky)
        cx = int(My['m10']/ (My['m00'] if My['m00'] != 0 else 1e-6))
        cy = int(My['m01']/ (My['m00'] if My['m00'] != 0 else 1e-6))

        maskz = cv2.inRange(imageXZ, (0,245,0),(0,255,0))
        maskz = cv2.dilate(maskz,kernel,iterations=3)
        Mz = cv2.moments(maskz)
        cx = int(Mz['m10']/ (Mz['m00'] if Mz['m00'] != 0 else 1e-6))
        cz = int(Mz['m01']/ (Mz['m00'] if Mz['m00'] != 0 else 1e-6))

        yCor = self.coordinate_convert(np.array([cx,cy]))
        zCor = self.coordinate_convert(np.array([cx,cz]))
        x = yCor[0]
        y = yCor[1]
        z = zCor[1]

        retArray = np.array([np.asscalar(x),np.asscalar(y),np.asscalar(z)])
        return retArray

    def detect_red(self,imageXY,imageXZ):
        masky = cv2.inRange(imageXY, (245,0,0),(255,0,0))
        kernel = np.ones((5,5),np.uint8)
        masky = cv2.dilate(masky,kernel,iterations=3)
        My = cv2.moments(masky)
        cx = int(My['m10']/ (My['m00'] if My['m00'] != 0 else 1e-6))
        cy = int(My['m01']/ (My['m00'] if My['m00'] != 0 else 1e-6))

        maskz = cv2.inRange(imageXZ, (245,0,0),(255,0,0))
        maskz = cv2.dilate(maskz,kernel,iterations=3)
        Mz = cv2.moments(maskz)
        cx = int(Mz['m10']/ (Mz['m00'] if Mz['m00'] != 0 else 1e-6))
        cz = int(Mz['m01']/ (Mz['m00'] if Mz['m00'] != 0 else 1e-6))

        yCor = self.coordinate_convert(np.array([cx,cy]))
        zCor = self.coordinate_convert(np.array([cx,cz]))
        x = yCor[0]
        y = yCor[1]
        z = zCor[1]

        retArray = np.array([np.asscalar(x),np.asscalar(y),np.asscalar(z)])
        return retArray

    def detect_Dblue(self,imageXY,imageXZ):
        masky = cv2.inRange(imageXY, (0,0,125),(0,0,130))
        kernel = np.ones((5,5),np.uint8)
        masky = cv2.dilate(masky,kernel,iterations=3)
        My = cv2.moments(masky)
        cx = int(My['m10']/ (My['m00'] if My['m00'] != 0 else 1e-6))
        cy = int(My['m01']/ (My['m00'] if My['m00'] != 0 else 1e-6))

        maskz = cv2.inRange(imageXZ, (0,0,125),(0,0,130))
        maskz = cv2.dilate(maskz,kernel,iterations=3)
        Mz = cv2.moments(maskz)
        cx = int(Mz['m10']/ (Mz['m00'] if Mz['m00'] != 0 else 1e-6))
        cz = int(Mz['m01']/ (Mz['m00'] if Mz['m00'] != 0 else 1e-6))

        yCor = self.coordinate_convert(np.array([cx,cy]))
        zCor = self.coordinate_convert(np.array([cx,cz]))
        x = yCor[0]
        y = yCor[1]
        z = zCor[1]

        retArray = np.array([np.asscalar(x),np.asscalar(y),np.asscalar(z)])
        return retArray

    def detect_target(self,imageXY,imageXZ):

        template_dataxy= []
        template_dataxz = []
        #make a list of all template images from a directory
        xyfiles= glob.glob('valid/validxy*.jpg')
        xzfiles = glob.glob('valid/validxz*.jpg')

        for files in xyfiles:
            image = cv2.imread(files,0)
            template_dataxy.append(image)
        for files in xzfiles:
            image = cv2.imread(files,0)
            template_dataxz.append(image)

        maskxy = cv2.inRange(imageXY, (175,175,175),(180,180,180))
        maskxz = cv2.inRange(imageXZ, (175,175,175),(180,180,180))

        largestVal = 0
        largestLeft = 0
        largestRight = 0
        for tmp in template_dataxy:
            result = cv2.matchTemplate(maskxy, tmp, cv2.TM_CCOEFF)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            top_left = max_loc
            bottom_right = (top_left[0] + 40, top_left[1] + 40)
            if(max_val>largestVal):
                largestLeft = top_left
                largestRight = bottom_right
                largestVal = max_val

        cx,cy = largestLeft
        yCor = self.coordinate_convert(np.array([cx+20,cy+20]))
        largestVal = 0
        largestLeftZ = 0
        largestRightZ = 0
        for tmp in template_dataxz:
            result = cv2.matchTemplate(maskxz, tmp, cv2.TM_CCOEFF)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            top_left = max_loc
            bottom_right = (top_left[0] + 40, top_left[1] + 40)
            if(max_val>largestVal):
                largestLeftZ = top_left
                largestRightZ = bottom_right
                largestVal = max_val
        cx,cz = largestLeftZ
        zCor = self.coordinate_convert(np.array([cx+20,cz+20]))
        fullCor = np.array([np.asscalar(yCor[0]),np.asscalar(yCor[1]),np.asscalar(zCor[1])])

        #print "realtarget " + str(self.env.ground_truth_valid_target)
        #print "target " + str(fullCor)
        #cv2.rectangle(maskxy,largestLeft, largestRight,255, 2)
        #cv2.rectangle(maskxz,largestLeftZ, largestRightZ,255, 2)
        #cv2.imshow("maskxy",maskxy)
        #cv2.imshow("maskxz",maskxz)
        #cv2.waitKey(0)

        return fullCor

    def detect_joint_angles(self,imageXY,imageXZ):
        jointPos4 = self.detect_Dblue(imageXY,imageXZ)
        jointPos3 = self.detect_blue(imageXY,imageXZ)
        jointPos2 = self.detect_green(imageXY,imageXZ)
        jointPos1 = self.detect_red(imageXY,imageXZ)

        #get angle one
        ja1 = np.arccos(jointPos1[0]/npl.norm(jointPos1))

        if(jointPos1[0]<0 and jointPos1[1]>0):
            ja1 = np.pi - ja1
        elif(jointPos1[0]<0 and jointPos1[1]<0):
            ja1 = -np.pi + ja1
        elif(jointPos1[0]>0 and jointPos1[1]>0):
            ja1 = ja1
        elif(jointPos1[0]>0 and jointPos1[1]<0):
            ja1 = - ja1

        #create rotation matrix
        rotationMatrixY = np.array([[np.cos(ja1),0,np.sin(ja1)],
                                    [0         ,1,          0],
                                    [-np.sin(ja1),0,np.cos(ja1)]])

        #create vector
        j2Vec = np.transpose(jointPos2 - jointPos1)

        #apply rotation matrix and translation
        j2Vec = np.dot(rotationMatrixY,j2Vec)
        j2Vec[1] = j2Vec[1] - np.dot(rotationMatrixY,np.transpose(jointPos1))[1]

        #get angle 2
        ja2 = np.arccos(j2Vec[0]/npl.norm(j2Vec))

        #repeat for angle 3
        rotationMatrixY2 = np.array([[np.cos(ja2),0,np.sin(ja2)],
                                    [0         ,1,          0],
                                    [-np.sin(ja2),0,np.cos(ja2)]])

        rotationMatrixZ = np.array([[np.cos(ja2),-np.sin(ja2),0],
                                    [np.sin(ja2),np.cos(ja2), 0],
                                    [0          ,0,           1]])


        j3Vec = np.transpose(jointPos3-jointPos2)

        j3Vec = np.dot(rotationMatrixY,j3Vec)
        j3Vec[1] = j3Vec[1] - np.dot(rotationMatrixY,np.transpose(jointPos1))[1]


        j3Vec = np.dot(rotationMatrixZ,j3Vec)
        j3Vec[1] = j3Vec[1] - np.dot(rotationMatrixZ,np.transpose(jointPos2-jointPos1))[1]

        ja3 = np.arccos(j3Vec[0]/npl.norm(j3Vec))

        #repeat for angle 4
        rotationMatrixY2 = np.array([[np.cos(ja3),0,np.sin(ja3)],
                                    [0         ,1,          0],
                                    [-np.sin(ja3),0,np.cos(ja3)]])

        rotationMatrixZ2 = np.array([[np.cos(ja3),-np.sin(ja3),0],
                                    [np.sin(ja3),np.cos(ja3), 0],
                                    [0          ,0,           1]])


        j4Vec = np.transpose(jointPos4-jointPos3)

        j3Vec = np.dot(rotationMatrixY,j4Vec)
        j4Vec[1] = j4Vec[1] - np.dot(rotationMatrixY,np.transpose(jointPos1))[1]


        j4Vec = np.dot(rotationMatrixZ,j4Vec)
        j4Vec[1] = j4Vec[1] - np.dot(rotationMatrixZ,np.transpose(jointPos2-jointPos1))[1]

        j4Vec = np.dot(rotationMatrixZ2,j4Vec)
        j4Vec[1] = j4Vec[1] - np.dot(rotationMatrixZ,np.transpose(jointPos3-jointPos2))[1]

        ja4 = np.arccos(j3Vec[0]/npl.norm(j3Vec))

        #normalise joint angle 2
        if(j2Vec[0]<0 and j2Vec[1]>0):
            #UL
            #print "UL"
            ja2 = ja2
        elif(j2Vec[0]<0 and j2Vec[1]<0):
            #LL
            #print "LL"
            ja2 = -ja2
        elif(j2Vec[0]>0 and j2Vec[1]>0):
            #UR
            ja2 = ja2
        elif(j2Vec[0]>0 and j2Vec[1]<0):
            #LR
            ja2 = -ja2

        #normalise joint angle 3
        if(j3Vec[0]<0 and j2Vec[1]>0):
            #UL
            ja3 = ja3
        elif(j3Vec[0]<0 and j3Vec[1]<0):
            #LL
            ja3 = ja3
        elif(j3Vec[0]>0 and j3Vec[1]>0):
            #UR
            ja3 = ja3
        elif(j3Vec[0]>0 and j3Vec[1]<0):
            #LR
            ja3 = ja3

        #normalise joint angle 4
        if(j4Vec[0]<0 and j4Vec[1]>0):
            #UL
            ja4 = ja4
        elif(j4Vec[0]<0 and j4Vec[1]<0):
            #LL
            ja4 = -ja4
        elif(j4Vec[0]>0 and j4Vec[1]>0):
            #UR
            ja4 = -ja4
        elif(j4Vec[0]>0 and j4Vec[1]<0):
            #LR
            ja4 = -ja4

        return np.array([ja1,ja2,ja3,ja4])

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

        #print("Error: {}".format(np.squeeze(np.array(ee_pos)) - np.array(self.env.ground_truth_end_effector)))

        
        #print(j2_pos, j3_pos, j4_pos, ee_pos)

        jacobian = np.zeros((3,4))
        pos_3D = np.zeros(3)

        y_vector = np.array([0,-1,0])
        z_vector = np.array([0,0,1])        
        pos_3D = (ee_pos-j1_pos).T
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

    def grav(self,joint_angles):
        
        t1 = self.link_transform_Y(joint_angles[0])
        t2 = self.link_transform_Z(joint_angles[1])
        t3 = self.link_transform_Z(joint_angles[2])
        t4 = self.link_transform_Y(joint_angles[3])
        
        jointPos1 = (t1)[0:3, 3]
        jointPos2 = (t1*t2)[0:3, 3]
        jointPos3 = (t1*t2*t3)[0:3, 3]
        jointPos4 = (t1*t2*t3*t4)[0:3, 3]
        
        #print(jointPos3)
        
	#The four centre of masses of the links
        com1 = np.squeeze(np.array(jointPos1/2))
        com2 = np.squeeze(np.array((jointPos2 - jointPos1)/2))
        com3 = np.squeeze(np.array((jointPos3 - jointPos2)/2))
        com4 = np.squeeze(np.array((jointPos4 - jointPos3)/2))
        g = np.array([0, 9.81, 0])
        
        y_vector = np.array([0,-1,0])
        z_vector = np.array([0,0,1])
        
        f1 = np.dot(np.cross(com1.T, g), y_vector) + np.dot(np.cross(com2.T, g), y_vector) + np.dot(np.cross(com3.T, g), y_vector) + np.dot(np.cross(com4.T, g), y_vector)
        
        y_vector = np.squeeze(np.array(self.rotation_matrix_Y(joint_angles[0])* np.matrix(y_vector).T))
        z_vector = np.squeeze(np.array(self.rotation_matrix_Y(joint_angles[0])* np.matrix(z_vector).T))
        
        f2 = np.dot(np.cross(com2.T, g), z_vector) + np.dot(np.cross(com3.T, g), z_vector) +np.dot(np.cross(com4.T, g), z_vector)
        
        y_vector = np.matrix([0,-1,0])
        z_vector = np.matrix([0,0,1])
        
        y_vector = np.squeeze(np.array((self.rotation_matrix_Y(joint_angles[0]) * self.rotation_matrix_Z(joint_angles[1]))* np.matrix(y_vector).T))
        z_vector = np.squeeze(np.array((self.rotation_matrix_Y(joint_angles[0]) * self.rotation_matrix_Z(joint_angles[1]))* np.matrix(z_vector).T))

        f3 = np.dot(np.cross(com3.T, g), z_vector) + np.dot(np.cross(com4.T, g), z_vector)
        
        y_vector = np.matrix([0,-1,0])
        z_vector = np.matrix([0,0,1])
        
        y_vector = np.squeeze(np.array((self.rotation_matrix_Y(joint_angles[0]) * self.rotation_matrix_Z(joint_angles[1]) * self.rotation_matrix_Z(joint_angles[2]))* np.matrix(y_vector).T))
        z_vector = np.squeeze(np.array((self.rotation_matrix_Y(joint_angles[0]) * self.rotation_matrix_Z(joint_angles[1]) * self.rotation_matrix_Z(joint_angles[2]))* np.matrix(z_vector).T))
        
        f4 = np.dot(np.cross(com4, g.T), y_vector.T)
        
        #print (f1, f2, f3, f4)
        
        return np.matrix([f1, f2, f3, f4]).T
    
    def ts_pd_grav_control(self, current_joint_angles, current_joint_velocities, desired_position):

        #Calculate the torque required to reach the desired position using a PD controller with gravity compensation (TASK SPACE)
        #Assume desired velocities are zero
        P = np.array([1000,1000,1000])
        D = np.array([50,50,50])
        J = self.Jacobian(current_joint_angles)[0:3,:]

        #Obtain end effector velocity from equation x_dot = Jacobian * q_dot
        xd = J*np.matrix(current_joint_velocities).T
        curr_pos = self.FK(current_joint_angles)[0:3,3]
        grav_torques = self.grav(current_joint_angles)

        #Add the line for grav torques
        return J.T*(np.diag(P)*(np.matrix(desired_position).T-curr_pos)-np.diag(D)*xd) + grav_torques


    def angle_normalize(self,x):
        #Normalizes the angle between pi and -pi
        return (((x+np.pi) % (2*np.pi)) - np.pi)

    def coordinate_convert(self,pixels):
        #Converts pixels into metres
        return np.array([(pixels[0]-self.env.viewerSize/2)/self.env.resolution,-(pixels[1]-self.env.viewerSize/2)/self.env.resolution])

    def go(self):
        #The robot has several simulated modes:
        #These modes are listed in the following format:
        #Identifier (control mode) : Description : Input structure into step function

        #POS : A joint space position control mode that allows you to set the desired joint angles and will position the robot to these angles : env.step((np.zeros(3),np.zeros(3), desired joint angles, np.zeros(3)))
        #POS-IMG : Same control as POS, however you must provide the current joint angles and velocities : env.step((estimated joint angles, estimated joint velocities, desired joint angles, np.zeros(3)))
        #VEL : A joint space velocity control, the inputs require the joint angle error and joint velocities : env.step((joint angle error (velocity), estimated joint velocities, np.zeros(3), np.zeros(3)))
        #TORQUE : Provides direct access to the torque control on the robot : env.step((np.zeros(3),np.zeros(3),np.zeros(3),desired joint torques))
        self.env.controlMode="TORQUE"
        prev_JAs = np.zeros(4)
        prev_jvs = np.zeros(4)
        self.env.enable_gravity(True)
        self.env.D_gains[0] = 80
        #Run 100000 iterations
        for _ in range(1000000):
            #The change in time between iterations can be found in the self.env.dt variable
            dt = self.env.dt
            #self.env.render returns two rgb images facing the xy and xz planes, they are orthogonal
            (xy,xz) = self.env.render('rgb-array')


	        #self.get_l_locs(xy, xz) #detects the links used 
            detectedJointAngles = self.detect_joint_angles(xy,xz)
            ee_target = self.detect_target(xy, xz)
            #print "Real JA" + str(self.env.ground_truth_joint_angles)
            #print "JA     " + str(detectedJointAngles)
            #print ""
            

            prev_jvs=(self.angle_normalize(self.env.ground_truth_joint_angles-prev_JAs))
            detectedJointVels = (prev_jvs/dt)
            prev_JAs = self.env.ground_truth_joint_angles
            jointAngles = np.array([0.5,-0.5,0.5,-0.5])
            
            jointAnglesIK = self.IK(self.env.ground_truth_joint_angles, ee_target)
            trqs = self.ts_pd_grav_control(self.env.ground_truth_joint_angles, detectedJointVels, ee_target)

            self.env.step((np.zeros(3),np.zeros(3),np.zeros(3),trqs)) #For the gravity compensated torque control
            #self.env.step((jointAnglesIK, detectedJointVels, np.zeros(3), np.zeros(3)))  #For the velocity control
	        #self.env.step((detectedJointAngles, detectedJointVels, jointAngles, np.zeros(3))) #For the target detection
            #The step method will send the control input to the robot, the parameters are as follows: (Current Joint Angles/Error, Current Joint Velocities, Desired Joint Angles, Torque input)

#main method
def main():
    reach = MainReacher()
    reach.go()

if __name__ == "__main__":
    main()