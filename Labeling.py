# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 13:46:31 2022
# to get the position of a color in 2D
to assign label to each color
@author: pza0029@Auburn.edu
Parisa Asadi
"""

######################################################################################
######################## read libraries ##############################################\
#####set directory to the place that I have saved my simple u_net code
import os
import glob
import cv2
import numpy as np
import tifffile as tiff
import natsort


######################################################################################
######################## get images ##################################################


def ColorToNumber(main_dir, folder_with_image):
    
    os.chdir(main_dir) #main folde0
    data_path_image = os.path.join(folder_with_image, '*.tif') 
    image = glob.glob(data_path_image)
    image = natsort.natsorted(image,reverse=False)
    try:
        os.makedirs(os.path.join(folder_with_image, 'ColorToNum'))
    except:
            print('the folder is already created')
    
    for f1 in range(len(image)): 
        img = cv2.imread(image[f1])
        name= (image[f1].split('\\')[-1]).split(".")[0]
        data = np.zeros((img.shape[0], img.shape[1]))
             
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      
        # define blue color range
        light_blue1 = np.array([0,0,0])#===============> change it
        light_blue2 = np.array([110,110,110])#===============> change it
        light_blue3 = np.array([255,0,255])#===============> change it
        light_blue4 = np.array([255,255,150])#===============> change it
        light_blue5 = np.array([255,200,150])#===============> change it
        light_blue6 = np.array([150,0,200])#===============> change it
        light_blue7 = np.array([255,165,0])#===============> change it
        light_blue8 = np.array([150,255,150])#===============> change it
        light_blue9 = np.array([0,150,255])#===============> change it
        light_blue10 = np.array([255,0,0])#===============> change it
        light_blue11= np.array([200,100,0])#===============> change it
        light_blue12 = np.array([0,100,0])#===============> change it
        light_blue13= np.array([0,255,0])#===============> change it
        light_blue14 = np.array([150,0,0])#===============> change it
        light_blue15 = np.array([0,255,255])#===============> change it
      
        
      
        mask1 = cv2.inRange(img, light_blue1, light_blue1)
        mask2 = cv2.inRange(img, light_blue2, light_blue2)
        mask3 = cv2.inRange(img, light_blue3, light_blue3)
        mask4 = cv2.inRange(img, light_blue4, light_blue4)
        mask5 = cv2.inRange(img, light_blue5, light_blue5)
        mask6 = cv2.inRange(img, light_blue6,light_blue6)
        mask7 = cv2.inRange(img, light_blue7, light_blue7)
        mask8 = cv2.inRange(img, light_blue8, light_blue8)
        mask9 = cv2.inRange(img, light_blue9, light_blue9)
        mask10 = cv2.inRange(img, light_blue10, light_blue10)
        mask11 = cv2.inRange(img, light_blue11,light_blue11)
        mask12 = cv2.inRange(img, light_blue12, light_blue12)
        mask13 = cv2.inRange(img, light_blue13,light_blue13)
        mask14 = cv2.inRange(img, light_blue14, light_blue14)
        mask15 = cv2.inRange(img, light_blue15, light_blue15)
        

        data[mask1==255]=0
        data[mask2==255]=1
        data[mask3==255]=2
        data[mask4==255]=3
        data[mask5==255]=4
        data[mask6==255]=5
        data[mask7==255]=6
        data[mask8==255]=7
        data[mask9==255]=8
        data[mask10==255]=9
        data[mask11==255]=10
        data[mask12==255]=11
        data[mask13==255]=12
        data[mask14==255]=13
        data[mask15==255]=14

        data=data.astype(np.uint8)
        cv2.imwrite(f"{folder_with_image}/ColorToNum/{name}_ColorToNum_labeled.tif",data)
                
#####Get data for each CT series #########
if __name__ =="__main__":

    main_dir = r'E:\Parisa\samples\Labels'

    folder_with_image = r'E:\Parisa\samples\Labels'
    ColorToNumber(main_dir, folder_with_image)