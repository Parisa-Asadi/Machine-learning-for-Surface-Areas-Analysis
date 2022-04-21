
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 18:53:55 2021

@author: pza0029@auburn.edu
Parisa Asadi



___________________ HOW to prepare the data to crop
first create a main folder
next put any folder that you want to crop the images in it as folder with image
run the code it will crop them and create a new folder in the "folder with images" with title of "cropped"
enjoy
 for 1 channel crop one, for 3 channel use the label one.

@author: Parisa Asadi
Croping the images
"""

############## croping the images ################
############ crop ###################

import cv2
import numpy as np 
import glob
import os 
import tifffile as tiff

#function
def Crop_Label_images(main_dir, folder_with_image, cut_stride = 128, real_remain_image = True):
    '''

    @Parisa Asadi
    @Date: 11/12/2021
    Parameters
    ----------
    main_dir : TYPE
        main directory that contains several folders.
    folder_with_image : TYPE
        each folder inside the directory.
    cut_stride : TYPE, optional
        the output size of image. The default is 512.

    Returns None
    -------
    None.

    '''
    
    import os
    os.chdir(main_dir) #main folde0
    data_path_image = os.path.join(folder_with_image, '*.tif') 
    image = glob.glob(data_path_image) 
    try:
        os.makedirs(os.path.join(folder_with_image, 'cropped'))
    except:
            print('the folder is already created')
    try:
        os.makedirs(os.path.join(folder_with_image, 'cropped', 'edges'))
    except:
            print('the folder is already created')            
        
    for f1 in range(len(image)): 

        img = cv2.imread(image[f1])
        name= (image[f1].split('\\')[-1]).split(".")[0]
    
        if real_remain_image == False:
            for r in range(0,img.shape[0], cut_stride):
                for c in range(0,img.shape[1], cut_stride):
               
                    if c+cut_stride > img.shape[1]:
                        cv2.imwrite(f"{folder_with_image}/cropped/edges/{name}_{r}_{c}_labeled.tif",img[r:r+cut_stride, img.shape[1]-cut_stride:img.shape[1],:])
                    elif r+cut_stride > img.shape[0]:
                        cv2.imwrite(f"{folder_with_image}/cropped/edges/{name}_{r}_{c}_labeled.tif",img[img.shape[0]-cut_stride:img.shape[0], c:c+cut_stride,:])
                    else:
                        cv2.imwrite(f"{folder_with_image}/cropped/{name}_{r}_{c}_labeled.tif",img[r:r+cut_stride, c:c+cut_stride,:])
        else:
            for r in range(0,img.shape[0], cut_stride):
                for c in range(0,img.shape[1], cut_stride):
                    if c+cut_stride > img.shape[1]:
                        cv2.imwrite(f"{folder_with_image}/cropped/edges/{name}_{r}_{c}_labeled.tif",img[r:r+cut_stride, c:img.shape[1],:])
                    elif r+cut_stride > img.shape[0]:
                        cv2.imwrite(f"{folder_with_image}/cropped/edges/{name}_{r}_{c}_labeled.tif",img[r:img.shape[0], c:c+cut_stride,:])
                    else:
                        cv2.imwrite(f"{folder_with_image}/cropped/{name}_{r}_{c}_labeled.tif",img[r:r+cut_stride, c:c+cut_stride,:])

#####Get data for each CT series #########
if __name__ =="__main__":

    main_dir = r"E:\Parisa\samples\Test_riskassessment&surfacearea\ranking_riskMaps\Labels\New folder"


    for i, folder_with_image in enumerate(glob.glob(r'E:\Parisa\samples\Test_riskassessment&surfacearea\ranking_riskMaps\Labels\New folder\*')):
        Crop_Label_images(main_dir, folder_with_image, cut_stride = 128, real_remain_image=True)
    
    
    
    
    
##############################################images as input to ML
# ___________________ HOW to prepare the data to crop
# first create a main folder
# next put any folder that you want to crop the images in it as folder with image
# run the code it will crop them and create a new folder in the "folder with images" with title of "cropped"
#enjoy
#this is for 1 channel for 3 channel use the label one.

#function
def Crop_images(main_dir, folder_with_image, cut_stride = 128, real_remain_image = True):
    '''

    @Parisa Asadi
    @Date: 11/12/2021
    Parameters
    ----------
    main_dir : TYPE
        main directory that contains several folders.
    folder_with_image : TYPE
        each folder inside the directory.
    cut_stride : TYPE, optional
        the output size of image. The default is 512.

    Returns None
    -------
    None.

    '''
    os.chdir(main_dir) #main folde0
    data_path_image = os.path.join(folder_with_image, '*.tif') 
    image = glob.glob(data_path_image) 
    try:
        os.makedirs(os.path.join(folder_with_image, 'cropped'))
    except:
            print('the folder is already created')
    try:
        os.makedirs(os.path.join(folder_with_image, 'cropped', 'edges'))
    except:
            print('the folder is already created')            
        
    for f1 in range(len(image)): 
        
        try:
            img = tiff.imread(image[f1])
            
            if np.max(img)==0:
                img=img
                print(f'{image}') 
            else:
                print(np.max(img)) 
                img = img/(np.max(img))
                
        except:
            img = cv2.imread(image[f1],0)
            
            if np.max(img)==0:
                img=img
                print(f'{image}') 
            else:
                print(np.max(img))
                img = img/(np.max(img))
            print("1")
        name= (image[f1].split('\\')[-1]).split(".")[0]
    
        if real_remain_image == False:
            for r in range(0,img.shape[0], cut_stride):
                for c in range(0,img.shape[1], cut_stride):
               
                    if c+cut_stride > img.shape[1]:
                        tiff.imsave(f"{folder_with_image}/cropped/edges/{name}_{r}_{c}_labeled.tif",img[r:r+cut_stride, img.shape[1]-cut_stride:img.shape[1]])
                    elif r+cut_stride > img.shape[0]:
                        tiff.imsave(f"{folder_with_image}/cropped/edges/{name}_{r}_{c}_labeled.tif",img[img.shape[0]-cut_stride:img.shape[0], c:c+cut_stride])
                    else:
                        tiff.imsave(f"{folder_with_image}/cropped/{name}_{r}_{c}_labeled.tif",img[r:r+cut_stride, c:c+cut_stride])
        else:
            for r in range(0,img.shape[0], cut_stride):
                for c in range(0,img.shape[1], cut_stride):
                    if c+cut_stride > img.shape[1]:
                        tiff.imsave(f"{folder_with_image}/cropped/edges/{name}_{r}_{c}_labeled.tif",img[r:r+cut_stride, c:img.shape[1]])
                    elif r+cut_stride > img.shape[0]:
                        tiff.imsave(f"{folder_with_image}/cropped/edges/{name}_{r}_{c}_labeled.tif",img[r:img.shape[0], c:c+cut_stride])
                    else:
                        tiff.imsave(f"{folder_with_image}/cropped/{name}_{r}_{c}_labeled.tif",img[r:r+cut_stride, c:c+cut_stride])

#####Get data for each CT series #########
if __name__ =="__main__":

    main_dir = r'E:\Parisa\samples\sample'


    for i, folder_with_image in enumerate(glob.glob(r'E:\Parisa\samples\sample\*')):
        Crop_images(main_dir, folder_with_image, cut_stride = 128, real_remain_image=True)


