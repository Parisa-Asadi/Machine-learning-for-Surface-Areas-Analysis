# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 13:15:57 2022

@author: pza0029
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 13:42:21 2021

@author: pza0029
"""
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 18:42:35 2021

@author: pza0029
"""

import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import ndimage as nd
from skimage.filters import sobel, sobel_h, sobel_v
from skimage.filters import difference_of_gaussians
from skimage.filters import hessian
from skimage.filters import gabor, laplace


#function
def filter_images(main_dir, folder_with_image):
    os.chdir(main_dir) #main folde0
    data_path_image = os.path.join(folder_with_image, '*.tif') 
    image = glob.glob(data_path_image)  
    
    jj = 0
    for f1 in range(len(image)): 
        img = cv2.imread(image[f1],0)
        name= (image[f1].split('\\')[-1]).split(".")[0]

        
        ############################################
        #####--------get filters----------------####
        ############################################
        gaussian_img = nd.gaussian_filter(img, sigma=1).astype(np.float64)
        sobel_h1 = sobel_h(img)
        sobel_v1 = sobel_v(img)
        differenceOfGaussians_1_10 = difference_of_gaussians(img, 1,10)
        differenceOfGaussians_1_5 = difference_of_gaussians(img, 1,5)
        differenceOfGaussians_0_5 = difference_of_gaussians(img, 0,5)
        differenceOfGaussians_1_2 = difference_of_gaussians(img, 1,2)
        hessian1= hessian(img,sigmas=1)
        hessian3= hessian(img,sigmas=5)
        differenceOfHessians = hessian3 - hessian1
        gaborfilt_realx, gaborfilt_imagx = gabor(img, theta = 0, frequency=0.7,mode='nearest')
        gaborfilt_imagx = gaborfilt_imagx.astype(np.float64)
        gaborfilt_realy, gaborfilt_imagy = gabor(img, theta = 90, frequency=0.7,mode='nearest')
        gaborfilt_imagy=gaborfilt_imagy.astype(np.float64)
        median_blur = cv2.medianBlur(img,5).astype(np.float64)
        laplace_filter = laplace(img, ksize=100, mask=None)
        nlMeans1=cv2.fastNlMeansDenoising(img).astype(np.float64)
        bilateral = cv2.bilateralFilter(img,2,255,5)
        ############################################
        #####--------write_filters--------------####
        ############################################
        
       #nameOfFilters = [bilateral,nlMeans1,gaussian_img,sobel_h1,sobel_v1,differenceOfGaussians_1_10,differenceOfGaussians_0_5,differenceOfGaussians_1_5,differenceOfGaussians_1_2,differenceOfHessians,gaborfilt_imagx,gaborfilt_imagy,median_blur,laplace_filter]
        nameOfFilters_string = ["bilateral","nlMeans1","gaussian_img","sobel_h1","sobel_v1","differenceOfGaussians_1_10","differenceOfGaussians_0_5","differenceOfGaussians_1_5","differenceOfGaussians_1_2","differenceOfHessians","gaborfilt_imagx","gaborfilt_imagy","median_blur","laplace_filter"]
        #df= pd.DataFrame()
        if jj == 0:
            for i, nameOfFilter in enumerate(nameOfFilters_string):
                try:
                    os.makedirs(f"{nameOfFilters_string[i]}")
                    v = name
                except:
                    print('the folder is already created')
                    
        #for i, nameOfFilter in enumerate(nameOfFilters):
        cv2.imwrite(f"{nameOfFilters_string[0]}/{nameOfFilters_string[0]}_{name}.tif", bilateral)
        cv2.imwrite(f"{nameOfFilters_string[1]}/{nameOfFilters_string[1]}_{name}.tif", nlMeans1)
        cv2.imwrite(f"{nameOfFilters_string[2]}/{nameOfFilters_string[2]}_{name}.tif", gaussian_img)
        cv2.imwrite(f"{nameOfFilters_string[3]}/{nameOfFilters_string[3]}_{name}.tif", sobel_h1)
        cv2.imwrite(f"{nameOfFilters_string[4]}/{nameOfFilters_string[4]}_{name}.tif", sobel_v1)
        cv2.imwrite(f"{nameOfFilters_string[5]}/{nameOfFilters_string[5]}_{name}.tif", differenceOfGaussians_1_10)
        cv2.imwrite(f"{nameOfFilters_string[6]}/{nameOfFilters_string[6]}_{name}.tif", differenceOfGaussians_0_5)
        cv2.imwrite(f"{nameOfFilters_string[7]}/{nameOfFilters_string[7]}_{name}.tif", differenceOfGaussians_1_5)
        cv2.imwrite(f"{nameOfFilters_string[8]}/{nameOfFilters_string[8]}_{name}.tif", differenceOfGaussians_1_2)
        cv2.imwrite(f"{nameOfFilters_string[9]}/{nameOfFilters_string[9]}_{name}.tif", differenceOfHessians)
        cv2.imwrite(f"{nameOfFilters_string[10]}/{nameOfFilters_string[10]}_{name}.tif", gaborfilt_imagx)
        cv2.imwrite(f"{nameOfFilters_string[11]}/{nameOfFilters_string[11]}_{name}.tif", gaborfilt_imagy)
        cv2.imwrite(f"{nameOfFilters_string[12]}/{nameOfFilters_string[12]}_{name}.tif", median_blur)
        cv2.imwrite(f"{nameOfFilters_string[13]}/{nameOfFilters_string[13]}_{name}.tif", laplace_filter)
        jj = 1
        # ############################################
        # ##---reshape_filters & make dataframe-----##
        # ############################################
        # gaussian_img = gaussian_img.reshape(-1)
        # df['Gaussian s1'] = gaussian_img
        # sobel_img = sobel_img.reshape(-1)
        # sobel_h1 = sobel_h1.reshape(-1)
        # sobel_v1 = sobel_v1.reshape(-1)
        # df['sobel_img'] = sobel_img
        # df['sobel_v1'] = sobel_v1
        # df['sobel_h1'] = sobel_h1
        # img2 = img.reshape(-1)
        # df['Original Image'] = img2
        # differenceOfHessians = differenceOfHessians.reshape(-1)
        # df['differenceOfHessians'] = differenceOfHessians
        # differenceOfGaussians_1_10 = differenceOfGaussians_1_10.reshape(-1)
        # df['differenceOfGaussians_1_10'] = differenceOfGaussians_1_10
        # differenceOfGaussians_1_5 = differenceOfGaussians_1_5.reshape(-1)
        # df['differenceOfGaussians_1_5'] = differenceOfGaussians_1_5
        # differenceOfGaussians_1_2 = differenceOfGaussians_1_2.reshape(-1)
        # df['differenceOfGaussians_1_2'] = differenceOfGaussians_1_2
        # differenceOfGaussians_0_5 = differenceOfGaussians_0_5.reshape(-1)
        # df['differenceOfGaussians_0_5'] = differenceOfGaussians_0_5
        # gaborfilt_imagx = gaborfilt_imagx.reshape(-1)
        # df['gaborfilt_imagx'] = gaborfilt_imagx
        # gaborfilt_imagy = gaborfilt_imagy.reshape(-1)
        # df['gaborfilt_imagy'] = gaborfilt_imagy
        # median_blur = median_blur.reshape(-1)
        # df['median_blur'] = median_blur
        # laplace_filter = laplace_filter.reshape(-1)
        # df['laplace_filter'] = laplace_filter        
        #     #name= (image[f1].split('\\')[2]).split(".")[0]
        # cv2.imwrite(f"{folder_with_image}/augmented1/{name}_LR_labeled.tif",flipped_img)
        # cv2.imwrite(f"{folder_with_image}/augmented1/{name}_UD_labeled.tif",Vflipped_img)
        # cv2.imwrite(f"{folder_with_image}/augmented1/{name}.tif",img)

main_dir = r'E:\Parisa\samples\New folder'

for i, folder_with_image in enumerate(glob.glob(r'E:\Parisa\samples\New folder\*')):
    filter_images(main_dir,folder_with_image)