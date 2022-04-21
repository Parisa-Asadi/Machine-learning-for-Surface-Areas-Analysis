# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 10:55:48 2022

@author: pza0029@Auburn.edu
Parisa Asadi

This code get an image inwhich the mineral were ranked from high dissolustion rate to low one
for axample
Mineral	Rank Chemical formula*
Quartz	0	SiO2
		
		
Albite	10	NaAlSi ₃O ₈
		
		
Kaolinite	4	Al2Si2O5(OH)4
		
		
Anatase	2	TiO2
		
		
Carbonate	13	CaCO₃/MgCO3·CaCO3
		
		
Biotite	7	K(Mg,Fe++)3[AlSi3O10(OH,F)2
		
		
Muscovite	5	KAl2[AlSi3O10]
		
		
K-feldspar	9	KAlSi3O8
		
		
Siderite	12	Fe(Ca,Mg)(CO3)2
		
		
Smectite/Illite	8	K 0.65 Al 2 [Al 0.65 Si 3.35 O10](OH)2
		
		
Magnetite	11	Fe₃O₄
		
		
Zircon	1	ZrSiO4
		
		
Ilmenite	3	(Fe,Ti)2O3
		
		
Chlorite	6	ClO− 2
		

and the mineral of interst that you are willing to analyze the dissolution risk on its surface.

classOfInterest # change it to the rank of the mineral of interest. as a results it will go through the image and find that rank which are the mineral of interst, then it will check the neighboring pixels to find the maximum neighboring rank and replace it on the surface of the mineral of interst.

the oupt is the dissolution risk assessment map.



Note: the predicted mineral map from RF and u-Net should be reactached to create the same image size to
the gound truth image as a results the output maps would be comparable.

"""

import numpy as np
import cv2

#import os
#os.chdir(r"E:\Parisa\samples\Labels\ColorToNum")
image=cv2.imread(r'E:\Parisa\samples\Test_riskassessment&surfacearea\ranking_riskMaps\Labels\TestToCompareWithUNetResults\Reattached_Test_label.tif',0) ###Segmented color mineral image, each mineral is a different color

def neighbors(radius, row_number, column_number, image=image, classOfInterest=13):
    if image[row_number,column_number]==classOfInterest:
        return [[image[i,j] if  i >= 0 and i < (image.shape[0]) and j >= 0 and j < (image.shape[1]) else 0
                    for j in range(column_number-radius, column_number+radius+1)]
                        for i in range(row_number-radius, row_number+radius+1)]
    else:
        return 0




image2=np.zeros((image.shape[0],image.shape[1]),dtype=np.uint8)
radius=1 #the kernel size is 9 for radius 1, and so on ..... 
classOfInterest=9 # change it to the rank of the mineral of interest. as a results it will go through the image and find that rank which are the mineral of interst, then it will check the neighboring pixels to find the maximum neighboring rank and replace it on the surface of the mineral of interst. 
for row_number in range(0,image.shape[0]):
    for column_number in range(0,image.shape[1]):
        a= neighbors(radius, row_number, column_number, image=image, classOfInterest=classOfInterest)
        image2[row_number,column_number] = np.max(np.array(a))

cv2.imwrite(r"E:\Parisa\samples\Test_riskassessment&surfacearea\ranking_riskMaps\Labels\TestToCompareWithUNetResults\RiskMap_LabeledData\LabelData_rank9.tif",image2)#cv2 always Write BGR so you should convert it. if it does not work try cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

