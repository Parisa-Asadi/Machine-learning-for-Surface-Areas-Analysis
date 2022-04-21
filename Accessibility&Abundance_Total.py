
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 13:59:48 2022

@author: parisa Asadi
This code consider that the class ==0 is pores and get the accessibitly to this class.
 thus, mineral numbering shuld start from 1 ...

"""
########################################################################################################
##### it processes accessibility and abundance for each image ##########################################

def accessibility_abundance_EachImage(Imagename,ImagesDirectory):

    
    import pandas as pd

    
    os.chdir(fr'{ImagesDirectory}') #main folder
    
    image=cv2.imread(f'{Imagename}',0) ###Segmented color mineral image, each mineral is a different color
    image=image.astype(np.uint8)
    
    ######total pore/mineral interface
    interface=0
    
    ######pixels for mineral accessibility quantification
    AQuartz=0
    AAlbite=0
    AKaolinite=0
    ASmectite_Illite=0
    AK_Feldspar=0
    AColoride=0
    AMagnetite=0
    AAnatase=0
    ACalcite_Dolomite=0
    AMuscovite=0
    ACalcite_Dolomite=0
    AMuscovite=0
    AZircon=0
    AIlmenite=0
    ASiderite=0
    ABiotite=0
    Mineral_Names = ["Pore","Quartz", "Albite", "Kaolinite", "Smectite_Illite", "K_Feldspar", "Coloride", "Magnetite", "Anatase", "Calcite_Dolomite", "Muscovite", "Zircon", "Ilmenite", "Siderite", "Biotite"]
    df = pd.DataFrame([])
    
    
    for row_number in range(1,image.shape[0]-1):
        for column_number in range(1,image.shape[1]-1):
            if ((image[row_number-1,column_number-1]==0) or (image[row_number-1,column_number]==0) or (image[row_number-1,column_number+1]==0) or (image[row_number,column_number-1]==0) or (image[row_number,column_number+1]==0) or (image[row_number+1,column_number-1]==0) or (image[row_number+1,column_number]==0) or (image[row_number+1,column_number+1]==0)):
                #it is an accessible mineral pixel, so lets figure out which mineral it is  
                    if (image[row_number,column_number]==1): #quartz
                        AQuartz=AQuartz+1
                        interface = interface+1
                    elif (image[row_number,column_number]==2): #Albite
                        AAlbite=AAlbite+1
                        interface = interface+1
                    
                    elif (image[row_number,column_number]==3): #Kaolinite
                        AKaolinite=AKaolinite+1
                        interface = interface+1
                        
                    elif (image[row_number,column_number]==4): #Smectite_Illite
                        ASmectite_Illite=ASmectite_Illite+1
                        
                        interface = interface+1
                
                    
                    elif (image[row_number,column_number]==5): #AKfeldspar
                        AK_Feldspar=AK_Feldspar+1
                        interface = interface+1
                        
                    elif (image[row_number,column_number]==6): #Colorite
                        AColoride=AColoride+1
                        interface = interface+1
                    
                    elif (image[row_number,column_number]==7): #quartz
                        AMagnetite=AMagnetite+1
                        interface = interface+1
                    
                    elif (image[row_number,column_number]==8): #quartz
                        AAnatase=AAnatase+1
                        interface = interface+1
                    
                    elif (image[row_number,column_number]==9): #quartz
                        ACalcite_Dolomite=ACalcite_Dolomite+1
                        interface = interface+1
                    
                    elif (image[row_number,column_number]==10): #quartz
                        AMuscovite=AMuscovite+1
                        interface = interface+1
                    
                    elif (image[row_number,column_number]==11): #quartz
                        AZircon=AZircon+1
                        interface = interface+1
                        
                    elif (image[row_number,column_number]==12): #quartz
                        AIlmenite=AIlmenite+1
                        interface = interface+1
                    
                    elif (image[row_number,column_number]==13): #quartz
                        ASiderite=ASiderite+1
                        interface = interface+1
                    
                    elif (image[row_number,column_number]==14): #quartz
                        ABiotite=ABiotite+1
                        interface = interface+1
                    
    #calculate total mineral pixles and total accessible mineral SA
    # Atotal=Aquartz+AKfeldspar+Akaolinite+Amuscovite+Amagnetite+ABiotite+AAlbite+ASmectite_Illite+AColorite+AAnatase+ACalcite_Dolomite+AZircon+AIlmenite+ASiderite+ABiotite
    # Atotal=sum(AccMinerals)
    # #Calculate accessibility
    # percentquartzACCESS=(100*Aquartz)/Atotal
    # percentKfeldsparACCESS=(100*AKfeldspar)/Atotal
    # percentkaoliniteACCESS=(100*Akaolinite)/Atotal
    # percentmuscoviteACCESS=(100*Amuscovite)/Atotal
    # percentmagnetiteACCESS=(100*Amagnetite)/Atotal
    # percentABiotiteACCESS=(100*ABiotite)/Atotal
    # percentAlbiteACCESS=(100*AAlbite)/Atotal
    # percentSmectite_IlliteACCESS=(100*ASmectite_Illite)/Atotal
    # percentAColoriteACCESS=(100*AColorite)/Atotal
    # percentAnataseACCESS=(100*AAnatase)/Atotal
    # percentCalcite_DolomiteACCESS=(100*ACalcite_Dolomite)/Atotal
    # percentZirconACCESS=(100*AZircon)/Atotal
    # percentSideriteACCESS=(100*ASiderite)/Atotal
    
    # totalACCESS=percentquartzACCESS+percentKfeldsparACCESS+percentkaoliniteACCESS+percentmuscoviteACCESS+percentmagnetiteACCESS+percentABiotiteACCESS+percentSmectite_IlliteACCESS+percentAColoriteACCESS+percentAColoriteACCESS+percentAnataseACCESS+percentCalcite_DolomiteACCESS+percentZirconACCESS+percentSideriteACCESS
    
    #write data to file
    
    
    df['Total_Num_pixels'] = [image.shape[0] * image.shape[1]]
    df['Total_accessible_pore_mineral_interface_pixels'] = [interface]
    print("this is Quartz ", locals()[f"A{Mineral_Names[1]}"])
    if locals()[f"A{Mineral_Names[1]}"] !=0:
        print(locals()[f"A{Mineral_Names[1]}"])
    df['Accessible_Quartz'] = AQuartz
    df['Accessible_Albite'] = AAlbite
    df['Accessible_Kaolinite'] = AKaolinite
    df['Accessible_Smectite_Illite'] = ASmectite_Illite
    df['Accessible_K_Feldspar'] = AK_Feldspar
    df['Accessible_Coloride'] = AColoride
    df['Accessible_Magnetite'] = AMagnetite
    df['Accessible_Anatase'] = AAnatase
    df['Accessible_Calcite_Dolomite']= ACalcite_Dolomite
    df['Accessible_Muscovite'] = AMuscovite
    df['Accessible_Zircon']= AZircon
    df['Accessible_Ilmenite'] = AIlmenite
    df['Accessible_Siderite']= ASiderite
    df['Accessible_Biotite'] = ABiotite
    
    
    Mineral_counts = np.unique(image,return_counts=True)
    for Num, minerals in enumerate(Mineral_counts[0]):
          
        df[f"count_{Mineral_Names[int(minerals)]}"]=[Mineral_counts[1][Num]]
    for Num, minerals in enumerate(Mineral_Names):
        if not f"count_{minerals}" in df.columns:
            df[f"count_{minerals}"]=[0]
    df['Total_pixels'] = sum(Mineral_counts[1])
    # df['Total # pixels'] = Npixel
    # df['Total accessible mineral pixels'] = Atotal
    # df['Total accessible pore/mineral interface pixels'] = interface
    # df['Accessible percent quartz'] = percentquartzACCESS
    # df['Accessible percent Kfeldspar'] = percentKfeldsparACCESS
    # df['Accessible percent Biotite'] = percentABiotiteACCESS
    # df['Accessible percent kaolinite'] = percentkaoliniteACCESS
    # df['Accessible percent muscovite'] = percentmuscoviteACCESS
    # df['Accessible percent magnetite'] = percentmagnetiteACCESS
    # df['Accessible percent Albite'] = percentAlbiteACCESS
    # df['Accessible percent Colorite'] = percentAColoriteACCESS
    # df['Accessible percent Anatase'] = percentAnataseACCESS
    # df['Accessible percent Zircon'] = percentZirconACCESS
    # df['Accessible percent Calcite_Dolomite'] = percentCalcite_DolomiteACCESS
    # df['Accessible percent Siderite'] = percentSideriteACCESS
    #df['total mineral access'] = totalACCESS
    return df  


######################################################################################################## to process all images
######################################################################################
######################## read libraries ##############################################
import os

import glob
import cv2
import numpy as np
import pandas as pd
import natsort


######################################################################################
######################## get images ##################################################

##################### input images##########################
SIZE_X = 128
SIZE_Y = 128
n_classes=15#Number of minerals+pore
N_images = 2835
df_Final = pd.DataFrame([])

ImagesDirectory = r"E:\Parisa\samples\Test_riskassessment&surfacearea\pooled_test\U_Net" 

n=len(glob.glob(f"{ImagesDirectory}\*.tif"))

#Capture training image info as a list
img_lst= glob.glob(f"{ImagesDirectory}\*.tif")

img_lst = natsort.natsorted(img_lst,reverse=False)



#### it processes all image in the ImagesDirectory and provide theresults for each image #########
i = 0
for i_image, image in enumerate(img_lst):

    df11 = accessibility_abundance_EachImage(image,ImagesDirectory)
    if i == 0:
        df_Final = df11.copy()
    else:
        df_Final = pd.concat([df_Final, df11], axis=0, ignore_index=True)
    i = i + 1
    print(i)
#####################################################################
### it processes all image in the ImagesDirectory and provide the Final results for the sample #####    
# for each mineral we got the accesibiltiy by getting the total accessable mineral pixel to pore / total interface. We had 15 classes 14 mineral + pore. 0=pore
Mineral_Names_1 = ["Quartz", "Albite", "Kaolinite", "Smectite_Illite", "K_Feldspar", "Coloride", "Magnetite", "Anatase", "Calcite_Dolomite", "Muscovite", "Zircon", "Ilmenite", "Siderite", "Biotite"]
df_Sample_summary = pd.DataFrame([])
for vvalue in range(0,14):
   df_Sample_summary[f"Abundance_{Mineral_Names_1[vvalue]}"] =[sum(df_Final[f"count_{Mineral_Names_1[vvalue]}"]) / (sum(df_Final['Total_pixels']) - sum(df_Final["count_Pore"]))]
   df_Sample_summary[f"Accessible_{Mineral_Names_1[vvalue]}"] =[sum(df_Final[f"Accessible_{Mineral_Names_1[vvalue]}"]) / sum(df_Final['Total_accessible_pore_mineral_interface_pixels'])]
 ###porosity
df_Sample_summary["Porosity"] =[sum(df_Final["count_Pore"]) / sum(df_Final['Total_pixels'])]

   
######output is the csv file and a summary csv file####################    
df_Final.to_csv('Mineral_Accesibiltiy&Abundance.csv', index=False)
df_Sample_summary.to_csv('Sample_summary_Mineral_Accesibiltiy&Abundance.csv', index=False)

