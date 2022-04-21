# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 09:50:31 2021

@author: pza0029
RF for project2 
it get sorted images and apply RF on them
"""
######################################################################################
######################## read libraries ##############################################
import os
#####
import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import tifffile as tiff
import natsort
from sklearn.model_selection import train_test_split

######################################################################################
######################## get images ##################################################

##################### input images##########################

#Resizing images, if needed

SIZE_X = 128
SIZE_Y = 128
n_classes=15 #Number of classes for segmentation
N_images = 2835

n=len(glob.glob(r"E:\Parisa\samples\Diff_inputVariabelSet\sample_BSE&EDS&Filter\*"))
#Capture training image info as a list
img_lst= glob.glob(r"E:\Parisa\samples\Diff_inputVariabelSet\sample_BSE&EDS&Filter\*")

img_lst = natsort.natsorted(img_lst,reverse=False)
stacked_data = np.empty((N_images, SIZE_X, SIZE_Y, n))
for i, directory_path in enumerate(img_lst): 
    train_images = []
    img_lst1= glob.glob(os.path.join(directory_path, 'cropped', "*.tif"))
    img_lst1 = natsort.natsorted(img_lst1,reverse=False)
    for j, img_path in enumerate(img_lst1):
        
        
        try:
            img = tiff.imread(img_path)
            #img= img/np.max(img)
        except:
            img = cv2.imread(img_path,0)
            #img= img/np.max(img)        
        train_images.append(img)
        
#Convert list to array for machine learning processing      
   
    train_images = np.array(train_images)   
    
    stacked_data[:, :, :, i] = train_images

    
##################### split input images#############
train_images, test_images = train_test_split(stacked_data, test_size=0.3, shuffle=True, random_state=42)
test_images, val_images = train_test_split(test_images , test_size=0.5,shuffle=True, random_state=42)
##################### reshape as 1D array ##############
del stacked_data
nn,xx,yy,zz=np.shape(train_images)
nnT,xxT,yyT,zzT=np.shape(test_images)
train_images = train_images.reshape(nn, -1, zz).reshape(-1, zz)
test_images = test_images.reshape(nnT, -1, zzT).reshape(-1, zzT)
# dd=np.empty((xx,yy,zz))
# dd[:,:,:]=train_images[1,:,:,:]
del directory_path
del img_path
del img 


##################### labels images#####################################
#Capture mask/label info as a list
train_masks = [] 

img_lst2 = glob.glob(r"E:\Parisa\samples\Labels\ColorToNum\cropped\*.tif") 
img_lst2 = natsort.natsorted(img_lst2,reverse=False)


for directory_path in img_lst2:
    mask = cv2.imread(directory_path, 0)       
    
    train_masks.append(mask)

#Convert list to array for machine learning processing          
train_masks = np.array(train_masks)
train_masks, test_masks = train_test_split(train_masks , test_size=0.3, shuffle=True, random_state=42)
test_masks, val_masks = train_test_split(test_masks , test_size=0.5,shuffle=True, random_state=42)
#del path to make sure they will not overwrite
del directory_path
#del mask_path
del mask


##################### reshape as 1D array ###########
nn_mask,xx_mask,yy_mask=np.shape(train_masks)
nnT_mask,xxT_mask,yyT_mask=np.shape(test_masks)
train_masks = train_masks.reshape(nn_mask, -1).reshape(-1)
test_masks = test_masks.reshape(nnT_mask, -1).reshape(-1)


######################################################################################
######################## balance data ################################################ if needed or use weight to balance the data
# # Upsample minority class
# def resampling(df_minority,m):
#     from sklearn.utils import resample
#     df_minority_upsampled = resample(df_minority, 
#                                       replace=True,     # sample with replacement
#                                       n_samples=m,    # to match majority class
#                                       random_state=123) # reproducible results
#     return  df_minority_upsampled
# # df_upsampled = pd.concat([df_majority, df_minority_upsampled])
 
# # df_upsampled["unique_leafwetness_binary$unique_leafwetness_binary"].value_counts()

# train_images1 = pd.DataFrame(train_images)
# train_images1["label"] = (train_masks)
# #locals()[f"df_minority_{ii} = train_images1[train_images1["label"]=={ii}]"]
# #df_minority_0= resampling(df_minority_0,1000000)
# #df_upsampled1 = pd.concat([train_images1, train_images2], axis=1)
# df_minority_0 = train_images1[train_images1["label"]==0]
# df_minority_1 = train_images1[train_images1["label"]==1]
# df_minority_2 = train_images1[train_images1["label"]==2]
# df_minority_3 = train_images1[train_images1["label"]==3]
# df_minority_4 = train_images1[train_images1["label"]==4]
# df_minority_5 =train_images1[train_images1["label"]==5]
# df_minority_6 = train_images1[train_images1["label"]==6]
# df_minority_7= train_images1[train_images1["label"]==7]
# df_minority_8 = train_images1[train_images1["label"]==8]
# df_minority_9 =train_images1[train_images1["label"]==9]
# df_minority_10 = train_images1[train_images1["label"]==10]
# df_minority_11 = train_images1[train_images1["label"]==11]
# df_minority_12 =train_images1[train_images1["label"]==12]
# df_minority_13 = train_images1[train_images1["label"]==13]
# df_minority_14 = train_images1[train_images1["label"]==14]


# df_minority_0= resampling(df_minority_0,5000000)
# df_minority_1= resampling(df_minority_1,5000000)
# df_minority_2= resampling(df_minority_2,5000000)
# df_minority_3= resampling(df_minority_3,5000000)
# df_minority_4= resampling(df_minority_4,5000000)
# df_minority_5= resampling(df_minority_5,5000000)
# df_minority_6= resampling(df_minority_6,5000000)
# df_minority_7= resampling(df_minority_7,5000000)
# df_minority_8= resampling(df_minority_8,5000000)
# df_minority_9= resampling(df_minority_9,5000000)
# df_minority_10= resampling(df_minority_10,5000000)
# df_minority_11= resampling(df_minority_11,5000000)
# df_minority_12= resampling(df_minority_12,5000000)
# df_minority_13= resampling(df_minority_13,5000000)
# df_minority_14= resampling(df_minority_14,5000000)




# df_upsampled = pd.concat([df_minority_0, df_minority_1, df_minority_2, df_minority_3, df_minority_4, df_minority_5, df_minority_6, df_minority_7, df_minority_8, df_minority_9, df_minority_10, df_minority_11, df_minority_12, df_minority_13, df_minority_14])
######################################################################################
######################## Random Forest ###############################################
#RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=0, n_estimators=50, min_samples_split=5,n_jobs=5, max_depth=32,class_weight="balanced")

model.fit(train_images, train_masks) 
#model.fit( df_upsampled.drop(["label"], axis=1), df_upsampled["label"]) 


######predict & confusion matrix
prediction = model.predict(test_images)
#prediction = loaded_model.predict(test_images)
test_images_df = pd.DataFrame(prediction)
import os
os.chdir(r'E:\Parisa\samples\Diff_inputVariabelSet\Results_04102022\RF') #main folder
test_images_df.to_csv('pred_test_images_RF_project2_Pooled_BSE&EDS&Filter.csv')


result = model.score(test_images , test_masks)
print(result)

from sklearn.metrics import confusion_matrix
confusion_S1 = confusion_matrix(test_masks, prediction,labels=np.array(np.unique(test_masks)))
print (confusion_S1)
import seaborn as sns

confusion_s1_df = pd.DataFrame(confusion_S1)
# confusion_s1_df.index= ['0', '68', '76', '105', '110', '117', '118','179', '211']
# confusion_s1_df.columns= ['0', '68', '76', '105', '110', '117', '118','179', '211']
res = sns.heatmap(confusion_s1_df, annot=True, fmt="d" )
#res.invert_yaxis()
confusion_s1_df.to_csv('confusion_matrix_RF034.csv')



from sklearn.metrics import classification_report
target_names = [ '0','1', '2', '3',"4","5","6"]
print(classification_report(test_masks,prediction))#, target_names=target_names))

#Using built in keras function
from tensorflow.keras.metrics import MeanIoU
n_classes = 15
IOU_keras = MeanIoU(num_classes=n_classes)  
IOU_keras.update_state(test_masks,prediction)
print("Mean IoU =", IOU_keras.result().numpy())

import sklearn
sklearn.metrics.f1_score(test_masks,prediction, average='macro')


########### summarize feature importance ################
importance = model.feature_importances_
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
from matplotlib import pyplot as plt
# plot feature importance
plt.bar([x for x in range(len(importance))], importance)
plt.show()
plt.savefig('feature importance_RF10features.png')
#same as above but with the name of features
feature_importances = pd.DataFrame(model.feature_importances_ )#, index = image1.columns,columns=['importance']).sort_values('importance',ascending=False)
feature_importances.to_csv('feature_importances_project2_Pooled_BSE&EDS&Filter.csv')

######################################################################################
######################## Save and load model for future use ##########################
filename = 'RF_project2_Pooled_BSE&EDS&Filter.sav'
import pickle
pickle.dump(model, open(filename, 'wb'))

#Load model.... 
#os.chdir(r"E:\Parisa\samples\Diff_inputVariabelSet\Results_04102022\RF")
#loaded_model = pickle.load(open(filename, 'rb'))
#del stacked_data
# nn,xx,yy,zz=np.shape(stacked_data)
# stacked_data = stacked_data.reshape(nn, -1, zz).reshape(-1, zz)
# prediction = loaded_model.predict(stacked_data)
pred3=prediction.reshape(nnT_mask,xxT_mask,yyT_mask)
test_images=test_images.reshape(nnT_mask,xxT_mask,yyT_mask,15)
test_masks=test_masks.reshape(nnT_mask,xxT_mask,yyT_mask)

ab=1
fig=plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_images[ab,:,:,1], cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
#plt.imshow(test_masks[20,:,:], cmap='jet')
plt.imshow(test_masks[ab,:,:])
plt.subplot(233)
plt.title('Testing pred.')
plt.imshow(pred3[ab,:,:])
plt.show()
fig.savefig(fr"Image{ab}_RF_project2_Pooled_BSE&EDS&Filter_both.tif", dpi=300)
cv2.imwrite(fr"Image{ab}_RF_project2_Pooled_BSE&EDS&Filter_pred.tif",pred3[ab,:,:])#cv2 always Write BGR so you should convert it. if it does not work try cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
cv2.imwrite(fr"Image{ab}_RF_project2_Pooled_BSE&EDS&Filter_test.tif",test_masks[ab,:,:])
