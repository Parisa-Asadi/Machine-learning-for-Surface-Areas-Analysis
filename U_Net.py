
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 10:50:13 2021
@author: pza0029
Note: first download "simple_multi_unet_model_Original" file and have it on the directory.

"""
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
print(tf.__version__)
tf.test.is_gpu_available(
    cuda_only=False, min_cuda_compute_capability=None)
tf.config.list_physical_devices('GPU')
import tensorflow as tf

######################################################################################
######################## read libraries ##############################################\
#####set directory to the place that I have saved my simple u_net code
os.chdir(r'C:\Users\pza0029\Box\Ph.D Civil\My_total_works\Codes_since_3.8.2021') #main folder
#####
#from simple_multi_unet_model import multi_unet_model #Uses softmax 
from simple_multi_unet_model_Original import multi_unet_model 
import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import tensorflow

import tifffile as tiff
import natsort
from sklearn.model_selection import train_test_split


######################################################################################
######################## get images ##################################################
#get time
import time
start_time = time.time()



end=time.time()
print("--- %s seconds ---" % (end - start_time))
##################### input images##########################
SIZE_X = 128
SIZE_Y = 128
n_classes=15#Number of classes for segmentation
N_images = 2835

#n is the number of input layers ready to overlay
n=len(glob.glob(r"E:\Parisa\samples\sample - Copy\*")) 
# i for layers, j for images in each 

img_lst2 = glob.glob(r"E:\Parisa\samples\sample - Copy\*") 
img_lst2 = natsort.natsorted(img_lst2,reverse=False)
stacked_data = np.empty((N_images, SIZE_X, SIZE_Y, n))
for i, directory_path in enumerate(img_lst2): 
    train_images = []
    for j, img_path in enumerate(natsort.natsorted(glob.glob(os.path.join(directory_path, 'cropped',"*.tif")),reverse=False)):
        # print(img_path)
        try:
            img = tiff.imread(img_path)
            #img= img/np.max(img)
            
        except:
            img = cv2.imread(img_path,0)
            #img= img/np.max(img)
            print("0")
        train_images.append(img)
        
    train_images = np.array(train_images)   
    
    
    stacked_data[:, :, :, i] = train_images

##################### split input images#############

del directory_path
del img_path
del img 

########augmentation
def Augmentation_Big_stack_images(stacked_data):
    stacked_data1 = np.empty(((stacked_data.shape[0])*3, stacked_data.shape[1], stacked_data.shape[2], stacked_data.shape[3]))
    stacked_data1[ 0:stacked_data.shape[0], :, :,:] = stacked_data[:,:,:,:]
     
    flipped_img= stacked_data[:,:, ::-1,:]
    #flipped_img = np.fliplr(stacked_data) #it reverse the columns order
    Vflipped_img = stacked_data[:,::-1, :,:]
    #Vflipped_img = np.flipud(stacked_data) #it reverse the rows order
    stacked_data1[ stacked_data.shape[0]:2*stacked_data.shape[0], :, :,:] = flipped_img[:,:,:,:]
    stacked_data1[ 2*stacked_data.shape[0]:3*stacked_data.shape[0], :, :,:] = Vflipped_img[:,:,:,:]       
    return stacked_data1

stacked_data = Augmentation_Big_stack_images(stacked_data)


train_images, test_images = train_test_split(stacked_data, test_size=0.3, shuffle=True, random_state=42)
test_images, val_images = train_test_split(test_images , test_size=0.5,shuffle=True, random_state=42)

del stacked_data
#stacked_data = Augmentation_Big_stack_images(stacked_data)
####################
#Capture mask/label info as a list
train_masks = [] 
img_lst3 = glob.glob(r"E:\Parisa\samples\Labels\ColorToNum\cropped\*.tif") 

img_lst3 = natsort.natsorted(img_lst3,reverse=False)
for i, directory_path in enumerate(img_lst3):
    mask = cv2.imread(directory_path,0)
    #mask = cv2.resize(mask, (SIZE_Y, SIZE_X), interpolation = cv2.INTER_NEAREST)  #Otherwise ground truth changes due to interpolation
    train_masks.append(mask)

    
#Convert list to array for machine learning processing          
train_masks = np.array(train_masks)

##################### split input masks#############

del directory_path
del mask

########augmentation
def Augmentation_Big_stack_masks(stacked_data):
    stacked_data1 = np.empty(((stacked_data.shape[0])*3, stacked_data.shape[1], stacked_data.shape[2]))
    stacked_data1[ 0:stacked_data.shape[0], :, :] = stacked_data[:,:,:]
    
    flipped_img= stacked_data[:,:, ::-1]
    Vflipped_img = stacked_data[:,::-1, :]  
    # flipped_img = np.fliplr(stacked_data) #it reverse the columns order
    # Vflipped_img = np.flipud(stacked_data) #it reverse the rows order
    stacked_data1[ stacked_data.shape[0]:2*stacked_data.shape[0], :, :] = flipped_img[:,:,:]
    stacked_data1[ 2*stacked_data.shape[0]:3*stacked_data.shape[0], :, :] = Vflipped_img[:,:,:]       
    return stacked_data1

train_masks = Augmentation_Big_stack_masks(train_masks)



###############################################
###############################################
#Encode labels... but multi dim array so need to flatten, encode and reshape

def FlattenEncodeReshape(test_masks):
    from sklearn.preprocessing import LabelEncoder
    labelencoder = LabelEncoder()
    nn, h, w = test_masks.shape
    test_masks_reshaped = test_masks.reshape(-1,1)
    test_masks_reshaped_encoded = labelencoder.fit_transform(test_masks_reshaped)
    test_masks_encoded_original_shape = test_masks_reshaped_encoded.reshape(nn, h, w) 
    return test_masks_encoded_original_shape , labelencoder


train_masks_encoded_original_shape,  labelencoder = FlattenEncodeReshape(train_masks)
np.unique(train_masks_encoded_original_shape)
np.unique(train_masks_encoded_original_shape,return_counts=True)



#################################################
###############################################use this part if your test and train is seperated otherwise comment it
X_train = train_images
X_test = test_images
y_train = train_masks_encoded_original_shape

X_val= val_images


###############################################
##################################################
#del path to make sure they will not overwrite
del train_images
del train_masks
del train_masks_encoded_original_shape



################################################
###############################################
print("Class values in the dataset are ... ", np.unique(y_train))  

#n_classes = 7
def ToCategorical(y_train, n_classes):
    from tensorflow.keras.utils import to_categorical
    train_masks_cat = to_categorical(y_train, num_classes=n_classes)
    y_train_cat = train_masks_cat.reshape((y_train.shape[0], y_train.shape[1], y_train.shape[2], n_classes))
    return y_train_cat

y_train_cat = ToCategorical(y_train, n_classes)

y_train_cat , y_test_cat = train_test_split(y_train_cat , test_size=0.3, shuffle=True, random_state=42)
y_test_ca , yval_cat = train_test_split(y_test_cat , test_size=0.5,shuffle=True, random_state=42)

y_train , y_test = train_test_split(y_train, test_size=0.3, shuffle=True, random_state=42)
y_test , yval = train_test_split(y_test , test_size=0.5,shuffle=True, random_state=42)

##################################################
#del path to make sure they will not overwrite
del y_train
del yval

#########################################################################################################################################
####get_model and train the U-net with three method 1-get from multi_unet model, 2- Autotune of tensorflow or 3- segmentation-models#####

##################################get_model and train the U-net with three method 1-get from multi_unet model########
#############################
IMG_HEIGHT = X_train.shape[1]
IMG_WIDTH  = X_train.shape[2]
IMG_CHANNELS = X_train.shape[3]
import segmentation_models as sm
from tensorflow.keras.metrics import MeanIoU

def get_model():
    return multi_unet_model(n_classes=n_classes, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS)

model = get_model()

#############################
#Reused parameters in all models
#n_classes=9
activation='softmax'
# import keras
LR = 0.0001
optim = tensorflow.keras.optimizers.Adam(LR)
focal_loss = sm.losses.CategoricalFocalLoss(gamma=2)


metrics1 = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5), ['accuracy'],[tf.keras.metrics.AUC()]]
model.compile(optim, loss= focal_loss , metrics=metrics1)
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())


#If starting with pre-trained weights. 
#model.load_weights('U_Net_project2_pooled_model_BSE&EDS&filter_300epochs_2GFocal_Ba3_M1_lrp0001_128.hdf5') 
########################################################################


os.chdir(r"E:\Parisa\samples\Diff_inputVariabelSet\Results_04102022\U_Net\Pooled_BSE")

check_point =  tensorflow.keras.callbacks.ModelCheckpoint('best_model_BSE.h5', monitor='val_loss', save_best_only=True)
stop_early = tensorflow.keras.callbacks.EarlyStopping(monitor="val_loss", mode='auto', patience=100)  
start_time=  time.time()               
history = model.fit(X_train, y_train_cat, 
                    batch_size = 3,
                    verbose=1, 
                    epochs=300, 
                    validation_data=(X_val, yval_cat), 
                    #class_weight=class_weights,
                    shuffle=False,
                    callbacks=[stop_early, check_point])
end=time.time()
print("--- %s seconds ---" % (end - start_time))

model.save('U_Net_project2_pooled_model_BSE_300epochs_2GFocal_Ba3_M1_lrp0001_128.hdf5')
##sandstone_300_epochs_catXentropy_acc_Unet_F1Score.hdf5

############################################################
######################################################## end of trainings

################################## Evaluate the model ################################################################
	# evaluate model
# _, acc = model.evaluate(X_test, y_test_cat)
# print("Accuracy is = ", (acc * 100.0), "%")

###
#plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
#plt.show()
#plt.savefig("squares.png")
plt.savefig("U_Net_project2_pooled_BSE&EDS&Filter_Loss.tif", dpi=300)
loss = history.history['iou_score']
val_loss = history.history['val_iou_score']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training iou')
plt.plot(epochs, val_loss, 'r', label='Validation iou')
plt.title('Training and validation iou')
plt.xlabel('Epochs')
plt.ylabel('iou')
plt.legend()
plt.savefig("U_Net_project2_pooled_BSE&EDS&Filter_iou.tif", dpi=300)
plt.show()

loss = history.history['accuracy']
val_loss = history.history['val_accuracy']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training accuracy')
plt.plot(epochs, val_loss, 'r', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.legend()
plt.savefig("U_Net_project2_pooled_BSE&EDS&Filter_accuracy.tif", dpi=300)
plt.show()
################################## validation


y_pred=model.predict(X_test)
y_pred_argmax=np.argmax(y_pred, axis=3)


import tensorflow 
m = tensorflow.keras.metrics.Accuracy()
m.update_state(y_test, y_pred_argmax)
m.result().numpy()

tensorflow.math.confusion_matrix(y_test.reshape(-1), y_pred_argmax.reshape(-1),name=True)

################################################## IOU & F1 score &  AUC

#Using built in keras function
from tensorflow.keras.metrics import MeanIoU
n_classes = 15
IOU_keras = MeanIoU(num_classes=n_classes)  
IOU_keras.update_state(y_test, y_pred_argmax)
print("Mean IoU =", IOU_keras.result().numpy())

import sklearn
Y1=y_test.reshape(-1)
Y2=y_pred_argmax.reshape(-1)
Y2_df = pd.DataFrame(Y2)
import os
os.chdir(r"E:\Parisa\samples\Diff_inputVariabelSet\Results_04102022\U_Net\pooled_BSE&EDS&Filter")
Y2_df.to_csv('pred_test_images_U_net_project2_pooled_BSE&EDS&Filter.csv')

sklearn.metrics.f1_score(Y1, Y2, average='macro')
from sklearn.metrics import classification_report
print(classification_report(Y1,Y2)) 


#####################################################################
##################################################################### SAVE HISTORY (loss, val)
os.chdir(r"E:\Parisa\samples\Diff_inputVariabelSet\Results_04102022\U_Net\pooled_BSE&EDS&Filter")
pd.DataFrame.from_dict(history.history).to_csv("history_Pooled_BSE&EDS&EDS.csv")
################plotting############################################
ab=800
fig=plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_images[ab,:,:,1], cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
#plt.imshow(test_masks[20,:,:], cmap='jet')
plt.imshow(y_test[ab,:,:])
plt.subplot(233)
plt.title('Testing pred.')
plt.imshow(y_pred_argmax[ab,:,:])
plt.show()
fig.savefig(fr"Image{ab}_U_Net_project2_pooled_BSE&EDS_test.tif", dpi=300)
cv2.imwrite(fr"Image{ab}_U_Net_project2_pooled_BSE&EDS_pred1.tif",y_pred_argmax[ab,:,:])#cv2 always Write BGR so you should convert it. if it does not work try cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
cv2.imwrite(fr"Image{ab}U_Net_project2_pooled_BSE&EDS_test2.tif",y_test[ab,:,:])
# #cv2.imshow('RGB Image2',train_masks[0])
######################################################################################################################
