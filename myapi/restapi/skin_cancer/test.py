#-----------------------------------
# TRAINING OUR MODEL
#-----------------------------------

# import the necessary packages
import h5py
import time
import numpy as np
import os
import glob
import cv2
from matplotlib import pyplot
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import mahotas
import joblib

num_trees = 1000
fixed_size = tuple((500, 500))
# bins for histogram
bins = 8

# train_test_split size
test_size = 0.10
train_path = "D:/PYTHON/PythonProjects/Django-api/myapi/restapi/skin_cancer/dataset/train"

# seed for reproducing same results
seed = 9

# feature-descriptor-1: Hu Moments
def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

# feature-descriptor-2: Haralick Texture
def fd_haralick(image):
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the haralick texture feature vector
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    # return the result
    return haralick

# feature-descriptor-3: Color Histogram
def fd_histogram(image, mask=None):
    # convert the image to HSV color-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # compute the color histogram
    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(hist, hist)
    # return the histogram
    return hist.flatten()
def eog_histogram(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(image,100,200)
    histo  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(histo, histo)
    # return the histogram
    return histo.flatten()

train_labels = os.listdir(train_path)


# sort the training labels
train_labels.sort()
print(train_labels)
# create all the machine learning models
models = []
#models.append(('RF', RandomForestClassifier(n_estimators=num_trees, random_state=9)))
models.append(('SVM', SVC(random_state=9)))


# variables to hold the results and names
results = []
names = []
scoring = "accuracy"

# import the feature vector and trained labels
h5f_data = h5py.File('output/data.h5', 'r')
h5f_label = h5py.File('output/labels.h5', 'r')

global_features_string = h5f_data['dataset_1']
global_labels_string = h5f_label['dataset_1']

global_features = np.array(global_features_string)
global_labels = np.array(global_labels_string)

h5f_data.close()
h5f_label.close()


# verify the shape of the feature vector and labels
print ("[STATUS] features shape: {}".format(global_features.shape))
print ("[STATUS] labels shape: {}".format(global_labels.shape))

print ("[STATUS] training started...")

# split the training and testing data
(trainDataGlobal, testDataGlobal, trainLabelsGlobal, testLabelsGlobal) = train_test_split(np.array(global_features),
                                                                                          np.array(global_labels),
                                                                                          test_size=test_size,
                                                                                          random_state=seed)

print ("[STATUS] splitted train and test data...")
print ("Train data  : {}".format(trainDataGlobal.shape))
print ("Test data   : {}".format(testDataGlobal.shape))
print ("Train labels: {}".format(trainLabelsGlobal.shape))
print ("Test labels : {}".format(testLabelsGlobal.shape))

# filter all the warnings
import warnings
warnings.filterwarnings('ignore')

# 10-fold cross validation
for name, model in models:
    kfold = KFold(n_splits=10, random_state=None)
    cv_results = cross_val_score(model, trainDataGlobal, trainLabelsGlobal, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# boxplot algorithm comparison
fig = pyplot.figure()
fig.suptitle('Machine Learning algorithm comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()


#-----------------------------------
# TESTING OUR MODEL
#-----------------------------------

# to visualize results
import matplotlib.pyplot as plt

# create the model - Random Forests
clf  = RandomForestClassifier(n_estimators=100, random_state=9)
#clf=SVC(kernel='rbf')
# fit the training data to the model
clf.fit(trainDataGlobal, trainLabelsGlobal)

pred=clf.predict(testDataGlobal)
# print(pred)
# write the classification report to file
w=classification_report(testLabelsGlobal, pred)
print(w)
# path to test data
test_path = "check"

# loop through the test images
for file in glob.glob(test_path + "/*.jpg"):
    # read the image
    image = cv2.imread(file)

    # resize the image
    image = cv2.resize(image, fixed_size)

    ####################################
    # Global Feature extraction
    ####################################
    fv_hu_moments = fd_hu_moments(image)
    fv_haralick   = fd_haralick(image)
    fv_histogram  = fd_histogram(image)
    fv_eoghistogram = eog_histogram(image)
    ###################################
    # Concatenate global features
    ###################################
    start_time = time.time()
    global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments,fv_eoghistogram])

    # predict label of test image
    prediction = clf.predict(global_feature.reshape(1,-1))[0]
    print("--- %s seconds ---" % (time.time() - start_time))
    if prediction==0:
        blurred = cv2.GaussianBlur(image, (11, 11), 0)
        #conver from bgr to hsv
        hsv=cv2.cvtColor(blurred,cv2.COLOR_BGR2HSV)
        #creating mask
        lower_green=np.array([36, 10, 50])
        upper_green=np.array([100, 180, 190])

        red=cv2.inRange(hsv,lower_green,upper_green)
        
        #apply morpological featurs
        kernal = np.ones((5 ,5), "uint8")
        red=cv2.dilate(red,None, iterations=2)
        red = cv2.erode(red, None, iterations=2)
        res=cv2.bitwise_and(image,image,mask=red)
        #draw rectangle
        (contours,hierarchy)=cv2.findContours(red,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
        
        for pic, contour in enumerate(contours):
            if len(contours) != 0:
                area = cv2.contourArea(contour)
                if(area>10):
                    # arduino.write(b'1')
                    # data=arduino.readline().decode('ascii')
                    # print(data)
                    c = max(contours, key = cv2.contourArea)
                    new=cv2.contourArea(c)
                    
                    
                    x,y,w,h = cv2.boundingRect(c)
                    #image=cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
                    image = cv2.rectangle(image,(x-50,y-100),(x+w+100,y+h+10),(0,255,150),10)
                    cv2.putText(image,"",(x-10,y-10),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,150))
    if prediction==1:
        blurred = cv2.GaussianBlur(image, (11, 11), 0)
        #conver from bgr to hsv
        hsv=cv2.cvtColor(blurred,cv2.COLOR_BGR2HSV)
        #creating mask
        lower_green=np.array([36, 10, 50])
        upper_green=np.array([100, 180, 190])

        red=cv2.inRange(hsv,lower_green,upper_green)
        
        #apply morpological featurs
        kernal = np.ones((5 ,5), "uint8")
        red=cv2.dilate(red,None, iterations=2)
        red = cv2.erode(red, None, iterations=2)
        res=cv2.bitwise_and(image,image,mask=red)
        #draw rectangle
        (contours,hierarchy)=cv2.findContours(red,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
        
        for pic, contour in enumerate(contours):
            if len(contours) != 0:
                area = cv2.contourArea(contour)
                if(area>50):
                    # arduino.write(b'1')
                    # data=arduino.readline().decode('ascii')
                    # print(data)
                    c = max(contours, key = cv2.contourArea)
                    new=cv2.contourArea(c)
                    
                    
                    x,y,w,h = cv2.boundingRect(c)
                    #image=cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
                    image = cv2.rectangle(image,(x-50,y-100),(x+w+100,y+h+10),(0,255,150),10)
                    cv2.putText(image,"",(x-10,y-10),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,150))
    if prediction==2:
        blurred = cv2.GaussianBlur(image, (11, 11), 0)
        #conver from bgr to hsv
        hsv=cv2.cvtColor(blurred,cv2.COLOR_BGR2HSV)
        #creating mask
        lower_green=np.array([36, 10, 50])
        upper_green=np.array([100, 180, 190])

        red=cv2.inRange(hsv,lower_green,upper_green)
        
        #apply morpological featurs
        kernal = np.ones((5 ,5), "uint8")
        red=cv2.dilate(red,None, iterations=2)
        red = cv2.erode(red, None, iterations=2)
        res=cv2.bitwise_and(image,image,mask=red)
        #draw rectangle
        (contours,hierarchy)=cv2.findContours(red,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
        
        for pic, contour in enumerate(contours):
            if len(contours) != 0:
                area = cv2.contourArea(contour)
                if(area>10):
                    # arduino.write(b'1')
                    # data=arduino.readline().decode('ascii')
                    # print(data)
                    c = max(contours, key = cv2.contourArea)
                    new=cv2.contourArea(c)
                    
                    
                    x,y,w,h = cv2.boundingRect(c)
                    #image=cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
                    image = cv2.rectangle(image,(x-50,y-100),(x+w+100,y+h+10),(0,255,150),10)
                    cv2.putText(image,"",(x-10,y-10),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,150))                    
    else:
        print("raza mara")

    # show predicted label on image
    cv2.putText(image, train_labels[prediction], (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)

    # display the output image
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()
