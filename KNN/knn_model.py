# import necessary packages

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import  train_test_split
from sklearn.metrics import classification_report
# from pyimagesearch import simplepreprocessor
# from pyimagesearch import simpledatasetloader
from imutils import paths
import os
import glob
import cv2
import numpy as np

def getListOfFiles(dirName):
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)

    return allFiles

imagePaths = getListOfFiles("/home/baiyang/liyazhao/train/") ## Folder structure: datasets --> sub-folders with labels name
#print(imagePaths)
testimagePaths = getListOfFiles("/home/baiyang/liyazhao/test/") ## Folder structure: datasets --> sub-folders with labels name




data = []
testdata = []
lables = []
testlables = []
c = 0 ## to see the progress
for image in imagePaths:

    lable = os.path.split(os.path.split(image)[0])[1]
    lables.append(lable)

    img = cv2.imread(image)
    img = cv2.resize(img, (32, 32), interpolation = cv2.INTER_AREA)
    data.append(img)
    c=c+1
    # print(c)
for image in testimagePaths:

    lablea = os.path.split(os.path.split(image)[0])[1]
    testlables.append(lablea)

    img = cv2.imread(image)
    img = cv2.resize(img, (32, 32), interpolation = cv2.INTER_AREA)
    testdata.append(img)
    c=c+1
    # print(c)

# print(lables)

# encode the labels as integer
data = np.array(data)
lables = np.array(lables)
testdata = np.array(data)
testlables = np.array(lables)


le = LabelEncoder()
lables = le.fit_transform(lables)
testlables = le.fit_transform(testlables)




myset = set(lables)
myseta = set(testlables)

# print(myset)

dataset_size = data.shape[0]
data = data.reshape(dataset_size,-1)

testedataset_size = testdata.shape[0]
testdata = testdata.reshape(testedataset_size,-1)


# print(data.shape)
# print(lables.shape)
# print(dataset_size)

# (trainX, testX, trainY, testY ) = train_test_split(data, lables, test_size= 0.25, random_state=42)
trainX = data
trainY = lables
testX = testdata
testY = testlables
model = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
model.fit(trainX, trainY)
print(classification_report(testY, model.predict(testX), target_names=le.classes_))
# print(classification_report(testY, model.predict(testX), labels=[0, 1]))


from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score

acc = accuracy_score(testY, model.predict(testX))
print('Accuracy: {0}'.format(acc))

reca = recall_score(testY, model.predict(testX), average='micro')
print('Recall: {0}'.format(reca))

f1_micro = f1_score(testY, model.predict(testX), average='micro')
print('F1_micro: {0}'.format(f1_micro))


#
# from sklearn.neural_network import MLPClassifier
#
# modela = MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, alpha=1e-4,
#                       solver='sgd', tol=1e-4, random_state=1,
#                       learning_rate_init=.1)
# modela.fit(trainX, trainY)
# print(classification_report(testY, modela.predict(testX), target_names=le.classes_))
#
# from sklearn.svm import SVC
#
# modelb = SVC(max_iter=1000,class_weight='balanced')
# modelb.fit(trainX, trainY)
# print(classification_report(testY, modelb.predict(testX), target_names=le.classes_))
