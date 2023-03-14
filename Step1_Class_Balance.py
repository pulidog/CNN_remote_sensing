
import numpy as np
import scipy.io as spio
import os


rootPath = os.getcwd() + "\\"
dataPath = rootPath + "Samples_20m\\"


print("------ Step 1: Class Balance --------")
print("rootPath: {}".format(rootPath))
print("dataPath: {}".format(dataPath))



data = spio.loadmat(dataPath + "samples_labels.mat")
samples = data['samples']
labels = data['labels'].squeeze()

print("sample size: {}".format(samples.shape))
print("Labels size: {}".format(labels.shape))

print("=========================================================================")
print("0: {}".format(len(np.where(labels == 0)[0])))
print("1: {}".format(len(np.where(labels == 1)[0])))
print("2: {}".format(len(np.where(labels == 2)[0])))
print("3: {}".format(len(np.where(labels == 3)[0])))


''' ===================> Randomly Select 1000 samples per class <================= '''

def classBalanceSample(classLabel, numOfSamplesPerClass):
    classImages = samples[np.where(labels == classLabel)[0], ...]
    classLabels = labels[np.where(labels == classLabel)[0]]

    idx = np.random.randint(0, len(classLabels), numOfSamplesPerClass)
    classImages = classImages[idx, ...]
    classLabels = classLabels[idx]

    return classImages, classLabels

numOfSamplesPerClass = 1000 # Number of samples in each class to balance different classes
cropImages, cropLabels = classBalanceSample(0, numOfSamplesPerClass)
builtImages, builtLabels = classBalanceSample(1, numOfSamplesPerClass)
vegImages, vegLabels = classBalanceSample(2, numOfSamplesPerClass)
waterImages, waterLabels = classBalanceSample(3, numOfSamplesPerClass)

mrgSamples = np.concatenate((cropImages, builtImages, vegImages, waterImages), axis=0)
mrgLabels = np.concatenate((cropLabels, builtLabels, vegLabels, waterLabels), axis=0)

spio.savemat(dataPath + "Samples_classBalanced_{}.mat".format(numOfSamplesPerClass), {'images': mrgSamples, 'labels': mrgLabels})