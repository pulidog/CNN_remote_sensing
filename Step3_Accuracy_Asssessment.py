import numpy as np
import scipy.io as spio
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, f1_score
import matplotlib.pyplot as plt
from gdal_tif2rgb import read_tif_and_get_bands
import os

rootPath = os.getcwd() + "\\"
dataPath = rootPath + "Samples_20m\\"

print("------ Step 3: Accuracy Assessment --------")
print("rootPath: {}".format(rootPath))
print("dataPath: {}".format(dataPath))

''' Assign Color '''
def toRGB(z):
    rgb = np.zeros((z.shape[0], z.shape[1], 3))
    rgb[np.where(z == 0)] = [255, 248, 18]  # 0: urban
    rgb[np.where(z == 1)] = [255, 119, 88] # 1: crop
    rgb[np.where(z == 2)] = [11, 131, 20] # 2: vegetation
    rgb[np.where(z == 3)] = [31, 59, 255] # 3: water
    return rgb / 255

''' Read Pixel-based Reference Map '''
pixelRefMap, _ = read_tif_and_get_bands(rootPath, 'ChongQing_LC_2017_20m.tif', ['landcover'])
plt.imsave(dataPath + "pixel_refMap.png", toRGB(pixelRefMap.squeeze()))


''' Read Superixel-based Reference Map '''
ChongQing0 = spio.loadmat(dataPath + "samples_labels.mat")
# images0 = ChongQing0['samples'][..., [0, 1, 2]]
spRefLabel = ChongQing0['labels'].transpose().squeeze()

segments = spio.loadmat(dataPath + "segments.mat")
segments = segments['segments']
spRefMap = segments
for spLabel in np.unique(spRefLabel):
    a = np.where(spRefLabel == spLabel)[0]
    spRefMap[np.isin(segments, a)] = spLabel
plt.imsave(dataPath + "superpixel_refMap.png", toRGB(spRefMap))



''' Read Superpixel map predicted by CNN '''
spPredLabels = spio.loadmat(dataPath + "predLabels.mat")
spPredLabels = spPredLabels['predLabels'].squeeze()

print(spRefLabel.shape)
print(spPredLabels.shape)


''' Accuracy Assessment '''
confMatrix = confusion_matrix(spPredLabels, spRefLabel)
acc_score = accuracy_score(spPredLabels, spRefLabel)
F1_scoreNone = f1_score(spRefLabel, spPredLabels, average=None)
F1_score = f1_score(spRefLabel, spPredLabels, average='weighted')


print("-------- Confusion Matrix --------")
print("  urban   crop   veg  water ")
print(confMatrix)
print("----------------------------------")
print("accuracy Score: {:.2f}".format(acc_score))
print("F1-score 4-class: ", F1_scoreNone)
print("F1-score : {:.2f}".format(F1_score))


