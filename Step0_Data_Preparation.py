from gdal_tif2rgb import GRID, read_tif_and_get_bands
import os, math, cv2
import numpy as np
from skimage.segmentation import slic, mark_boundaries
import matplotlib.pyplot as plt
from astropy.visualization import PercentileInterval
import scipy.io as spio
import skimage.color as skcolor

interval = PercentileInterval(98.0)

from time import perf_counter

'''======> Configuration <======'''
superpixelBasedSamplingFlag = False  # True for sampling, false for testing SLIC parameters.
slicBand = [2, 1, 0]  # select 3 bands for SLIC from: 0-B, 1-G, 2-R, 3-NIR, 4-SWIR, 5-NDVI, 6-NDWI, 7-NDBI


landCoverList = ['Crop', 'Urban', 'Vegetation', 'Water']
print("\n\n---------------------- Step0: Data Preparation --------------------------")
print("Please check rootPath (data and code path): ")

rootPath = os.getcwd() + "\\"
print("rootPath: {}".format(rootPath))
print("-----------------------------------------------------------------------------")




def superpixel_based_patch_sampling(segments, Data, winSize, refMap):
    '''
    segments: slic segmentation map
    Data: Data to sample
    winSize: window size of pacth centering at a superpixel
    refMap: reference map
    '''
    if winSize % 2 != 0:
        os.error("Window Size is not odd.")
    if len(Data.shape) < 3:
        Data = Data[..., np.newaxis]

    rows, cols, numChannels = Data.shape[0], Data.shape[1], Data.shape[2]

    w = int(math.floor(winSize / 2))

    # Computing the Center of superpixels
    numSuperPixels = len(np.unique(segments))
    spLabels = np.zeros([numSuperPixels])

    spTrainSet = np.zeros([numSuperPixels, winSize, winSize, numChannels])

    # mirroredInput = np.zeros([numChannels, rows + 2*w, cols + 2*w])
    mirroredInput = cv2.copyMakeBorder(Data, w, w, w, w, cv2.BORDER_REFLECT)
    if len(mirroredInput.shape) < 3:
        mirroredInput = mirroredInput[..., np.newaxis]

    spLabelsList = list()
    for i in range(0, numSuperPixels):
        # i = 2000
        x, y = (np.where(segments == i))
        cx = math.ceil((x.min() + x.max()) / 2)
        cy = math.ceil((y.min() + y.max()) / 2)

        arr = refMap[x, y]
        tu = sorted([(np.sum(arr == i), i) for i in set(arr.flat)])
        maxCntEle = tu[-1][1]

        # spLabelsList.append(maxCntEle)
        spLabels[i] = maxCntEle
        # print("maxCntEle", maxCntEle)

        spTrainSet[i, ...] = mirroredInput[cx:cx + 2 * w + 1, cy:cy + 2 * w + 1, :]

        ### Save images
        savePath = dataPath + "{}\\".format(landCoverList[int(maxCntEle)])
        if not os.path.exists(savePath):
            os.mkdir(savePath)

        if i % 1000 == 0:
            # print("sp {}: ".format(i))
            print("save {} patch: ".format(i))
            used = perf_counter() - start
            print("Superpixel-based Sampling Time used: {:.2f} min".format(used / 60))

            # print(spTrainSet.shape, spTrainSet[i, :, :, visBand].shape)
            plt.imsave(savePath + "{}.png".format(int(i)), spTrainSet[i, :, :, visBand].transpose(1, 2, 0))

    # spTrainSet = spTrainSet.transpose([0, 3, 1, 2])
    return spTrainSet, spLabels
    # return 0, 0

dataPath = rootPath + "Samples_20m\\"
if not os.path.exists(dataPath):
    os.mkdir(dataPath)

dataName = "ChongQing_S2_2017_20m"
run = GRID()
proj, geotrans, _ = run.read_data(rootPath + dataName + ".tif")  # read data

''' ===========  Read Sentinel-2 Multispectral Data ============'''
inputData0, _ = read_tif_and_get_bands(rootPath, dataName, ['B', 'G', 'R', 'NIR', 'SWIR', \
                                                            'NDVI', 'NDBI', 'NDWI'])  # ['R', 'G', 'B']

''' =================  Read Land Cover Data ================='''
ref, _ = read_tif_and_get_bands(rootPath, "ChongQing_LC_2017_20m", ['landcover'])

for idx in range(0, inputData0.shape[0]):
    inputData0[idx, :, :] = interval(inputData0[idx, ...])
inputData = inputData0.transpose(1, 2, 0)
print(inputData.shape)

''' Choose 3 band for SLIC segmentation'''
# slicBand = [2, 1, 0]  # 8 bands (0-7): 0-B, 1-G, 2-R, 3-NIR, 4-SWIR, 5-NDVI, 6-NDWI, 7-NDBI
image = inputData[:, :, slicBand]


start = perf_counter()
print('SLIC superpixel segmentation ...')

compactness = 30
segments = slic(image, n_segments=30000, compactness=compactness)
plt.imsave(dataPath + "slic_comp_{}.png".format(compactness), mark_boundaries(image, segments, color=(0.8, 0, 0)))
spio.savemat(dataPath + "segments.mat", {'segments': segments})

used = perf_counter() - start
print("SLIC Used Time: {:.2f} min".format(used/60))



''' /////////////////    Superpixel-based Sampling     //////////////////// '''
# superpixelBasedSamplingFlag = True  # Change into True if suitable SLIC parameters have been chosen.
if superpixelBasedSamplingFlag:
    samples, labels = superpixel_based_patch_sampling(segments, inputData, 25, ref.squeeze())
    used = perf_counter() - start
    print("Superpixel-based Sampling Time used: {:.2f} min".format(used/60))

    print("sample shape: {}".format(samples.shape))
    spio.savemat(dataPath + "samples_labels.mat", {'samples': samples.astype(float), 'labels': labels.astype(float)})





