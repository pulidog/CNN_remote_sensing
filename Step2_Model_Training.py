import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import scipy.io as spio
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time
from tensorboardX import SummaryWriter
import datetime, os

writer = SummaryWriter()

''' ========> Configuration <========'''
band4train = [2, 1, 0] # select [1,8] bands used for training, 0-7: 0-B, 1-G, 2-R, 3-NIR, 4-SWIR, 5-NDVI, 6-NDWI, 7-NDBI
band4projector = [7, 5, 6] # select 3 bands used for projector, at most 3 bands: 0-B, 1-G, 2-R, 3-NIR, 4-SWIR, 5-NDVI, 6-NDWI, 7-NDBI
applyModelFlag = True # True for applying model over the whole study area, false for not applying


rootPath = os.getcwd() + "\\"
dataPath = rootPath + "Samples_20m\\"

print("------ Step 2: Model Training --------")
print("rootPath: {}".format(rootPath))
print("dataPath: {}".format(dataPath))

''' ///////////////////////////////// Read All Samples and Labels /////////////////////////////// '''
# band4train = [7, 5, 6] # bands used for training, 0-7: 0-B, 1-G, 2-R, 3-NIR, 4-SWIR, 5-NDVI, 6-NDWI, 7-NDBI
ChongQing0 = spio.loadmat(dataPath + "samples_labels.mat")
images0 = ChongQing0['samples'][..., band4train]
labels0 = ChongQing0['labels'].transpose().squeeze()
dataset0 = tf.data.Dataset.from_tensor_slices((images0, labels0)).batch(3000)

''' ////////////////////////////// Read Class-Balanced Sample Set /////////////////////////////// '''
ChongQing = spio.loadmat(dataPath + "Samples_classBalanced_1000.mat")

images = ChongQing['images'][..., band4train]
labels = ChongQing['labels'].transpose().squeeze()

print("images shape: {}".format(images.shape))
print("labels shape: {}".format(labels.shape))

''' //////////////////////// Original Data Embedding ////////////////////////////////// '''
# band4projector = [7, 5, 6] # bands used for projector, at most 3 bands: 0-B, 1-G, 2-R, 3-NIR, 4-SWIR, 5-NDVI, 6-NDWI, 7-NDBI
subImg = ChongQing['images'][..., band4projector].transpose(0, 3, 1, 2)
print("subImg shape: ", subImg.shape)
n, c, w, h = subImg.shape

# featImg = subImg.reshape([-1, c*w*h])
# print(images.max())
# writer.add_embedding(featImg, metadata=labels, label_img=subImg)




images = images
labels = labels.astype(np.int32)

print("image range: [{}, {}]".format(images.min(), images.max()))

# Split Balanced Sample Set into training and Validation Set
train_x, val_x, train_y, val_y = train_test_split(images, labels, test_size=0.3)
print("Val: {}-->{}".format(val_x.shape, val_y.shape))

# Balanced Samples Set
dataset = tf.data.Dataset.from_tensor_slices((images, labels)).batch(128)

train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y)).shuffle(20).batch(128)
val_ds = tf.data.Dataset.from_tensor_slices((val_x, val_y)).shuffle(20).batch(128)

""" //////////////////////// CNN Model ///////////////////////////////////"""
model = tf.keras.Sequential([
    layers.Conv2D(64, (4, 4), activation='relu', input_shape=(25, 25, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(4, activation='softmax')
])

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy']
              )

log_dir="runs\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

history = model.fit(train_ds,
                    epochs=20,
                    validation_data=val_ds,
                    callbacks=[tensorboard_callback])


dict = history.history.keys()
print("keys: ", dict)

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.savefig(dataPath + "learning.png")




if applyModelFlag:
    ''' ///////////////////// Apply Model /////////////////////// '''
    segments = spio.loadmat(dataPath + "segments.mat")
    segments = segments['segments']
    print("seg [{}, {}]".format(segments.min(), segments.max()))
    pred = model.predict(dataset0)

    print(pred.shape)
    predLabels = np.argmax(pred, axis=1)

    print(np.unique(predLabels))
    print("0: {}".format(len(np.where(predLabels == 0)[0])))
    print("1: {}".format(len(np.where(predLabels == 1)[0])))
    print("2: {}".format(len(np.where(predLabels == 2)[0])))
    print("3: {}".format(len(np.where(predLabels == 3)[0])))

    '''////////////// Transform superpixel labels into a classified map //////////////'''
    start = time.perf_counter()
    predMap = segments
    for spLabel in np.unique(predLabels):
        a = np.where(predLabels == spLabel)[0]
        predMap[np.isin(segments, a)] = spLabel


    print("predLabels: ", predLabels.shape)


    spio.savemat(dataPath + 'predLabels.mat', {'predLabels': predLabels})
    spio.savemat(dataPath + 'predMap.mat', {'predMap': predMap})

    ''' Assign Color '''
    def toRGB(z):
        rgb = np.zeros((z.shape[0], z.shape[1], 3))
        rgb[np.where(z == 0)] = [255, 248, 18]  # 0: urban
        rgb[np.where(z == 1)] = [255, 119, 88] # 1: crop
        rgb[np.where(z == 2)] = [11, 131, 20] # 2: vegetation
        rgb[np.where(z == 3)] = [31, 59, 255] # 3: water
        return rgb / 255

    predMapRGB = toRGB(predMap)

    plt.imsave(dataPath + "predMap_rgb.png", predMapRGB)

    used = time.perf_counter() - start
    print("pred size: {}, {} min".format(pred.shape, used/60))

''' ////////////// Visualized the learned Features ///////////////'''
model.pop()
feature = model.predict(dataset)
print("feature shape to Projector: ", feature.shape)
writer.add_embedding(feature, metadata=labels, label_img=subImg)
writer.close()
