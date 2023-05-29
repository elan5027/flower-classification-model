import numpy as np 
import pandas as pd

from PIL import ImageFile
from tqdm import tqdm
import h5py
import cv2

import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split


import tensorflow as tf

def image_to_tensor(img_path):
    img = tf.keras.preprocessing.image.load_img("flower_images/" + img_path, target_size=(128, 128))
    x = tf.keras.preprocessing.image.img_to_array(img)
    return np.expand_dims(x, axis=0)

def data_to_tensor(img_paths):
    list_of_tensors = [image_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

data = pd.read_csv("flower_images/flower_labels.csv")
files = data['file']
targets = data['label'].values
tensors = data_to_tensor(files)
data.head()

names = ['phlox', 'rose', 'calendula', 'iris', 'max chrysanthemum', 
         'bellflower', 'viola', 'rudbeckia laciniata', 'peony', 'aquilegia']

def display_images(img_path, ax):
    img = cv2.imread("flower_images/" + img_path)
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
fig = plt.figure(figsize=(20, 10))
for i in range(4):
    ax = fig.add_subplot(2, 4, i + 1, xticks=[], yticks=[])
    ax.set_title(names[targets[i+10]], color='yellow')
    display_images(files[i+10], ax)

with h5py.File('FlowerColorImages.h5', 'w') as f:
    f.create_dataset('images', data = tensors)
    f.create_dataset('labels', data = targets)
    f.close()
    
f = h5py.File('FlowerColorImages.h5', 'r')

# List all groups
keys = list(f.keys())   
print(keys)
tensors = np.array(f[keys[0]])
targets = np.array(f[keys[1]])
print ('Tensor shape:', tensors.shape)
print ('Target shape', targets.shape)
    
images_csv = tensors.reshape(210,128*128*3)
np.savetxt("flower_images.csv", images_csv, fmt='%i', delimiter=",")
data_images = pd.read_csv("flower_images.csv", header=None)
data_images.head()

data_images.iloc[:10,:10]
print(data_images.shape)
tensors = data_images.values
print(tensors.shape)
tensors = tensors.reshape(-1,128,128,3)
print(tensors.shape)
tensors = tensors.astype('float32')/255

targets = tf.keras.utils.to_categorical(targets, 10)

x_train, x_test, y_train, y_test = train_test_split(tensors, targets, 
                                                    test_size = 0.2, 
                                                    random_state = 1)
n = int(len(x_test)/2)
x_valid, y_valid = x_test[:n], y_test[:n]
x_test, y_test = x_test[n:], y_test[n:]

print('Label: ', names[np.argmax(y_train[7])])


def makemodel():
    model = tf.keras.models.Sequential()
    print(x_train.shape[1:])
    model.add(tf.keras.layers.Conv2D(128, (3, 3), input_shape=x_train.shape[1:]))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.02))
    
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))

    model.add(tf.keras.layers.Conv2D(128, (3, 3)))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.02))
    
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))

    model.add(tf.keras.layers.GlobalMaxPooling2D())
    
    model.add(tf.keras.layers.Dense(512))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.02))
    model.add(tf.keras.layers.Dropout(0.5)) 

    model.add(tf.keras.layers.Dense(10))
    model.add(tf.keras.layers.Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

model = makemodel()

checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath='weights.best.model.hdf5', verbose=2, save_best_only=True)

# To reduce learning rate dynamically
lr_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=5, verbose=2, factor=0.2)

model.summary()

history = model.fit(x_train, y_train, epochs=75, batch_size=32, verbose=2,
                    validation_data=(x_valid, y_valid),
                    callbacks=[checkpointer,lr_reduction])

data_generator = tf.keras.preprocessing.image.ImageDataGenerator(shear_range=0.3, 
                                                zoom_range=0.3,
                                                rotation_range=30,
                                                horizontal_flip=True)

dg_history = model.fit_generator(data_generator.flow(x_train, y_train, batch_size=64),
                                 steps_per_epoch = len(x_train)//64, epochs=7, verbose=2, 
                                 validation_data=(x_valid, y_valid),
                                 callbacks=[checkpointer,lr_reduction])

model.load_weights('weights.best.model.hdf5')
score = model.evaluate(x_test, y_test)
print(score)
score = model.evaluate(x_train, y_train)
print(score)
score = model.evaluate(x_valid, y_valid)
print(score)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

model.save('model.h5')



model1 = tf.keras.models.load_model('model.h5')
y_test_predict = model1.predict(x_test)
y_test_predict = np.argmax(y_test_predict,axis=1)
fig = plt.figure(figsize=(18, 18))
for i, idx in enumerate(np.random.choice(x_test.shape[0], size=16, replace=False)):
    ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(x_test[idx]))
    pred_idx = y_test_predict[idx]
    true_idx = np.argmax(y_test[idx])
    ax.set_title("{} ({})".format(names[pred_idx], names[true_idx]),
                 color=("#4876ff" if pred_idx == true_idx else "darkred"))
