import numpy as np 
import pandas as pd

from PIL import ImageFile
from tqdm import tqdm
import h5py
import cv2
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

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
    
images_csv = tensors.reshape(210,128*128*3)
np.savetxt("flower_images.csv", images_csv, fmt='%i', delimiter=",")
data_images = pd.read_csv("flower_images.csv", header=None)
data_images.head()

data_images.iloc[:10,:10]
tensors = data_images.values
tensors = tensors.reshape(-1,128,128,3)
tensors = tensors.astype('float32')/255

targets = tf.keras.utils.to_categorical(targets, 10)

x_train, x_test, y_train, y_test = train_test_split(tensors, targets, 
                                                    test_size = 0.2, 
                                                    random_state = 1)
n = int(len(x_test)/2)
x_valid, y_valid = x_test[:n], y_test[:n]
x_test, y_test = x_test[n:], y_test[n:]

model = tf.keras.models.load_model('model.h5')
y_test_predict = model.predict(x_test)
y_test_predict = np.argmax(y_test_predict,axis=1)

fig = plt.figure(figsize=(18, 18))
for i, idx in enumerate(np.random.choice(x_test.shape[0], size=16, replace=False)):
    ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(x_test[idx]))
    pred_idx = y_test_predict[idx]
    true_idx = np.argmax(y_test[idx])
    ax.set_title("{} ({})".format(names[pred_idx], names[true_idx]),
                 color=("#4876ff" if pred_idx == true_idx else "darkred"))
    
y_test_orig = []
for i in y_test:
  y_test_orig.append(np.argmax(i)) 
y_test_orig = np.array(y_test_orig)
print(y_test_orig)
print(y_test_predict)
cnf = confusion_matrix(y_test_predict,y_test_orig)

df_cnf = pd.DataFrame(cnf, range(10), range(10))
plt.title("Confusion Matrix")
plt.show()