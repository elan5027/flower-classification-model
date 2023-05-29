import tensorflow as tf
import pathlib
from tensorflow import keras
import numpy as np


data_dir = pathlib.Path('C:\\gitblog\\deep\\flower\\flowers') 
image_count = len(list(data_dir.glob('*/*.jpg')))

batch_size = 32
img_height = 180
img_width = 180

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=2023523,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

class_names = train_ds.class_names
#모델 불러오는 부분
print(class_names)
new_model = tf.keras.models.load_model('flowerclassification')
#테스트 파일 경로
test_path = pathlib.Path('C:\\gitblog\\deep\\image_00012.jpg')

img = keras.preprocessing.image.load_img(
    test_path, target_size=(img_height, img_width)
)

img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

predictions = new_model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "{}={:2f}%"
    .format(class_names[np.argmax(score)], 100*np.max(score))
)
