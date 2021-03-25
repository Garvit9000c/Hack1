from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def Mark1(path,label,N):
  image_array=[]
  label_array=[]
  for i in range(1,N+1):
    img= Image.open(path+'/'+str(i)+'.jpg')
    np_img = np.array(img)
    if np_img.shape==(100,100):
      pass
    else:
      black=np_img
    image_array+=[black]
    label_array+=[label]
  return np.array(image_array),np.array(label_array)

test_yimg,test_ylab=Mark1('/content/drive/MyDrive/Data_set/Test/Yes',1,7)
test_nimg,test_nlab=Mark1('/content/drive/MyDrive/Data_set/Test/No',0,5)
test_images=np.concatenate((test_yimg, test_nimg), axis=0)
test_labels=np.concatenate((test_ylab, test_nlab), axis=0)

train_yimg,train_ylab=Mark1('./Data_set/Train/Yes',1,127)
train_nimg,train_nlab=Mark1('./Data_set/Train/No',0,71)
train_images=np.concatenate((train_yimg, train_nimg), axis=0)
train_labels=np.concatenate((train_ylab, train_nlab), axis=0)

train_images = train_images / 255.0
test_images = test_images / 255.0
class_names = ['Normal', 'Tumor']

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(100, (3, 3), activation='relu', input_shape=(100, 100, 3)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(126, activation='relu'))
model.add(tf.keras.layers.Dense(2))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=20)
tf.keras.models.save_model(model,'./Data_set/Tumor_CNN.hdf5')
