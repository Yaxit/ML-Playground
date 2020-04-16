import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.regularizers import l2
import numpy as np
import matplotlib.pyplot as plt

print("TF Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], train_images.shape[1], train_images.shape[2], 1)
test_images = test_images.reshape(test_images.shape[0], test_images.shape[1], test_images.shape[2], 1)

class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print("Train shape:\t", train_images.shape)
print("Train labels:\t", len(train_labels))

print("Test shape:\t", test_images.shape)
print("Test labels:\t", len(test_labels))

# normalize pixel values
train_images = train_images / 255.0
test_images = test_images / 255.0

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i].reshape(28,28))
    plt.xlabel(class_names[train_labels[i]])
plt.show()

model = keras.Sequential()
# Conv2D expects the image dimension and the channel number (1, 3 for RGB). To use Conv2D on grayscale
# we need to reshape the images from (x,y) to (x,y,1)
model.add(keras.layers.Conv2D(filters= 32, padding='same', kernel_size=(3,3), activation='relu', kernel_regularizer=l2(0.0005), input_shape=(28,28,1)))
model.add(keras.layers.Conv2D(filters= 64, padding='same', kernel_size=(3,3), activation='relu', kernel_regularizer=l2(0.0005)))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Conv2D(filters=64, padding='same', kernel_size=(3,3), activation='relu', kernel_regularizer=l2(0.0005)))
model.add(keras.layers.Conv2D(filters=128, padding='same', kernel_size=(3,3), activation='relu', kernel_regularizer=l2(0.0005)))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(10))
model.add(keras.layers.Softmax())


# another way to generate the model
# model = keras.Sequential([
#     keras.layers.Conv2D(filters= 8, kernel_size=(5,5), activation='relu', input_shape=(28,28)),
#     keras.layers.Conv2D(filters=16, kernel_size=(5,5), activation='relu'),
#     keras.layers.Conv2D(filters=32, kernel_size=(5,5), activation='relu'),
#     keras.layers.Flatten(),
#     keras.layers.Dense(10),
#     keras.layers.Softmax()
# ])


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()
model.fit(train_images, train_labels, epochs=10)

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

predictions_array = model.predict(test_images)

def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array, true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img.reshape(28,28))

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array, true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

  # Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 5
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions_array[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions_array[i], test_labels)
plt.tight_layout()
plt.show()
