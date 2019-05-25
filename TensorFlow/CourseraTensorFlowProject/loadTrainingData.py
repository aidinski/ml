
import tensorflow as tf
print(tf.__version__)

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('acc')>0.99):
      print("\nReached 99% accuracy so cancelling training!")
      self.model.stop_training = True


mnist = tf.keras.datasets.fashion_mnist
callbacks = myCallback()


(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
#You'll notice that all of the values in the number are between 0 and 255. If we are training a neural network, for various reasons it's easier if we treat all values as between 0 and 1, 
#a process called 'normalizing'...and fortunately in Python it's easy to normalize a list like this without looping. You do it like this:
training_images, test_images = (training_images / 255.0), (test_images / 255.0)

#to plot the data deriven from pictures uncomment the next 4 lines
#import matplotlib.pyplot as plt
#plt.imshow(training_images[0])
#print(training_labels[1])
#print(training_images[1])


model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)), 
                                    tf.keras.layers.Dense(512, activation=tf.nn.relu), 
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])


model.compile(optimizer='adam',
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=10, callbacks=[callbacks])

model.evaluate(test_images, test_labels)

