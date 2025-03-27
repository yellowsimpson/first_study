from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(512, activation = "relu"),
    layers.Dense(10, activation = "relu")    
])

model.compile(optimizer="rmsprop",
              loss = "sparse_cataegorical_crossentropy",
              metrics=["accuracy"]
              )

train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype("float32")/255

