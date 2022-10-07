import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras import Input
from tensorflow.keras.optimizers import Adam

def generator(img_shape, num_resnet):

    model = Sequential()

    # generates tensors with a normal distribution
    initialiser = RandomNormal(mean = 0.0, stddev = 0.05)

    input_image = Input(shape = img_shape)

    model.add(Conv2D(64, (7, 7), padding = "same", kernel_initializer = initialiser, input_shape = img_shape))
    model.add(InstanceNormalization(axis = -1))
    model.add(Activation("relu"))

    model.add(Conv2D(128, (3, 3), strides = (2, 2), padding = "same", kernel_initializer = initialiser))
    model.add(InstanceNormalization(axis = -1))
    model.add(Activation("relu"))

    model.add(Conv2D(256, (3, 3), strides = (2, 2), padding = "same", kernel_initializer = initialiser))
    model.add(InstanceNormalization(axis = -1))
    model.add(Activation("relu"))

    for _ in range(num_resnet):
        resnet(256, model)

def resnet(num_filters, input_layer):
    pass

def discriminator(shape):
    pass

def model(generator1, discriminator, generator2, shape):
    pass
