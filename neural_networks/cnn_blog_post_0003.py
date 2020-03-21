import tensorflow as tf
import math
from IPython.display import clear_output
import matplotlib.pyplot as plt

from keras.layers import Flatten, Conv2D, MaxPool2D, Dense, Dropout
from keras.callbacks import LearningRateScheduler,TensorBoard

layers = tf.keras.layers
Conv2D = tf.keras.layers.Conv2D
Flatten = tf.keras.layers.Flatten
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
MaxPooling2D = tf.keras.layers.MaxPool2D
TensorBoard = tf.keras.callbacks.TensorBoard
LearningRateScheduler = tf.keras.callbacks.LearningRateScheduler
regularizers = tf.keras.regularizers

from time import time
tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
name = ''

def step_decay(epoch):


    initial_lrate=0.1
    drop=0.6
    epochs_drop = 3.0
    lrate= initial_lrate * math.pow(drop,
                                    math.floor((1+epoch)/epochs_drop))
    return lrate

lrate = LearningRateScheduler(step_decay)

def display_activation(activations, col_size, row_size, act_index):
    activation = activations[act_index]
    activation_index=0
    fig, ax = plt.subplots(row_size, col_size, figsize=(row_size*2.5,col_size*1.5))
    for row in range(0,row_size):
        for col in range(0,col_size):
            ax[row][col].imshow(activation[0, :, :, activation_index], cmap='gray')
            activation_index += 1
    plt.savefig('activation' + str(act_index) + ".png")


class PlotLearning(tf.keras.callbacks.Callback):


    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        self.fig = plt.figure()

        self.logs = []

    def on_epoch_end(self, epoch, logs={}):

        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('accuracy'))
        self.i += 1
        f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)

        clear_output(wait=True)

        ax1.set_yscale('log')
        ax1.plot(self.x, self.losses, label="loss")
        ax1.legend()

        ax2.plot(self.x, self.acc, label="accuracy")
        ax2.legend()
        plt.savefig(name + '_growth_data' + str(self.i) + '.png', format='png')
        plt.show()

plot = PlotLearning()

def neural_network_model(number_sequence = [], input_shape = (0, 0 ,0)):

    """

    Arguments:
        number_sequence (List): List of the numbers for the building of the network.
        input_shape (Tuple): Shape of the data (Ex. 28 x 28 dimensions for the pixel.

    Returns:
        model (Object): Model of the convolutional neural network with the nodes representative of a sequence of numbers.

    """

    # Importing the required Keras modules containing model and layers
    Sequential = tf.keras.Sequential

    # Creating a Sequential Model and adding the layers
    model = Sequential()

    # First Layers
    model.add(Conv2D(number_sequence[0], kernel_size=(3,3), input_shape=input_shape, activation='relu', kernel_regularizer=regularizers.l2(0.01),))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    # Second Layers
    model.add(Conv2D(number_sequence[1], kernel_size=(3,3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    # Second Layers
    model.add(Conv2D(number_sequence[2], kernel_size=(3,3), padding='same',activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    # Second Layers
    model.add(Conv2D(number_sequence[3], kernel_size=(3,3), padding='same',activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Dense(number_sequence[4], activation='relu', kernel_regularizer=regularizers.l2(0.01)) )
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(10,activation='softmax'))

    learning_rate = 0.0001
    optimzer = tf.keras.optimizers.Adam(lr=learning_rate, decay=1e-6)

    model.compile(optimizer=optimzer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

if __name__ == '__main__':


    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # Reshaping the array to 4-dims so that it can work with the Keras API
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)

    # # Making sure that the values are float so that we can get decimal points after division
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    #
    # # Normalizing the RGB codes by dividing it to the max RGB value.
    x_train /= 255
    x_test /= 255


    ram_growth = [32, 64, 128, 256, 512]
    fibonacci_sequence = [34, 55, 89, 144, 233]
    catalan_number_sequence = [42, 132, 429, 1430, 4862]
    lucas_numbers = [29, 47, 76, 123, 199]
    prime_numbers = [31, 37, 41, 43, 47]
    #
    accuracy_output = []
    prediction_output = []


    name = 'ram'
    model = neural_network_model(number_sequence=ram_growth, input_shape=input_shape)
    history = model.fit(x=x_train,y=y_train, epochs=5, callbacks=[tensorboard, plot, lrate], verbose=3)
    output = model.evaluate(x_test, y_test)

    accuracy_output.append(output[1])
    pred = model.predict(x_test[4444].reshape(1, 28, 28, 1))
    prediction_output.append(pred)
    name = 'fibonacci'
    model = neural_network_model(number_sequence=fibonacci_sequence, input_shape=input_shape)
    history = model.fit(x=x_train,y=y_train, epochs=10, callbacks=[plot])
    output = model.evaluate(x_test, y_test)
    accuracy_output.append(output[1])
    pred = model.predict(x_test[4444].reshape(1, 28, 28, 1))
    prediction_output.append(pred)
    name = 'catalan'
    model = neural_network_model(number_sequence=catalan_number_sequence, input_shape=input_shape)
    history = model.fit(x=x_train,y=y_train, epochs=10, callbacks=[plot])
    output = model.evaluate(x_test, y_test)
    accuracy_output.append(output[1])
    pred = model.predict(x_test[4444].reshape(1, 28, 28, 1))
    prediction_output.append(pred)
    name = 'lucas'
    model = neural_network_model(number_sequence=lucas_numbers, input_shape=input_shape)
    history = model.fit(x=x_train,y=y_train, epochs=10, callbacks=[plot])
    output = model.evaluate(x_test, y_test)
    accuracy_output.append(output[1])
    pred = model.predict(x_test[4444].reshape(1, 28, 28, 1))
    prediction_output.append(pred)
    name = 'prime'
    model = neural_network_model(number_sequence=prime_numbers, input_shape=input_shape)
    history = model.fit(x=x_train,y=y_train, epochs=10, callbacks=[plot])
    output = model.evaluate(x_test, y_test)
    accuracy_output.append(output[1])

    pred = model.predict(x_test[4444].reshape(1, 28, 28, 1))
    prediction_output.append(pred)
    print(accuracy_output)
    print (prediction_output)

