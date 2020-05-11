
# Standard Python Modules
# -----------------------
import math
from IPython.display import clear_output
from vis.visualization import visualize_activation
from vis.utils import utils
import matplotlib.pyplot as plt

# Keras modules
# -------------
import keras
import keras.regularizers as regularizers
from keras.layers import Flatten, Conv2D, MaxPool2D, Dense, Dropout
from keras.callbacks import LearningRateScheduler, EarlyStopping
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator

# Standard configurations
# -----------------------
from wandb.keras import WandbCallback
# Inside my model training code
import wandb
wandb.init(project="visualizer_number_sequence")


# Config is a variable that holds and saves hyperparameters and inputs
config = wandb.config
config.learning_rate = 0.0001
config.epochs = 2
config.img_width=28
config.img_height=28
config.num_classes = 10
config.padding = 'same'
config.activation = 'relu'
config.padding = 'same'
config.optimizer = 'adam'
config.batch_size = 128

# Number Sequences Constants
# --------------------------
ram_growth = [32, 64, 128, 256, 512]
fibonacci_sequence = [34, 55, 89, 144, 233]
catalan_number_sequence = [42, 132, 429, 1430, 4862]
lucas_numbers = [29, 47, 76, 123, 199]
prime_numbers = [31, 37, 41, 43, 47]

def step_decay(epoch):


    initial_lrate=0.1
    drop=0.6
    epochs_drop = 3.0
    lrate= initial_lrate * math.pow(drop,
                                    math.floor((1+epoch)/epochs_drop))
    return lrate

lrate = LearningRateScheduler(step_decay)

class PlotLearning(keras.callbacks.Callback):


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

def plot_classes(model, number_sequence):

    """

    Creates a keras model visualization of its learning state at a certain epoch depending on the model trained.

    Arguments:
        model (Keras Model Object): The neural network of the keras CNN.
        number_sequence (String): Number sequence primary name.


    """

    # Numbers to visualize
    numbers_to_visualize = [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ]
    images_learned = []

    # Visualize
    for number_to_visualize in numbers_to_visualize:
        visualization = visualize_activation(model, layer_index, filter_indices=number_to_visualize)
        plt.imshow(visualization[..., 0])
        plt.title(f'MNIST target = {number_to_visualize}')
        plt.savefig(number_sequence + str(number_to_visualize) + ".png")
        images_learned.append(wandb.Image(plt))

    wandb.log({"Learning Visualization for" + number_sequence: images_learned})

def neural_network_model(number_sequence = None, input_shape = (0, 0 ,0)):

    """

    Arguments:
        number_sequence (List): List of the numbers for the building of the network.
        input_shape (Tuple): Shape of the data (Ex. 28 x 28 dimensions for the pixel.

    Returns:
        model (Object): Model of the convolutional neural network with the nodes representative of a sequence of numbers.

    """

    # Importing the required Keras modules containing model and layers
    from keras.models import Sequential

    # Creating a Sequential Model and adding the layers
    model = Sequential()

    # First Layers
    model.add(Conv2D(number_sequence[0], kernel_size=(3,3), input_shape=input_shape, activation=config.activation, kernel_regularizer=regularizers.l2(0.01),))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    # Second Layers
    model.add(Conv2D(number_sequence[1], kernel_size=(3,3), padding='same', activation=config.activation, kernel_regularizer=regularizers.l2(0.01)))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    # Second Layers
    model.add(Conv2D(number_sequence[2], kernel_size=(3,3), padding='same',activation=config.activation, kernel_regularizer=regularizers.l2(0.01)))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    # Second Layers
    model.add(Conv2D(number_sequence[3], kernel_size=(3,3), padding='same',activation=config.activation, kernel_regularizer=regularizers.l2(0.01)))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Dense(number_sequence[4], activation='relu', kernel_regularizer=regularizers.l2(0.01)) )
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(10,activation='softmax',  name='visualized_layer'))

    optimzer = keras.optimizers.Adam(lr=config.learning_rate, decay=1e-6)

    model.compile(optimizer=optimzer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def load_saved_model(name_model):

    """

    Load the saved model of a previous run

    Arguments:
        name_model (String): model we would like to load

    Returns:
        loaded_model (Keras Model Object): The loaded model with the appropiate saved weights

    """

    # load json and create model
    json_file = open('model_' + name_model + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights('model_' + name_model + '.h5')
    print("Loaded model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    return loaded_model

def save_model(model_to_save, file_name):

    """

    Arguments:

        model (Keras Model Object): The neural network model that you would like to be saved.
        file_name (String): The dedicated file name to be saved.

    """

    # serialize model to JSON
    model_json = model_to_save.to_json()
    with open("model_" + file_name + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model_to_save.save_weights("model" + file_name + ".h5")
    print("Saved model to disk")

def create_image_data_generator():

    """

    Returns:
        datagen (Keras ImageDataGenerator Object): An instatiated class from keras pre processing for an image.

    """

    # Data augmentation
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False # randomly flip images
    )

    return datagen

if __name__ == '__main__':

    # Param to use for testing.
    load_model = False

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    # Reshaping the array to 4-dims so that it can work with the Keras API
    x_train = x_train.reshape(x_train.shape[0], config.img_width, config.img_height, 1)
    x_test = x_test.reshape(x_test.shape[0], config.img_width, config.img_height, 1)
    input_shape = (config.img_width, config.img_height, 1)

    # # Making sure that the values are float so that we can get decimal points after division
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    #
    # # Normalizing the RGB codes by dividing it to the max RGB value.
    x_train /= 255
    x_test /= 255

    labels = [str(i) for i in range(0, 10)]

    accuracy_output = []
    prediction_output = []

    name = 'ram'
    if load_model:
        model = load_saved_model(name)
    else:
        model = neural_network_model(number_sequence=ram_growth, input_shape=input_shape)
        datagen = create_image_data_generator()
        datagen.fit(x_train)

        history = model.fit_generator(
                            datagen.flow(x_train, y_train, batch_size=config.batch_size),
                            epochs=config.epochs,
                            validation_data=(x_test, y_test),
                            verbose=3,
                            callbacks=[WandbCallback(data_type="image", labels=labels, validation_data=(x_test, y_test))])

    output = model.evaluate(x_test, y_test)
    layer_index = utils.find_layer_idx(model, 'visualized_layer')
    model.layers[layer_index].activation = keras.activations.linear
    model = utils.apply_modifications(model)

    plot_classes(model, name)

    accuracy_output.append(output[1])
    pred = model.predict(x_test[4444].reshape(1, config.img_width, config.img_height, 1))
    prediction_output.append(pred)

    save_model(model, name)
    #
    #
    # name = 'fibonacci'
    # model = neural_network_model(number_sequence=fibonacci_sequence, input_shape=input_shape)
    # history = model.fit(x=x_train,y=y_train, epochs=epochs,verbose=3)
    # output = model.evaluate(x_test, y_test)
    # layer_index = utils.find_layer_idx(model, 'visualized_layer')
    # model.layers[layer_index].activation = keras.activations.linear
    # model = utils.apply_modifications(model)
    # plot_classes(model, name)
    # accuracy_output.append(output[1])
    # pred = model.predict(x_test[4444].reshape(1, 28, 28, 1))
    # prediction_output.append(pred)
    #
    # name = 'catalan'
    # model = neural_network_model(number_sequence=catalan_number_sequence, input_shape=input_shape)
    # history = model.fit(x=x_train,y=y_train, epochs=epochs,verbose=3)
    # output = model.evaluate(x_test, y_test)
    # layer_index = utils.find_layer_idx(model, 'visualized_layer')
    # model.layers[layer_index].activation = keras.activations.linear
    # model = utils.apply_modifications(model)
    # plot_classes(model, name)
    # accuracy_output.append(output[1])
    # pred = model.predict(x_test[4444].reshape(1, 28, 28, 1))
    # prediction_output.append(pred)
    #
    # name = 'lucas'
    # model = neural_network_model(number_sequence=lucas_numbers, input_shape=input_shape)
    # history = model.fit(x=x_train,y=y_train, epochs=epochs,verbose=3)
    # output = model.evaluate(x_test, y_test)
    # layer_index = utils.find_layer_idx(model, 'visualized_layer')
    # model.layers[layer_index].activation = keras.activations.linear
    # model = utils.apply_modifications(model)
    # plot_classes(model, name)
    # accuracy_output.append(output[1])
    # pred = model.predict(x_test[4444].reshape(1, 28, 28, 1))
    # prediction_output.append(pred)
    #
    # name = 'prime'
    # model = neural_network_model(number_sequence=prime_numbers, input_shape=input_shape)
    # history = model.fit(x=x_train,y=y_train, epochs=epochs,verbose=3)
    # output = model.evaluate(x_test, y_test)
    # layer_index = utils.find_layer_idx(model, 'visualized_layer')
    # model.layers[layer_index].activation = keras.activations.linear
    # model = utils.apply_modifications(model)
    # plot_classes(model, name)
    # accuracy_output.append(output[1])
    # pred = model.predict(x_test[4444].reshape(1, 28, 28, 1))
    # prediction_output.append(pred)

    print(accuracy_output)
    print (prediction_output)

