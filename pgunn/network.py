"""
file contains implementation of several neural network models
"""
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv2D, Flatten, Dense, Concatenate, Lambda, Subtract, Add
from keras import backend as K
from keras import optimizers, losses
from keras.utils import plot_model
from tools import get_name, err_print

def mse_mae(y_true, y_pred):
    """
    loss function, which combines MSE and MAE
    """
    error = y_true - y_pred
    cond = K.abs(error) < 1.0
    MSE = K.pow(error, 2)
    MAE = K.abs(error)

    loss = tf.where(cond, MSE, MAE)

    return K.mean(loss)


class Network:
    """
    class implements several neural network constructor methods
    """
    def __init__(self, state_size, action_size, learning_rate, loss, plot_mdl=[True, False]):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        if loss == "MSE":
            self.loss = losses.mean_squared_error
        elif loss == "MSE_MAE":
            self.loss = mse_mae
        elif loss == "HUBER":
            self.loss = tf.losses.huber_loss
        else:
            err_print("[Model file doesn't exist.]")
            sys.exit(-1)
        self.plot_mdl = plot_mdl
        self.plotted = False


    def make_2layer_mdl(self, units):
        """
        method returns 2 layer neural network model
        """
        network_input = Input(shape=(self.state_size,))

        net = Dense(units=units[0], activation="relu", kernel_initializer="he_uniform")(network_input)
        net = Dense(units=units[1], activation="relu", kernel_initializer="he_uniform")(net)
        net = Dense(units=self.action_size, activation="linear", kernel_initializer="he_uniform")(net)

        model = Model(inputs=network_input, outputs=net)

        self.visualize_model(model)
        model.compile(loss=self.loss, optimizer=optimizers.Adam(lr=self.learning_rate), metrics=['accuracy'])

        return model


    def make_4layer_mdl(self, units):
        """
        method returns 2 layer neural network model
        """
        network_input = Input(shape=(self.state_size,))

        net = Dense(units=units[0], activation="relu", kernel_initializer="he_uniform")(network_input)
        net = Dense(units=units[1], activation="relu", kernel_initializer="he_uniform")(net)
        net = Dense(units=units[2], activation="relu", kernel_initializer="he_uniform")(net)
        net = Dense(units=units[3], activation="relu", kernel_initializer="he_uniform")(net)
        net = Dense(units=self.action_size, activation="linear", kernel_initializer="he_uniform")(net)

        model = Model(inputs=network_input, outputs=net)

        self.visualize_model(model)
        model.compile(loss=self.loss, optimizer=optimizers.Adam(lr=self.learning_rate), metrics=['accuracy'])

        return model


    def make_2layer_duel_mdl(self, units):
        """
        method returns 2 layer dueling neural network model
        """
        network_input = Input(shape=(self.state_size,))

        net = Dense(units=units[0], activation="relu", kernel_initializer="he_uniform")(network_input)
        net = Dense(units=units[1], activation="relu", kernel_initializer="he_uniform")(net)

        state_value = Dense(units=1, activation="linear", kernel_initializer="he_uniform")(net)
        value_function = Concatenate(axis=-1)([state_value, state_value])

        action_values = Dense(units=self.action_size, activation="linear", kernel_initializer="he_uniform")(net)
        avg_action = Lambda(lambda x: K.mean(x, axis=1, keepdims=True))(action_values)
        concat_avg_action = Concatenate(axis=-1)([avg_action, avg_action])

        for _ in range(self.action_size-2):
            value_function = Concatenate(axis=-1)([value_function, state_value])
            concat_avg_action = Concatenate(axis=-1)([concat_avg_action, avg_action])

        advantage_function = Subtract()([action_values, concat_avg_action])

        net = Add()([value_function, advantage_function])

        model = Model(inputs=network_input, outputs=net)

        self.visualize_model(model)
        model.compile(loss=self.loss, optimizer=optimizers.Adam(lr=self.learning_rate), metrics=["accuracy"])

        return model


    def make_bsc_img_mdl(self):
        """
        method returns DeepMind's neural network model
        """
        network_input = Input(shape=(self.state_size))

        net = Conv2D(filters=32, kernel_size=(8, 8), strides=(4, 4), activation="relu",
                     kernel_initializer="he_uniform", data_format="channels_first")(network_input)
        net = Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), activation="relu",
                     kernel_initializer="he_uniform")(net)
        net = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation="relu",
                     kernel_initializer="he_uniform")(net)
        net = Flatten()(net)

        net = Dense(units=512, activation="relu", kernel_initializer="he_uniform")(net)
        net = Dense(units=self.action_size, activation="linear", kernel_initializer="he_uniform")(net)

        model = Model(inputs=network_input, outputs=net)

        self.visualize_model(model)
        model.compile(loss=self.loss, optimizer=optimizers.Adam(lr=self.learning_rate), metrics=['accuracy'])

        return model


    def make_duel_img_mdl(self):
        """
        method returns DeepMind's dueling neural network model
        """
        network_input = Input(shape=(self.state_size))

        net = Conv2D(filters=32, kernel_size=(8, 8), strides=(4, 4), activation="relu",
                     kernel_initializer="he_uniform", data_format="channels_first")(network_input)
        net = Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), activation="relu",
                     kernel_initializer="he_uniform")(net)
        net = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation="relu",
                     kernel_initializer="he_uniform")(net)
        net = Flatten()(net)

        net = Dense(units=512, activation="relu", kernel_initializer="he_uniform")(net)

        state_value = Dense(units=1, activation="linear", kernel_initializer="he_uniform")(net)
        value_function = Concatenate(axis=-1)([state_value, state_value])

        action_values = Dense(units=self.action_size, activation="linear", kernel_initializer="he_uniform")(net)
        avg_action = Lambda(lambda x: K.mean(x, axis=1, keepdims=True))(action_values)
        concat_avg_action = Concatenate(axis=-1)([avg_action, avg_action])

        for _ in range(self.action_size-2):
            value_function = Concatenate(axis=-1)([value_function, state_value])
            concat_avg_action = Concatenate(axis=-1)([concat_avg_action, avg_action])

        advantage_function = Subtract()([action_values, concat_avg_action])

        net = Add()([value_function, advantage_function])

        model = Model(inputs=network_input, outputs=net)

        self.visualize_model(model)
        model.compile(loss=self.loss, optimizer=optimizers.Adam(lr=self.learning_rate), metrics=['accuracy'])

        return model


    def make_1layer_mdl(self, units):
        """
        method returns 1 layer neural network model
        """
        network_input = Input(shape=(self.state_size,))

        net = Dense(units=units[0], activation="relu", kernel_initializer="he_uniform")(network_input)
        net = Dense(units=self.action_size, activation="linear", kernel_initializer="he_uniform")(net)

        model = Model(inputs=network_input, outputs=net)

        self.visualize_model(model)
        model.compile(loss=self.loss, optimizer=optimizers.Adam(lr=self.learning_rate), metrics=['accuracy'])

        return model


    def visualize_model(self, model):
        """
        method prints model to stdout and pdf
        """
        if not self.plotted:
            if self.plot_mdl[0]:
                model.summary()
            if self.plot_mdl[1]:
                name = get_name("model")
                plot_model(model, to_file=name, show_shapes=True, show_layer_names=False)
                print("[PDF blueprint of neural network was saved.]")
            self.plotted = True
