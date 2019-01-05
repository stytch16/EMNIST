# Mute tensorflow debugging information console
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.layers import MaxPooling2D, Convolution2D, Dropout, Dense, Flatten, BatchNormalization 
from keras.models import Sequential, save_model
import argparse
import keras
import numpy as np
import pandas as pd

IMG_SZ = 28

def load_data(train_file_path, test_file_path, verbose=False):
    ''' Load data in from .csv file as specified by the paper.

        Arguments:
            train_file_path: path to train data .csv, should be in sample/
            test_file_path: path to test data .csv, should be in sample/

        Optional Arguments:
            width: specified width
            height: specified height

        Returns:
            A tuple of training and test data and labels

    '''
    def flip_and_rotate(img):
        img = img.reshape([IMG_SZ, IMG_SZ])
        img = np.fliplr(img)
        img = np.rot90(img)
        return img.reshape([IMG_SZ*IMG_SZ])

    # Load data
    train = pd.read_csv(train_file_path, header=None)
    test = pd.read_csv(test_file_path, header=None)

    # Split data and labels
    train_data = train.iloc[:, 1:]
    train_labels = train.iloc[:, 0]
    test_data = test.iloc[:, 1:]
    test_labels = test.iloc[:, 0]

    # One hot encode labels
    train_labels = pd.get_dummies(train_labels)
    test_labels = pd.get_dummies(test_labels)

    # Get numpy array form
    train_data = train_data.values
    train_labels = train_labels.values
    test_data = test_data.values
    test_labels = test_labels.values

    # Apply flip and rotate transform on horizontal axis of data.
    train_data = np.apply_along_axis(flip_and_rotate, 1, train_data)
    test_data = np.apply_along_axis(flip_and_rotate, 1, test_data)

    train_data = train_data.reshape([-1, IMG_SZ, IMG_SZ, 1])
    test_data = test_data.reshape([-1, IMG_SZ, IMG_SZ, 1])

    # Normalize
    train_data = train_data/255
    test_data = test_data/255

    if verbose:
        print('Train dimensions: {}, {}'.format(train_data.shape, train_labels.shape))
        print('Test dimensions: {}, {}'.format(test_data.shape, test_labels.shape))

    return (train_data, train_labels), (test_data, test_labels)
   
def build_nn(verbose=False):
    ''' Build and train neural network. Also offloads the net in .yaml and the
        weights in .h5 to the bin/.

        Arguments:
            training_data: the packed tuple from load_data()

        Optional Arguments:
            width: specified width
            height: specified height
            epochs: the number of epochs to train over
            verbose: enable verbose printing
    '''
    input_shape = (IMG_SZ, IMG_SZ, 1)

    model = Sequential()
    model.add(Convolution2D(64,
                            (5, 5),
                            padding='same',
                            input_shape=input_shape,
                            activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(BatchNormalization())
    model.add(Convolution2D(128,
                            (2, 2),
                            padding='same',
                            activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(47, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    if verbose == True: print(model.summary())
    return model

def train_and_save(model, x_train, y_train, x_test, y_test, batch_size=256, epochs=10):

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.1)

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    # Offload model to file
    model_yaml = model.to_yaml()
    with open("bin/model.yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)
    save_model(model, 'bin/model.h5')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage='A training program for classifying the EMNIST dataset')
    parser.add_argument('-f', '--train_file', type=str, help='Path train .csv file data', required=True)
    parser.add_argument('-g', '--test_file', type=str, help='Path test .csv file data', required=True)
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train on')
    parser.add_argument('--verbose', action='store_true', default=False, help='Enables verbose printing')
    args = parser.parse_args()

    bin_dir = os.path.dirname(os.path.realpath(__file__)) + '/bin'
    if not os.path.exists(bin_dir):
        os.makedirs(bin_dir)

    (train_data, train_labels), (test_data, test_labels) = load_data(args.train_file, args.test_file, verbose=args.verbose)
    model = build_nn(verbose=args.verbose)
    train_and_save(model, train_data, train_labels, test_data, test_labels, epochs=args.epochs)
