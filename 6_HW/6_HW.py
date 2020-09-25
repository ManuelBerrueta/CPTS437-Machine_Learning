from matplotlib import pyplot
from keras.datasets import mnist
import keras
import numpy
import tensorflow as tf
import colorama
import random
from termcolor import colored, cprint

(X_train, y_train), (X_test, y_test) = mnist.load_data()
# Check data
print(f"X_train = {X_train.shape} |  y_train = {y_train.shape}")
print(f"X_test  = {X_test.shape} |  y_test = {y_test.shape}")

class_names = [ '0', '1', '2', '3', '4', '5', '6', '7', '8', '9' ]

X_train = X_train / 255.0
X_test = X_test / 255.0

pyplot.figure(figsize=(10,10))

for each_i in range(25):
    pyplot.subplot(5, 5, each_i + 1)
    pyplot.xticks([])
    pyplot.yticks([])
    pyplot.grid(False)
    pyplot.imshow(X_train[each_i], cmap=pyplot.cm.binary)
    pyplot.xlabel(class_names[y_train[each_i]])
#pyplot.show()

# Reshape images for compatibility with convolutional layer
X_train = numpy.reshape(X_train, (60000, 28, 28, 1))
X_test = numpy.reshape(X_test, (10000, 28, 28, 1))


# Build, train, and evaluate model
def evaluate_model(model, epochs=10):
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print ('num', len(y_train))
    hist = model.fit(X_train, y_train, epochs=epochs)
    #train_acc = hist.history['acc'][-1]
    train_acc = hist.history['accuracy'][-1]
    test_loss, test_acc = model.evaluate(X_test, y_test)
    model.summary()
    print('Epochs: ' + str(epochs))
    print('Training accuracy: ' + str(train_acc))
    print('Testing accuracy: ' + str(test_acc))


def model1(epochs=10):
    print("-=-=-=-=-=[ Model 1: ]=-=-=-=-=-") # Dense, Dense
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28, 1)), #flattened image input
        keras.layers.Dense(units=32, activation=tf.nn.relu),
        keras.layers.Dense(units=32, activation=tf.nn.relu),
        keras.layers.Dense(units=32, activation=tf.nn.relu),
        keras.layers.Dense(units=10, activation=tf.nn.softmax) #10-node dense softmax output
        ])
    evaluate_model(model, epochs)


#Doubled the number of layers
def model2(epochs=10):
    print("-=-=-=-=-=[ Model 2: ]=-=-=-=-=-") # Dense, Dense
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28, 1)), #flattened image input
        keras.layers.Dense(units=32, activation=tf.nn.relu),
        keras.layers.Dense(units=32, activation=tf.nn.relu),
        keras.layers.Dense(units=32, activation=tf.nn.relu),
        keras.layers.Dense(units=32, activation=tf.nn.relu),
        keras.layers.Dense(units=32, activation=tf.nn.relu),
        keras.layers.Dense(units=32, activation=tf.nn.relu),
        keras.layers.Dense(units=20, activation=tf.nn.softmax) #10-node dense softmax output
        ])
    evaluate_model(model, epochs)


def model3(epochs=10):
    print("-=-=-=-=-=[ Model 3: ]=-=-=-=-=-") # Dense, Dense
    model = keras.Sequential([
        keras.layers.Conv2D(300, (8,8), activation='relu', input_shape=(28, 28, 1)),
        keras.layers.MaxPooling2D(pool_size=(2,2)),
        keras.layers.Conv2D(150, (6,6), activation='relu', input_shape=(28, 28, 1)),
        keras.layers.MaxPooling2D(pool_size=(2,2)),
        keras.layers.Flatten(),
        keras.layers.Dense(units=50, activation=tf.nn.softmax) #10-node dense softmax output
        ])
    evaluate_model(model, epochs)


def model4(epochs=10):
    print("-=-=-=-=-=[ Model 4: ]=-=-=-=-=-") # Dense, Dense
    model = keras.Sequential([
        keras.layers.Conv2D(300, (8,8), activation='relu', input_shape=(28, 28, 1)),
        keras.layers.MaxPooling2D(pool_size=(2,2)),
        keras.layers.Conv2D(150, (6,6), activation='relu', input_shape=(28, 28, 1)),
        keras.layers.MaxPooling2D(pool_size=(2,2)),
        keras.layers.Flatten(input_shape=(28, 28, 1)), #flattened image input
        keras.layers.Dense(units=32, activation=tf.nn.relu),
        keras.layers.Dense(units=16, activation=tf.nn.relu),
        keras.layers.Dense(units=8, activation=tf.nn.relu),
        keras.layers.Dense(units=20, activation=tf.nn.softmax) #10-node dense softmax output
        ])
    evaluate_model(model, epochs)


if __name__ == "__main__":
    model1()
    model2()
    model3()
    model4()

    thank_you = "Thank you for an awesome semester and all the help provide!\n"
    colors = list(vars(colorama.Fore).values())
    colored_chars = [random.choice(colors) + char for char in thank_you]
    print(''.join(colored_chars))