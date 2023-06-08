import csv
import random
import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn.metrics as sk
from sklearn.metrics import confusion_matrix,accuracy_score,f1_score
from tensorflow.keras.models import Sequential
from keras import regularizers
from keras.layers import Dense, Input, Dropout
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler, EarlyStopping
from keras.optimizers import SGD, RMSprop, Adam, Adagrad, Adadelta, Nadam

RANDOM_SEED = 42
NUM_CLASSES = 4

dataset = 'model/keypoint_classifier/keypoint.csv'
model_save_path = 'model/keypoint_classifier/keypoint_classifier.hdf5'
tflite_save_path = 'model/keypoint_classifier/keypoint_classifier.tflite'

train_set = 'C:\\Users\\Ana\\Downloads\\keypoints_train.csv'
val_set = 'C:\\Users\\Ana\\Downloads\\keypoints_val.csv'
test_set = 'C:\\Users\\Ana\\Downloads\\keypoints_test.csv'

X_train = np.loadtxt(train_set, delimiter=',', dtype='float32', usecols=list(range(1, (21 * 2) + 1)))
y_train = np.loadtxt(train_set, delimiter=',', dtype='int32', usecols=(0))
X_val = np.loadtxt(val_set, delimiter=',', dtype='float32', usecols=list(range(1, (21 * 2) + 1)))
y_val = np.loadtxt(val_set, delimiter=',', dtype='int32', usecols=(0))
X_test = np.loadtxt(test_set, delimiter=',', dtype='float32', usecols=list(range(1, (21 * 2) + 1)))
y_test = np.loadtxt(test_set, delimiter=',', dtype='int32', usecols=(0))

def confusionmat(y, y_hat):
    y_pred = np.argmax(y_hat, axis=1)  
    cm = confusion_matrix(y, y_pred)
    accu = accuracy_score(y, y_pred)

    # Create a heatmap of the confusion matrix
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", square=True)

    # Set labels, title, and accuracy
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.text(0, -0.5, "Accuracy: {:.2f}".format(accu), fontsize=12, ha="center")

    # Set x-axis and y-axis tick labels
    classes = np.unique(y)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    # Show the plot
    plt.show()

def learningcurve(history):
    print(history.history.keys())
    # summarize history for accuracy
    plt.figure(figsize=(5, 4))
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.figure(figsize=(5, 4))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def evaluation_function(model, X_test, y_test, history, epoch):
    y_pred = model.predict(X_test)
    y_pred_labels = np.argmax(y_pred, axis=1)
    test_results = model.evaluate(X_test, y_test)
    print("For epoch = {0}, the model test accuracy is {1}.".format(epoch,test_results[1]))
    f1 = f1_score(y_test, y_pred_labels, average='weighted')
    print("F1 score:", f1)
    confusionmat(y_test,y_pred)
    learningcurve(history)

def neural_network(nodes, activation_functions, optimizer, callbacks, epochs, model_save_path):
    model = Sequential()
    model.add(Input((21 * 2, )))
    for node, func in zip(nodes, activation_functions):
        model.add(Dense(node,activation=func))
    model.add(Dense(NUM_CLASSES,activation="softmax"))
    model.compile(optimizer = optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    history=model.fit(X_train, 
                      y_train, 
                      batch_size = 128, 
                      epochs=epochs,
                      validation_data=(X_val, y_val), 
                      verbose=0, 
                      callbacks=callbacks)
    
    model.save(model_save_path)
    evaluation_function(model, X_test, y_test, history, epochs)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    model_save_path, verbose=0, save_weights_only=False)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.001)
callbacks_cp_plateau = [reduce_lr, cp_callback]

neural_network(nodes=[100, 10],
               activation_functions=["relu", "relu"],
               optimizer="adam",
               callbacks=callbacks_cp_plateau,
               epochs=200,
               model_save_path=model_save_path
              )