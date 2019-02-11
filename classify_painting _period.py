import os
import re
import tensorflow as tf
import tensorflow.python.platform
from tensorflow.python.platform import gfile
import numpy as np
import pandas as pd
import sklearn
from sklearn import model_selection
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC, LinearSVC
import matplotlib.pyplot as plt
import pickle


#Creates graph for the network
def create_graph():
    with gfile.FastGFile(os.path.join(
    model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

def mylistdir(directory):
    '''
    A specialized version of os.listdir() that ignores files that
    start with a leading period.
    '''
    
    filelist = os.listdir(directory)
    return [x for x in filelist
            if not (x.startswith('.'))]

def extract_features_labels(train_dir):
    '''
    The label is taken from the subfolder under train_dir.
    Name of the subfolder should be the year of painting.
    train_dir: directory containing the training images
    '''

    #Create list of all images in train_dir
    train_image_list = []
    for folder in mylistdir(train_dir):
        for file in mylistdir(os.path.join(train_dir,folder+'/')):
            if (file.endswith('.jpg')):
                train_image_list.append(
                    os.path.join(train_dir, folder, file))

    nb_features = 2048
    features = np.empty((len(train_image_list),nb_features))
    labels = []

    create_graph()

    with tf.Session() as sess:

        next_to_last_tensor = sess.graph.get_tensor_by_name('pool_3:0')

        for ind, image in enumerate(train_image_list):
            if (ind%100 == 0):
                print('Processing %s...' % (image))
            if not gfile.Exists(image):
                tf.logging.fatal('File does not exist %s', image)

            image_data = gfile.FastGFile(image, 'rb').read()
            predictions = sess.run(
                next_to_last_tensor,{'DecodeJpeg/contents:0': image_data})
            features[ind,:] = np.squeeze(predictions)
            labels.append(re.split("/", image.split('Period/')[1])[0])

    pickle.dump(features, open('painting_features', 'wb'))
    pickle.dump(labels, open('painting_labels', 'wb'))
    return features, labels

def plot_confusion_matrix(y_true,y_pred):
    cm_array = confusion_matrix(y_true,y_pred)
    true_labels = np.unique(y_true)
    pred_labels = np.unique(y_pred)
    plt.imshow(cm_array[:-1,:-1], interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix", fontsize=16)
    cbar = plt.colorbar(fraction=0.046, pad=0.04)
    cbar.set_label('Number of images', rotation=270, labelpad=30, fontsize=12)
    xtick_marks = np.arange(len(true_labels))
    ytick_marks = np.arange(len(pred_labels))
    plt.xticks(xtick_marks, true_labels, rotation=90)
    plt.yticks(ytick_marks,pred_labels)
    plt.tight_layout()
    plt.ylabel('True label', fontsize=14)
    plt.xlabel('Predicted label', fontsize=14)
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 12
    fig_size[1] = 12
    plt.rcParams["figure.figsize"] = fig_size

def train_model(feature_name,label_name):
    '''
    Trains the model using saved feature and label.
    Saves the model for use in later classification.
    feature_name: name of the saved feature file to be used
    label_name: name of the saved label file to be used
    '''

    #Load features and labels
    if type(feature_name) is str:
        features = pickle.load(open(feature_name))
        labels = pickle.load(open(label_name))
    else:
        features = feature_name
        labels = label_name

    #Use 20% of training images for testing
    X_train, X_test, y_train, y_test = (
        model_selection.train_test_split(
            features, labels, test_size=0.2, random_state=42))

    #Train the model
    clf = LinearSVC(
        C=1.0, loss='squared_hinge', penalty='l2',multi_class='ovr')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    #Print accuracy and plot confusion matrix
    print("Accuracy: {0:0.1f}%".format(accuracy_score(
        y_test,y_pred)*100))
    plot_confusion_matrix(y_test,y_pred)
    plt.show()

    #Save the model
    pickle.dump(clf, open('classify_painting_model.sav', 'wb'))
    print("Model saved as \"classify_painting_model.sav\".")


def extract_features(test_image_list):
    '''
    Extracts features from an unknown image. This function
    is used almost exclusively by the function classify_monet.
    test_image_list: list created from all the images in test_dir
    '''

    nb_features = 2048
    features = np.empty((len(test_image_list),nb_features))

    create_graph()

    with tf.Session() as sess:

        next_to_last_tensor = sess.graph.get_tensor_by_name('pool_3:0')

        for ind, image in enumerate(test_image_list):
            if (ind%100 == 0):
                print('Processing %s...' % (image))
            if not gfile.Exists(image):
                tf.logging.fatal('File does not exist %s', image)

            image_data = gfile.FastGFile(image, 'rb').read()
            predictions = sess.run(
                next_to_last_tensor,{'DecodeJpeg/contents:0': image_data})
            features[ind,:] = np.squeeze(predictions)

    return features

def classify_painting(model):
    '''
    Predicts the years of unknown images in test_dir and plots
    the images labeled with the predicted years
    If the year is known, name the image as "year_number.jpg"
    so that it can be labeled as actual year in the plot. If
    the year is not known, name the image as "unknown_number.jpg"
    model: name of the saved model with file extension ".sav" to
    be used in the prediction (e.g. 'Model.sav')
    '''

    #Make a list of images in the given directory and extract features
    test_image_list = [test_dir+f for f in os.listdir(
        test_dir) if re.search('jpg|JPG', f)]

    extract_features(test_image_list)

    #Load the trained model
    loaded_model = pickle.load(open(model, 'rb'))
    features = extract_features(test_image_list)

    #Predict the year
    year = loaded_model.predict(features)

    #Print the image names along with predicted years
    for num, image in enumerate(year):
        print year[num]

    #Plot the images labeled with predicted/actual year
    fig = plt.figure()
    for num, image in enumerate(test_image_list):
        y = fig.add_subplot(3,4,num+1)
        figure = plt.imread(image)
        y.imshow(figure)
        plt.title(year[num] + "/" + re.split(
            "test/", image.split('_')[0])[1])
        y.axes.get_xaxis().set_visible(False)
        y.axes.get_yaxis().set_visible(False)
    plt.show()


#contains the pre-trained Inception model
model_dir = '/Users/Alice/Programs/Python/Inception/' 
#directory of training images with subfolders
train_dir = os.path.join(os.path.dirname(__file__), "Period/")
#directory of unknown images
test_dir = os.path.join(os.path.dirname(__file__), "test/")

features, labels = extract_features_labels(train_dir)
train_model(features, labels)
#classify_painting('classify_painting_model.sav')


