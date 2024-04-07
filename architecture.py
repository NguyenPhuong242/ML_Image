#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score


"""
@author: cecile capponi, AMU
L3 Informatique, 2023/24
"""

"""
Computes a representation of an image from the (gif, png, jpg...) file 
-> representation can be (to extend) 
'HC': color histogram
'PX': tensor of pixels
'GC': matrix of gray pixels 
other to be defined
--
input = an image (jpg, png, gif)
output = a new representation of the image
"""
def raw_image_to_representation(img, representation_type='HC'):
  if representation_type == 'HC':
    representation = compute_color_histogram(img)
  elif representation_type == 'PX':
    representation = img.flatten()
  elif representation_type == 'GC':
    representation = compute_gray_matrix(img)
  else:
    representation = None
    print(f"Representation type '{representation_type}' not implemented.")

  return representation

def compute_color_histogram(img):
  img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  color_histogram = cv2.calcHist([img_rgb], [0, 1, 2], None, [256, 256, 256],
                                 [0, 256, 0, 256, 0, 256])
  color_histogram = color_histogram.flatten()
  return color_histogram


def compute_gray_matrix(img):
  gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  gray_matrix = gray_img.flatten()
  return gray_matrix

"""
Returns a relevant structure embedding train images described according to the 
specified representation and associate each image (name or/and location) to its label.
-> Representation can be (to extend) 
'HC': color histogram
'PX': tensor of pixels 
'GC': matrix of gray pixels
other to be defined
--
input = where are the examples, which representation of the data must be produced ? 
output = a relevant structure (to be discussed, see below) where all the images of the
directory have been transformed, named and labelled according to the directory they are
stored in (the structure lists all the images, each image is made up of 3 informations,
namely its name, its representation and its label)
This structure will later be used to learn a model (function learn_model_from_dataset)
-- uses function raw_image_to_representation
"""
def load_transform_label_train_dataset(directory, representation, target_size=(512,512)):
  dataset_structure = []

  for label in os.listdir(directory):
      label_path = os.path.join(directory, label)
      if os.path.isdir(label_path):
        for image_name in os.listdir(label_path):
            image_path = os.path.join(label_path, image_name)

            #print(image_path)
            img = cv2.imread(image_path)
            img = preprocess_image(img, target_size)

            # Compute representation
            feature = raw_image_to_representation(img, representation)
            
            #print(feature.shape)
            # Add to dataset structure
            entry = {
                'name': image_name,
                'representation': feature,
                'label': label
            }
            dataset_structure.append(entry)

  return dataset_structure


"""
Returns a relevant structure embedding test images described according to the 
specified representation.
-> Representation can be (to extend) 
'HC': color histogram
'PX': tensor of pixels 
'GC': matrix of gray pixels 
other to be defined
--
input = where are the data, which represenation of the data must be produced ? 
output = a relevant structure, preferably the same chosen for function load_transform_label_train_data
-- uses function raw_image_to_representation
-- must be consistant with function load_transform_label_train_dataset
-- while be used later in the project
"""
def load_transform_test_dataset(directory, representation, target_size=(512, 512)):
    dataset_structure = []
    label_path = os.path.basename(directory)

    for image_name in os.listdir(directory):
        image_path = os.path.join(directory, image_name)

        #print(image_path)
        img = cv2.imread(image_path)

        if img is not None:
            img = preprocess_image(img, target_size)

  
            feature = raw_image_to_representation(img, representation)

            #print(feature.shape)
          
            entry = {
                'name': image_name,
                'representation': feature,
                'label': None 
            }
            dataset_structure.append(entry)
        else:
            print(f"Failed to read image: {image_path}")

    return dataset_structure

"""
Preprocesses an image to a given target size
--
input = an image, the target size
output = the preprocessed image

"""
def preprocess_image(img, target_size):
  h, w, _ = img.shape

  if h != target_size[0] or w != target_size[1]:
    preprocessed_img = cv2.resize(img,
                                  target_size,
                                  interpolation=cv2.INTER_AREA)
  else:
    preprocessed_img = img

  return preprocessed_img


"""
Learn a model (function) from a pre-computed representation of the dataset, using the algorithm 
and its hyper-parameters described in algo_dico
For example, algo_dico could be { algo: 'decision tree', max_depth: 5, min_samples_split: 3 } 
or { algo: 'multinomial naive bayes', force_alpha: True }
--
input = transformed labelled dataset, the used learning algo and its hyper-parameters (better a dico)
output =  a model fit with data
"""
def learn_model_from_dataset(train_dataset, algo_dico):
  print("Learning model from dataset")
  print("--------------------------")
    # Extract data and labels from the dataset structure
  data = np.array(
      [entry['representation'].flatten() for entry in train_dataset])
  labels = np.array([entry['label'] for entry in train_dataset])

  if len(data) == 0:
        print("Error: The dataset is empty.")
        return None

  # Split the dataset into training and testing sets
  X_train, X_test, y_train, y_test = train_test_split(data,
                                                      labels,
                                                      test_size=0.2,
                                                      random_state=42)

  # Select the learning algorithm based on algo_dico
  algo = algo_dico.get('algo', 'decision tree')
  if algo.lower() == 'decision tree':
    model = DecisionTreeClassifier(max_depth=algo_dico.get('max_depth', None),
                                   min_samples_split=algo_dico.get(
                                       'min_samples_split', 2))
  elif algo.lower() == 'multinomial naive bayes':
    model = MultinomialNB(
        alpha=1.0 if algo_dico.get('force_alpha', False) else 0.01)
  elif algo.lower() == 'random forest':
    model = RandomForestClassifier(
        n_estimators=algo_dico.get('n_estimators', 100),
        max_depth=algo_dico.get('max_depth', None),
        min_samples_split=algo_dico.get('min_samples_split', 2))
  elif algo.lower() == 'svm':
    model = svm.SVC(
        C=algo_dico.get('C', 1.0),
        kernel=algo_dico.get('kernel', 'poly'),
        degree=algo_dico.get('degree', 3),
        gamma=algo_dico.get('gamma', 'scale'))
  elif algo.lower() == 'knn':
        model = KNeighborsClassifier(
            n_neighbors=algo_dico.get('n_neighbors', 5),
            weights=algo_dico.get('weights', 'uniform'),
            algorithm=algo_dico.get('algorithm', 'auto'))
  else:
    raise ValueError(f"Unknown algorithm: {algo}")

  print(X_train.shape, y_train.shape)
  model.fit(X_train, y_train)

  accuracy = model.score(X_test, y_test)
  #print(f"Model Accuracy: {accuracy * 100:.2f}% on the test set")

  return model

"""
Given one example (previously loaded with its name and representation),
computes its class according to a previously learned model.
--
input = representation of one data, the learned model
output = the label of that one data (+1 or -1)
-- uses the model learned by function learn_model_from_dataset
"""
def predict_example_label(example, model):
    img = cv2.imread(example)
    img = preprocess_image(img, (512, 512))
    # Compute representation
    feature = raw_image_to_representation(img, 'GC')
    example = feature.flatten()

    return model.predict([example])[0]


"""
Computes a structure that computes and stores the label of each example of the dataset, 
using a previously learned model. 
--
input = a structure embedding all transformed data to a representation, and a model
output =  a structure that associates a label to each identified data (image) of the input dataset
"""
def predict_sample_label(dataset, model):
    predictions = []
    for entry in dataset:
        feature = entry['representation']
        predicted_label = model.predict([feature])[0]
        predictions.append({'name': entry['name'], 'predicted_label': predicted_label})
    return predictions

"""
Save the predictions on dataset to a text file with syntax:
image_name <space> label (either -1 or 1)  
NO ACCENT  
In order to be perfect, the first lines of this file should indicate the details
of the learning methods used, with its hyper-parameters (in order to do so, the signature of
the function must be changed, as well as the signatures of some previous functions in order for 
these details to be transmitted along the pipeline. 
--
input = where to save the predictions, structure embedding the dataset
output =  OK if the file has been saved, not OK if not
"""
def write_predictions(directory, filename, predictions, algo_dico):
    try:
        with open(os.path.join(directory, filename), 'w') as file:
            countMers = 0
            countAilleurs = 0
            for prediction in predictions:
                if(prediction['predicted_label'] == 'Mer'):
                   countMers = countMers + 1
                   #file.write("{} {} {}\n".format(prediction['name'], prediction['predicted_label'], "+1"))
                   file.write("{} {}\n".format(prediction['name'], "+1"))

                elif(prediction['predicted_label'] == 'Ailleurs'):
                  countAilleurs = countAilleurs + 1
                  #file.write("{} {} {}\n".format(prediction['name'], prediction['predicted_label'], "-1"))
                  file.write("{} {}\n".format(prediction['name'], "-1"))
                else:
                   continue
            
            file.write("\n")
            #file.write("Mer: {} , Ailleur: {}".format(countMers, countAilleurs))
        return "Done."
    except Exception as e:
        return f"Error: {e}"

"""
Estimates the accuracy of a previously learned model using train data, 
either through CV or mean hold-out, with k folds.
input = the train labelled data as previously structured, the type of model to be learned
(as in function learn_model_from_data), and the number of split to be used either 
in a hold-out or by cross-validation 
output =  The score of success (betwwen 0 and 1, the higher the better, scores under 0.5
are worst than random guess)
"""
def estimate_model_score(train_dataset, algo_dico, k):
    # Extract data and labels from the dataset structure
    data = np.array(
        [entry['representation'].flatten() for entry in train_dataset])
    labels = np.array([entry['label'] for entry in train_dataset])

    # Select the learning algorithm based on algo_dico
    algo = algo_dico.get('algo', 'decision tree')
    if algo.lower() == 'decision tree':
        model = DecisionTreeClassifier(max_depth=algo_dico.get('max_depth', None),
                                    min_samples_split=algo_dico.get(
                                        'min_samples_split', 2))
    elif algo.lower() == 'multinomial naive bayes':
        model = MultinomialNB(
            alpha=1.0 if algo_dico.get('force_alpha', False) else 0.01)
    elif algo.lower() == 'random forest':
        model = RandomForestClassifier(
            n_estimators=algo_dico.get('n_estimators', 100),
            max_depth=algo_dico.get('max_depth', None),
            min_samples_split=algo_dico.get('min_samples_split', 2))
    elif algo.lower() == 'svm':
        model = svm.SVC(
            C=algo_dico.get('C', 1.0),
            kernel=algo_dico.get('kernel', 'poly'),
            degree=algo_dico.get('degree', 3),
            gamma=algo_dico.get('gamma', 'scale'))
    elif algo.lower() == 'knn':
        model = KNeighborsClassifier(
            n_neighbors=algo_dico.get('n_neighbors', 5),
            weights=algo_dico.get('weights', 'uniform'),
            algorithm=algo_dico.get('algorithm', 'auto'))
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    scores = cross_val_score(model, data, labels, cv=k)
    return scores.mean()


def generate_predictions_for_algorithms(data_path, test_path, repr_type, folder_path, list_of_algo):
    for idx, algo_dico in enumerate(list_of_algo, start=1):
        print(f"Generating predictions for Algorithm {idx}")
        
        model_train = load_transform_label_train_dataset(data_path, repr_type)

        test_data = load_transform_test_dataset(test_path, repr_type)

        trained_model = learn_model_from_dataset(model_train, algo_dico)

        predictions = predict_sample_label(test_data, trained_model)

        filename = f"predictions_algo_{idx}.txt"

        result = write_predictions(folder_path, filename, predictions, algo_dico)

        #print(estimate_model_score(model_train, algo_dico, 5))

        print(result)


def result(prediction_files):
    image_votes = {}

    # Read all prediction files
    for file_path in prediction_files:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 2:
                    image_name, vote = parts
                    if image_name in image_votes:
                        image_votes[image_name].append(int(vote))
                    else:
                        image_votes[image_name] = [int(vote)]

    with open("PK.txt", 'w') as pk_file:
        for image_name, votes in image_votes.items():

            count_mer = votes.count(1)
            count_ailleur = votes.count(-1)

            if count_mer > count_ailleur:
                final_vote = "+1"
            elif count_ailleur > count_mer:
                final_vote = "-1"
            else:
                final_vote = "0"  


            pk_file.write(f"{image_name} {final_vote}\n")

#Count -1 and +1 in a file and print the result
def count_votes(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        count_mer = 0
        count_ailleur = 0
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 2:
                image_name, vote = parts
                if vote == "+1":
                    count_mer += 1
                elif vote == "-1":
                    count_ailleur += 1
        print(f"Mer: {count_mer}, Ailleur: {count_ailleur}")
    

def run():
    print("Running")

    folder_path = ''
    data_path = 'Data/'
    test_path = 'AllTest/'
    repr_type = 'GC'

    algo_dico1 = {'algo': 'decision tree', 'max_depth': 5, 'min_samples_split': 3}
    algo_dico2 = {'algo': 'multinomial naive bayes', 'force_alpha': True}
    algo_dico3 = {'algo': 'random forest', 'n_estimators': 100, 'max_depth': None, 'min_samples_split': 2}
    algo_dico4 = {'algo': 'svm', 'C': 1.0, 'kernel': 'poly', 'degree': 3, 'gamma': 'scale'}
    algo_dico5 = {'algo': 'knn', 'n_neighbors': 5, 'weights': 'uniform', 'algorithm': 'auto'}

    list_of_algo = [algo_dico1, algo_dico2, algo_dico3, algo_dico4, algo_dico5]

    generate_predictions_for_algorithms(data_path, test_path, repr_type, folder_path, list_of_algo)

    list_of_predictions = ["predictions_algo_1.txt", "predictions_algo_2.txt", "predictions_algo_3.txt", "predictions_algo_4.txt", "predictions_algo_5.txt"]

    result(list_of_predictions)

    count_votes("PK.txt")


    # model_train = load_transform_label_train_dataset(data_path, repr_type)

    # test_train = load_transform_test_dataset(test_path, repr_type)

    # train_model = learn_model_from_dataset(model_train, algo_dico5)

    # print("Estimate_model_score: ",estimate_model_score(model_train, algo_dico4, 5))

    # resultOfExample = predict_example_label('Test/1.jpeg', train_model)

    # print(resultOfExample)

    # predictions = predict_sample_label(test_train, train_model)

    # result = write_predictions(folder_path, 'predictions.txt', predictions, algo_dico5)

    # print(result)

run()



