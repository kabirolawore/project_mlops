import mlflow
import argparse
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def load_data(file):
    # Read the csv file
    df = pd.read_csv(file)
    # input columns
    X = df.drop(['Activity', 'subject'], axis=1)
    # output columns
    Y = df['Activity'].values
    # return input X & output Y
    return X, Y

def preprocess(X, Y):
    # Feature scaling
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    Y = Y
    return X, Y

# Split the dataset to train and test
def split_data(X, Y, percentage, rand_state):
    # split the data using train_test_split
    return train_test_split(X, Y, test_size=percentage, random_state=rand_state)

def build_model(X, Y):
    # create a model with "rbf" kernel and C 100
    model = SVC(kernel='rbf',C=100.0)
    model.fit(X, Y)
    # return the model
    return model

def assess_model(model, Y, X):
    y_predict = model.predict(X)
    accuracy = accuracy_score(Y, y_predict)
    precision = precision_score(Y, y_predict, average='weighted', zero_division=1)
    return accuracy, precision

def train_model(data, model_save_path):
    x, y = load_data(data)
    X, Y = preprocess(x, y)
    X_train, X_test, Y_train, Y_test = split_data(X, Y, 0.2, 100)
    print('Length of train', len(X_train))
    print('Length of test', len(X_test))
    
    model = build_model(X_train, Y_train)
    accuracy, precision = assess_model(model, Y_test, X_test)
    print("Accuracy", accuracy)
    print("Precision", precision)

    if model_save_path:
        print("Saving model to", model_save_path)
        mlflow.sklearn.save_model(model, model_save_path)
    return model, X_train, Y_train

if __name__ == "__main__":
	argparser = argparse.ArgumentParser()
	argparser.add_argument('--trainingdata', required=True, help='load data file')
	argparser.add_argument('--model', default=False, help='save model file')
	args = argparser.parse_args()
	
	mlflow.autolog()
	train_model(args.trainingdata, args.model)
