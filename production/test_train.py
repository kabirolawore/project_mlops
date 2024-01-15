import unittest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from train import load_data, split_data, preprocess, build_model, assess_model
from sklearn.svm import SVC

class TestModelTrain(unittest.TestCase):

    def setUp(self):
        self.test_x=[]
        self.test_y = ['LAYING', 'SITTING', 'STANDING', 'WALKING', 'WALKING_DOWNSTAIRS', 'WALKING_UPSTAIRS']
        
        input_file_path = 'har_dataset.csv'
        df = pd.read_csv(input_file_path)
        result_df = pd.DataFrame(columns=df.columns)

        # Extract a single row for each activity
        for activity in self.test_y:
            activity_row = df[df['Activity'] == activity].iloc[0]  # Extract the first row for each activity
            result_df = pd.concat([result_df, activity_row.to_frame().transpose()], ignore_index=True)

        self.test_Xx = result_df.drop(['Activity', 'subject'], axis=1)
        scaler = StandardScaler()
        X = scaler.fit_transform(self.test_Xx)
        
        for i in range(len(self.test_y)):
            self.test_x.append(X[i])


    def test_load_data(self):
        X, Y = load_data('har_dataset.csv')
        self.assertGreaterEqual(len(X), 7352)
        self.assertEqual(len(Y), len(X))

        self.assertEqual(len(self.test_x[0]), 561)


    def test_pre_process(self):
        scaled_x, _ = preprocess(self.test_Xx,self.test_y)
        self.assertTrue(np.allclose(np.mean(scaled_x, axis=0), 0))
        self.assertTrue(np.allclose(np.std(scaled_x, axis=0), 1))


    def test_split_data(self):
        X_train, X_test, Y_train, Y_test = split_data(self.test_x, self.test_y, 0.2, 100)
        self.assertEqual(len(X_train) + len(X_test), len(self.test_x))
        self.assertEqual(len(Y_train) + len(Y_test), len(self.test_y))

    def test_build_model(self):
        #Test for model of the correct type
        model = build_model(self.test_x, self.test_y)
        self.assertIsInstance(model, SVC)

    def test_assess_model(self):
        #Test accuracy and precision returns a value between 0 and 1
    
        model = build_model(self.test_x, self.test_y)
        accuracy, precision = assess_model(model, self.test_y, self.test_x)

        self.assertGreaterEqual(accuracy, 0)
        self.assertLessEqual(accuracy, 1)

        self.assertGreaterEqual(precision, 0)
        self.assertLessEqual(precision, 1)

if __name__ == '__main__':
    unittest.main()