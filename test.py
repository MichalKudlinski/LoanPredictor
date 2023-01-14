import unittest

from ml_project import (X, Y, data_frame, data_frame1, test_dataframe,
                        train_dataframe)


class TestDataProvider(unittest.TestCase):
    """Testing DataProvider class"""
    def test_data_frame_different_number_of_rows(self):
        self.assertNotEqual(data_frame.shape[0],data_frame1.shape[0])
    def test_data_train_data_frame_has_loan_status(self):
        self.assertIn('Loan_Status', data_frame.columns)
    def test_data_test_frame_doesnt_have_loan_status(self):
        self.assertNotIn('Loan_Status', data_frame1.columns)


class TestDataAfterManipulation(unittest.TestCase):
    """Testing Data in dataframes after manipulating """
    def test_nulls_in_dataframe(self):
        self.assertEqual(data_frame.isnull().values.any(), False)
    def test_if_loan_id_in_dataframes(self):
        self.assertNotIn('Loan_ID',data_frame1)
        self.assertNotIn('Loan_ID',data_frame)
    def test_if_all_dtypes_numeric(self):
        for el in train_dataframe.dtypes:
            self.assertIn(el,['int64','float64'])
        for el in train_dataframe.dtypes:
            self.assertIn(el,['float64', 'int64'])


class TestModelTrainer(unittest.TestCase):
    """testing the ModelTrainer class"""
    def test_training_prep(self):
        self.assertNotIn('Loan_Status',X.columns)
        self.assertEqual('Loan_Status',Y.name)
    




        
        