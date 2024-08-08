import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from scripts.train import load_data, preprocess_data, train_and_log_model


class TestTrain(unittest.TestCase):
    @patch("scripts.train.pd.read_csv")
    def test_load_data(self, mock_read_csv):
        # Mock the data that pd.read_csv would return
        mock_df = pd.DataFrame(
            {
                "date": ["2021-01-01"],
                "fips": [1001],
                "PS": [0.1],
                "PRECTOT": [0.1],
                "QV2M": [0.1],
                "T2M": [0.1],
                "T2MDEW": [0.1],
                "T2M_MAX": [0.1],
                "T2M_MIN": [0.1],
                "T2M_RANGE": [0.1],
                "WS10M": [0.1],
                "WS10MDEW": [0.1],
                "WS10M_MAX": [0.1],
                "WS10M_MIN": [0.1],
                "WS10M_RANGE": [0.1],
                "WS50M": [0.1],
                "WS50MDEW": [0.1],
                "WS50M_MAX": [0.1],
                "WS50M_MIN": [0.1],
                "WS50M_RANGE": [0.1],
                "TS": [0.1],
                "T2MWET": [0.1],
            }
        )

        mock_read_csv.return_value = mock_df

        train_data, test_data, val_data = load_data()

        self.assertEqual(len(train_data), 1)
        self.assertEqual(len(test_data), 1)
        self.assertEqual(len(val_data), 1)

    @patch("scripts.train.DictVectorizer")
    def test_preprocess_data(self, mock_dict_vectorizer):
        # Mock the DictVectorizer

        # Create a simple dataframe for testing
        mock_df = pd.DataFrame(
            {
                "date": ["2021-01-01"],
                "fips": [1001],
                "PRECTOT": [0.1],
                "PS": [0.1],
                "QV2M": [0.1],
                "T2M": [0.1],
                "T2MDEW": [0.1],
                "T2M_MAX": [0.1],
                "T2M_MIN": [0.1],
                "T2M_RANGE": [0.1],
                "WS10M": [0.1],
                "WS10MDEW": [0.1],
                "WS10M_MAX": [0.1],
                "WS10M_MIN": [0.1],
                "WS10M_RANGE": [0.1],
                "WS50M": [0.1],
                "WS50MDEW": [0.1],
                "WS50M_MAX": [0.1],
                "WS50M_MIN": [0.1],
                "WS50M_RANGE": [0.1],
                "TS": [0.1],
                "T2MWET": [0.1],
                "score": [5],
            }
        )
        mock_df_for_dv = pd.DataFrame(
            {
                "PRECTOT": [0.1],
                "PS": [0.1],
                "QV2M": [0.1],
                "T2M": [0.1],
                "T2MDEW": [0.1],
                "T2M_MAX": [0.1],
                "T2M_MIN": [0.1],
                "T2M_RANGE": [0.1],
                "WS10M": [0.1],
                "WS10MDEW": [0.1],
                "WS10M_MAX": [0.1],
                "WS10M_MIN": [0.1],
                "WS10M_RANGE": [0.1],
                "WS50M": [0.1],
                "WS50MDEW": [0.1],
                "WS50M_MAX": [0.1],
                "WS50M_MIN": [0.1],
                "WS50M_RANGE": [0.1],
                "year": [2004],
                "month": [1],
                "day": [1],
            }
        )

        mock_dv = MagicMock()
        return_value = mock_df_for_dv.to_dict(orient="records")
        mock_dv.fit_transform.return_value = return_value
        mock_dict_vectorizer.return_value = mock_dv

        X_train, y_train, X_val, y_val, X_test, y_test, dv = preprocess_data(
            mock_df, mock_df, mock_df
        )
        self.assertEqual(len(X_train[0]), 21)  # Based on the mock return value
        print(y_train)
        self.assertEqual(y_train.iloc[0], 20)  # Ensure the target column is correct

    @patch("scripts.train.mlflow.start_run")
    @patch("scripts.train.xgb.XGBClassifier")
    def test_train_and_log_model(self, mock_xgb_classifier, mock_start_run):
        # Mock the model and MLflow
        mock_model = MagicMock()
        mock_model.fit.return_value = None
        mock_xgb_classifier.return_value = mock_model
        mock_run = MagicMock()
        mock_start_run.return_value.__enter__.return_value = mock_run

        X_train, y_train, X_val, y_val, X_test, y_test = (
            [[1, 2]],
            [2],
            [[3, 4]],
            [2],
            [[5, 6]],
            [2],
        )
        dv = MagicMock()

        model_version = train_and_log_model(
            X_train, y_train, X_val, y_val, X_test, y_test, dv
        )

        mock_model.fit.assert_called_once()
        self.assertIsNotNone(
            model_version
        )  # This would be the model version returned from MLflow


if __name__ == "__main__":
    unittest.main()
