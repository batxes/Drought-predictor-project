import unittest
import pandas as pd
from scripts.predict import preprocess_input_data, load_model_and_dv
from unittest.mock import patch, MagicMock


class TestPredict(unittest.TestCase):
    @patch("scripts.predict.pickle.load")
    @patch("scripts.predict.open")
    def test_load_model_and_dv(self, mock_open, mock_pickle_load):
        # Arrange
        mock_model = MagicMock()
        mock_dv = MagicMock()
        mock_pickle_load.side_effect = [mock_model, mock_dv]

        # Act
        model, dv = load_model_and_dv("fake_model_path", "fake_dv_path")

        # Assert
        mock_open.assert_any_call("fake_model_path", "rb")
        mock_open.assert_any_call("fake_dv_path", "rb")
        self.assertEqual(model, mock_model)
        self.assertEqual(dv, mock_dv)

    def test_preprocess_input_data(self):
        # Create a simple dataframe for testing
        df = pd.DataFrame(
            {
                "date": ["2021-01-01"],
                "fips": [1001],
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
                "PS": [0.1],
                "TS": [0.1],
                "T2MWET": [0.1],
            }
        )

        columns = [
            "PRECTOT",
            "PS",
            "QV2M",
            "T2M",
            "T2MDEW",
            "T2M_MAX",
            "T2M_MIN",
            "T2M_RANGE",
            "WS10M",
            "WS10M_MAX",
            "WS10M_MIN",
            "WS10M_RANGE",
            "WS50M",
            "WS50M_MAX",
            "WS50M_MIN",
            "WS50M_RANGE",
            "year",
            "month",
            "day",
        ].to_dict(orient="records")

        mock_dv = MagicMock()
        mock_dv.transform.return_value = columns

        X_input = preprocess_input_data(df, mock_dv)

        self.assertEqual(X_input.shape, (1, 19))


if __name__ == "__main__":
    unittest.main()
