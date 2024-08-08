import unittest
import requests
from scripts.send_prediction_request import generate_data, convert_ndarray_to_list


class TestIntegration(unittest.TestCase):
    def test_prediction_endpoint(self):
        data = generate_data()
        data = convert_ndarray_to_list(data)

        url = "http://localhost:5002/predict"
        headers = {"Content-Type": "application/json"}
        response = requests.post(url, headers=headers, json=data)

        self.assertEqual(response.status_code, 200)
        json_response = response.json()
        self.assertIn("predictions", json_response)
        self.assertTrue(len(json_response["predictions"]) > 0)


if __name__ == "__main__":
    unittest.main()
