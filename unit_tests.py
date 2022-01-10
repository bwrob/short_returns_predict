import unittest
import helpers as hlp
import pandas as pd
import fastparquet
import pyarrow


class Tests(unittest.TestCase):
    def test_historical_data_fetched(self):
        """
        Test that data fetched from YF has all necessary columns
        """
        result = hlp.get_ticker_historical_data("GME", use_persisted_data=False)
        mandatory_columns = ["Open", "High", "Low", "Close", "Volume"]
        for column in mandatory_columns:
            self.assertTrue(column in result.columns)

    def test_historical_data_persisted(self):
        """
        Test that streamed in data has all necessary columns
        """
        result = hlp.get_ticker_historical_data("GME", use_persisted_data=False)
        mandatory_columns = ["Open", "High", "Low", "Close", "Volume"]
        for column in mandatory_columns:
            self.assertTrue(column in result.columns)

    def test_regression_statistics(self):
        """
        Test correctness of calculated statistics is checked against regression data that was checked manually
        """
        input_data = pd.read_csv("GME" + hlp.STR_CSV_FORMAT)
        # using regression data saved in parquet for float-like precision
        expected_data = pd.read_parquet("GME_statistics_regression_data.park")
        result = hlp.add_statistics(input_data, sma_windows=(5, 30, 100), ema_alphas=(0.5, 0.25))
        compare = result.compare(expected_data)
        self.assertTrue(compare.empty)


if __name__ == '__main__':
    unittest.main()
