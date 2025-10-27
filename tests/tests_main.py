import os
import unittest
import pandas as pd
from main import CSVDataLoader, DataLoadError, IdealFunctionMatcher

class TestLoaders(unittest.TestCase):
    def test_csv_loader_success(self):
        p = os.path.join(os.getcwd(), "train.csv")
        loader = CSVDataLoader(p)
        df = loader.load()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn('x', df.columns)

    def test_csv_loader_missing(self):
        loader = CSVDataLoader("no_file_here_123.csv")
        with self.assertRaises(DataLoadError):
            loader.load()


class TestMatcher(unittest.TestCase):
    def setUp(self):
        base = os.getcwd()
        self.train = pd.read_csv(os.path.join(base, "train.csv"))
        self.ideal = pd.read_csv(os.path.join(base, "ideal.csv"))
        self.test = pd.read_csv(os.path.join(base, "test.csv"))
        self.matcher = IdealFunctionMatcher(self.train, self.ideal, self.test)

    def test_find_best(self):
        mapping = self.matcher.find_best_ideal_functions()
        self.assertTrue(all(col in mapping for col in [c for c in self.train.columns if c != 'x']))

    def test_map_test_points(self):
        self.matcher.find_best_ideal_functions()
        res = self.matcher.map_test_points()
        self.assertIn('ideal_function', res.columns)
        self.assertIn('delta_y', res.columns)


if __name__ == "__main__":
    unittest.main()
