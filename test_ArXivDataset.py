import unittest
import os
from ArXivDataset import ArXivDataset

class TestArXivDataset(unittest.TestCase):

    def setUp(self):
        # Set up the object to be tested
        self.dataset = ArXivDataset(os.environ['ARXIV_SRC_PATH'])

    def test_init(self):
        # Tests whether the object is properly initialized
        self.assertEqual(self.dataset.data_dir, os.environ['ARXIV_SRC_PATH'])
        self.assertIsNotNone(self.dataset.manifest)
        
    def test_len(self):
        # Tests the __len__ method
        self.assertEqual(len(self.dataset), 2267263)

    def test_getitem(self):
        # Tests the __getitem__ method
        first_item = self.dataset[0]
        self.assertIsNotNone(first_item)
        # Provide the expected first item name
        self.assertEqual(first_item, '9107/hep-lat9107001.gz')

        last_item = self.dataset[len(self.dataset)-1]
        self.assertIsNotNone(last_item)
        # Provide the expected last item name
        self.assertEqual(last_item, '2305/2305.20092.gz')

if __name__ == '__main__':
    unittest.main()
