import unittest
import parallel_sorter

class testSort(unittest.TestCase):
    def setUp(self):
        pass

    def testSort_Parallel(self):
        result = parallel_sorter.sort_parallel(10000)
        self.assertEqual(result, sorted(result), 'test sorted result fail')


if __name__ =='__main__':
    unittest.main()