import unittest

from Perceptron.Perceptron import Perceptron


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.p1 = Perceptron(3, [2,2])


    def test_perceptronnand(self):
        self.assertEqual(self.p1.feed([1,0]),1,"Perceptron dio un 1")


if __name__ == '__main__':
    unittest.main()
