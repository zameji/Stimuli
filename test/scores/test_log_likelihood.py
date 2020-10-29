import unittest
from src.scores.log_likelihood import log_likelihood

class TestLogLikelihood(unittest.TestCase):
    
    def test_dunning_1993_examples(self):
        self.assertEqual(270.72, round(log_likelihood(110,2442,111,29114), 2))
        self.assertEqual(100.96, round(log_likelihood(8,2,27,31740), 2))
        self.assertEqual(61.61, round(log_likelihood(3,0,0,31774), 2))
        self.assertEqual(43.35, round(log_likelihood(20,2532,25,29200), 2))
        self.assertEqual(36.98, round(log_likelihood(3,10,5,31759), 2))