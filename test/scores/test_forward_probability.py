import unittest

from src.scores.forward_probability import forward_probability
from src.scores.backward_probability import backward_probability

class TestForwardProbability(unittest.TestCase):
  
  def test_forward_probability(self):
    self.assertAlmostEqual(0.1, forward_probability(5,50))
    self.assertAlmostEqual(1, forward_probability(50,50))
    self.assertAlmostEqual(0, forward_probability(0,50))
    self.assertAlmostEqual(1/3, forward_probability(1,3))

  def test_backward_probability(self):
    self.assertAlmostEqual(0.1, backward_probability(5,50))
    self.assertAlmostEqual(1, backward_probability(50,50))
    self.assertAlmostEqual(0, backward_probability(0,50))
    self.assertAlmostEqual(1/3, backward_probability(1,3))