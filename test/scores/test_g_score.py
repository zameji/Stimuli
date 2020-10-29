import unittest

from src.scores.gscore import g_score

class TestGScore(unittest.TestCase):
  
  def test_daudarivicius_marcinkieviciene_example(self):
    self.assertAlmostEqual(3, g_score(4,50,350,35,250))