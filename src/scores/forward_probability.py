from decimal import Decimal

def forward_probability(bigram_frequency: int, word_1_frequency: int) -> float:
  """Calculate forward probability
  Equals 
    fwd_prob = bigram frequency / left word frequency

  Args:
      bigram_frequency (int): number of times the bigram was observed in the corpus
      word_1_frequency (int): number of occurrences of the left word

  Returns:
      float: forward probability
  """
  return float(Decimal(bigram_frequency) / Decimal(word_1_frequency))