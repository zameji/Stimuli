from decimal import Decimal
from math import log

def g_score(bigram_freq: int, w_freq_item_1: int, w_freq_item_2: int, fwd_type_count_item_1: int, bckw_type_count_item_2: int) -> float:
  """Calculate G-Score of a node-collocate pair. 
  Based on Daudaravicius and Marcinkeviciene (or Petrauskaite) 2004. Gravity Counts for the boundaries of collocations
  Calculated as 
  
  G = log2((bigram frequency * number of different words to the right of the left word) / frequency of the left word) +
    log2((bigram_frequency * number of different words to the left of the right word) / frequency of the right word)

  Args:
      bigram_freq (int): bigram frequency
      w_freq_item_1 (int): frequency of the left word
      w_freq_item_2 (int): frequency of the right word
      fwd_type_count_item_1 (int): number of different words to the right of the left word
      bckw_type_count_item_2 (int): number of different words to the left of the right wor

  Returns:
      float: Gravity Counts
  """
  return float(log(Decimal(bigram_freq) * Decimal(fwd_type_count_item_1)/Decimal(w_freq_item_1),2) + log(Decimal(bigram_freq) * Decimal(bckw_type_count_item_2)/Decimal(w_freq_item_2),2))
    