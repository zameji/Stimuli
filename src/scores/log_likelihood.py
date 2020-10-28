from decimal import Decimal
from math import log as built_in_log

def zero_log(num):
    """Wrapper to return log as a Decimal. If zero is given, zero is returned

    Args:
        num (Decimal): a number to calculate the log of
        base (int): base of the logarithm

    Returns:
        Decimal: logarithm of number with base, or 0 if input is zero
    """
    return Decimal(built_in_log(num)) if num != 0 else 0
    
def log_likelihood(a,b,c,d):
    """Calculate log likelihood of a bigram
    The collocation value is calculated as follows:

    2*( a*log(a) + b*log(b) + c*log(c) + d*log(d)
    - (a+b)*log(a+b) - (a+c)*log(a+c)
    - (b+d)*log(b+d) - (c+d)*log(c+d)
    + (a+b+c+d)*log(a+b+c+d))
    
    [T]he limit of x*ln(x) as x goes to zero is still zero, so when summing we can just ignore cells where x = 0. 
    (http://ucrel.lancs.ac.uk/llwizard.html)
    
    Args:
        bigram (string): a node-collocate pair
        a (int): the frequency of node - collocate pairs
        b (int): number of instances where the node does not co-occur with the collocate
        c (int): number of instances where the collocate does not co-occur with the node
        d (int): the number of words in the corpus minus the number of occurrences of the node and the collocate
        
    Returns:
        float: log-likelihood of a collocation
    """    

    a = Decimal(a)
    b = Decimal(b)
    c = Decimal(c)
    d = Decimal(d)
    
    base = [a, b, c, d, a+b, a+c, b+d, c+d, a+b+c+d]
        
    likelihood = 2*( a*zero_log(a) + b*zero_log(b) + c*zero_log(c) + d*zero_log(d) - (a+b)*zero_log(a+b) - (a+c)*zero_log(a+c) - (b+d)*zero_log(b+d) - (c+d)*zero_log(c+d) + (a+b+c+d)*zero_log(a+b+c+d))
 
    return(float(likelihood))