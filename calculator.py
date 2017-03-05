# -----------------------------------------------------------------------------
# calculator.py
#
# original: 1.67 s
# refined: 14.6 ms
#
# speedup = 114.38
#
# eliminating loops with numpy functions:
# 1. replacing the function 'multiply' with 'np.square'
# 2. using xx+yy instead of add(xx, yy)
# 3. replacing 'sqrt' with 'np.sqrt'
#
# ----------------------------------------------------------------------------- 
import numpy as np

def hypotenuse(x,y):
    """
    Return sqrt(x**2 + y**2) for two arrays, a and b.
    x and y must be two-dimensional arrays of the same shape.
    """
    xx = np.square(x)
    yy = np.square(y)
    return np.sqrt(xx + yy)