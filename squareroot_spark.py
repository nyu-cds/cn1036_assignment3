from pyspark import SparkContext
from operator import add
from math import sqrt

if __name__ == '__main__':
    sc = SparkContext("local", "squareroot")
        
    # Create an RDD of numbers from 1 to 1,000 #
    nums = sc.parallelize(range(1, 1001))
    # Compute the square root of each number in the RDD #
    sqrts = nums.map(sqrt)
    # Compute the average of the square roots #
    avg_of_sqrt = sqrts.fold(0, add)/sqrts.count()
    print(avg_of_sqrt)