from pyspark import SparkContext
from operator import mul

if __name__ == '__main__':
    sc = SparkContext("local", "product")
        
    # Create an RDD of numbers from 1 to 1,000 #
    nums = sc.parallelize(range(1, 1001))
    # Compute the product of numbers in the RDD #
    product = nums.fold(1, mul)
    print(product)