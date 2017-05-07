from pyspark import SparkContext
from operator import and_
import re

# remove any non-words and split lines into separate words
# finally, convert all words to lowercase
def splitter(line):
    line = re.sub(r'^\W+|\W+$', '', line)
    return map(str.lower, re.split(r'\W+', line))

if __name__ == '__main__':
    sc = SparkContext("local", "distinctword")
        
    text = sc.textFile('pg2701.txt')
    words = text.flatMap(splitter)
    words_mapped = words.map(lambda x: (x,1))
    sorted_map = words_mapped.sortByKey()
    appearance = sorted_map.reduceByKey(and_)
    print(appearance.count())