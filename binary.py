from itertools import permutations

def zbits(n, k):
    """
    The function prints all binary strings of length n that contain k zero bits.
    """
    
    try:
        # check arguments #
        if type(n)!=int or type(k)!=int:
            raise Exception("Error: n,k should be integers.")
        elif n<k:
            raise Exception("Error: n should be greater than k.")
        
        else:
            string = "0"*k + "1"*(n-k)

            # generate qualified strings using itertools.permutations #
            qualified_str = set()
            for i in permutations(string, n):
                qualified_str.add(''.join(i))

            print(qualified_str)
            return qualified_str

    except Exception as e:
        print(e)


# test #
#assert zbits(4, 3) == {'0100', '0001', '0010', '1000'}
#assert zbits(4, 1) == {'0111', '1011', '1101', '1110'}
#assert zbits(5, 4) == {'00001', '00100', '01000', '10000', '00010'}