import sys

from PIL.ImImagePlugin import split

import cs481_p03_A20483983_Bag_of_Words as BoW
import cs481_P03_A20483983_Naive_Bayes as NB
import cs481_P03_A20483983_K_Nearest_Neighbors as KNN

if __name__ == '__main__':
    try:
        algo = int(sys.argv[1])
        if not (1 >= algo >= 0):
            raise Exception
    except Exception:
        algo = 0

    try:
        size = int(sys.argv[2])
        if not (90 >= size >= 50):
            raise Exception
    except Exception:
        size = 80
    #print(BoW.normalizing("test string with punction.-&", False))
    #BoW.build_vocab(False)
    #print(BoW.create_bag_of_words('This is a test sentence. but need more to test the sentence properly', False))
    #print(BoW.split_documents(size, False))