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
    if algo == 0:
        algo_type = "Naive Bayes"
    else:
        algo_type = "k Nearest Neighbors"
    try:
        size = int(sys.argv[2])
        if not (90 >= size >= 50):
            raise Exception
    except Exception:
        size = 80

    print("Prymon, Alan, A20483983 solution:\nTraining set size: "+str(size)+"%\nClassifier type: "+algo_type)
    remove_stop_words = False
    train_set, test_set = BoW.split_documents(size, remove_stop_words)
    if algo == 0:
        NB.NB_model(train_set, test_set, remove_stop_words)
    else:
        exit(-1)
