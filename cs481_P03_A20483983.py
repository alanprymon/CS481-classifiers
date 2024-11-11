import sys

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
    print(str(algo) + ' ' + str(size))
