import cs481_p03_A20483983_Bag_of_Words as BoW

def NB_model(train_set:[(int,{str,int})], test_set:[(int,{str,int})], stop_words_removed:bool):
    #size 9 scores * number of item in score * dictionary of str and int
    #scores between 1, 1.5, ..., 5
    print("Training classifier...")
    model = []
    for i in range(9):
        model.append({})
    samples = [0]*9
    for doc in train_set:
        samples[doc[0]-2] += 1
        for word, amount in doc[1].items():
            try:
                model[doc[0]-2][word] = model[doc[0]-2][word] + amount
            except KeyError:
                model[doc[0]-2][word] = amount
    total_samples = 0
    for sample in samples:
        total_samples += sample
    totals = fix_model(model, stop_words_removed)
    print("Testing classifier...")
    matrix = [[0]*9]*9
    vocab_size = BoW.get_vocab_size(stop_words_removed)
    for doc in test_set:
        testing = [(0, 0)]*9
        for i in range(9):
            testing[i] = test(model[i], samples[i], total_samples, totals[i]+vocab_size, doc[1])
        max_index = 0
        max_value = testing[0][0]/testing[0][1]
        for i in range(9):
            if testing[i][0]/testing[i][1] > max_value:
                max_value = testing[i][0]/testing[i][1]
                max_index = i
        matrix[max_index][doc[0]-2] += 1
    tp = 0
    tn = 0
    fp = 0
    for x in range(9):
        tp += matrix[x][x]
    for x in range(9):
        for y in range(9):
            if x != y:
                fp += matrix[x][y]
    fn = fp
    for x in range(9):
        for y in range(9):
            tn += matrix[x][y]
    tn = tn + tn
    tn - fn
    print("\nTest results / metrics:")
    print("Number of true positives: "+str(tp))
    print("Number of true negatives: " + str(tn))
    print("Number of false positives: " + str(fp))
    print("Number of false negatives: " + str(fn))
    print("Sensitivity (recall): " + str(tp/(tp+fn)))
    print("Specificity: "+str(tn/(tn+fp)))
    print("Precision: " + str(tp/(tp+fp)))
    print("Negative predictive value: " + str(tn/(tn+fn)))
    print("Accuracy: " + str((tp+tn)/(tp+tn+fp+fn)))
    print("F-score: " + str(tp/(tp+(fp+fn)/2)))


    while True:
        sentence_input = input("\nEnter your sentence/document: ")
        print("\nSentence/document S: "+sentence_input)
        sentence_input = BoW.create_bag_of_words(BoW.normalizing(sentence_input, stop_words_removed), stop_words_removed)
        testing = [(0, 0)] * 9
        for i in range(9):
            testing[i] = test(model[i], samples[i], total_samples, totals[i] + vocab_size, sentence_input)
        max_index = 0
        max_value = testing[0][0] / testing[0][1]
        for i in range(9):
            if testing[i][0] / testing[i][1] > max_value:
                max_value = testing[i][0] / testing[i][1]
                max_index = i
        print("\nwas classified as "+str((max_index+2)/2))
        for i in range(9):
            print("P("+str((i+2)/2)+" | S) = "+str(testing[i][0]/testing[i][1]))
        choice = 'n'
        while choice != 'y':
            choice = input("\nDo you want to enter another sentence [y/n]? ")
            if choice == "n":
                exit(0)


def fix_model(model:[{str,int}], stop_words_removed:bool) -> [int]:
    vocab_size = BoW.get_vocab_size(stop_words_removed)
    totals = []
    for score in model:
        total = 0
        for add in score.values():
            total = total + add
        totals.append(total)
        for word, amount in score.items():
            score[word] = (amount+1, total+vocab_size) #add 1 smoothing
    return totals

def test(model:{str,(int,int)}, samples:int, samples_total:int, defden:int, testing:{str,int}) -> (int,int):
    numerator = samples
    denominator = samples_total
    for word, amount in testing.items():
        times = amount
        while times > 0:
            try:
                numerator *= model[word][0]
                denominator *= model[word][1]
            except KeyError:
                #num * 1
                denominator *= defden
            times -= 1
    return numerator, denominator
