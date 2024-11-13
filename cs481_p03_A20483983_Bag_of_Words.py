import nltk
import os
import csv
import string

# returns a list of strings of stop words in english language
def stop_words() -> list:
    return nltk.corpus.stopwords.words('english')

def normalizing(sentence:str, remove_stop_words:bool) -> str:
    output = sentence.lower()
    for remove_word in string.punctuation:
        output = output.replace(remove_word, ' ')
    if remove_stop_words:
        words = output.split(' ')
        all_stop_words = stop_words()
        output = ''
        for word in words:
            if word not in all_stop_words:
                output = output + word + ' '
        output = output[:-1]
    return output

def build_vocab(remove_stop_words:bool):
    path = "RateMyProfessor_Data_Set/RateMyProfessor_SampleData/"
    list_of_files = os.listdir(path)
    os.makedirs("stored_data/", exist_ok=True)
    vocab = []
    for file in list_of_files:
        file_read = csv.reader(open(path + file, "r"))
        first = True
        for line in file_read:
            if first:
                first = False
                continue
            for word in normalizing(line[22], remove_stop_words).split(' '):
                if word.isascii():
                    vocab.append(word)
        vocab = list(set(vocab))
    try:
        vocab.remove('')
    except ValueError:
        pass
    if remove_stop_words:
        file_write = open("stored_data/vocabulary_swr.txt", "w") #swr = stop words removed
    else:
        file_write = open("stored_data/vocabulary_swnr.txt", "w")  # swnr = stop words NOT removed
    file_write.write(str(len(vocab)) + "\n")
    for word in vocab:
        file_write.write(word + "\n")
    file_write.close()

def get_vocab_file(stop_words_removed:bool):
    if stop_words_removed:
        if os.path.isfile("/stored_data/vocabulary_swr.txt"):
            file = open("stored_data/vocabulary_swr.txt", "r")
        else:
            build_vocab(stop_words_removed)
            file = open("stored_data/vocabulary_swr.txt", "r")
    else:
        if os.path.isfile("/stored_data/vocabulary_swr.txt"):
            file = open("stored_data/vocabulary_swnr.txt", "r")
        else:
            build_vocab(stop_words_removed)
            file = open("stored_data/vocabulary_swnr.txt", "r")
    return file

def get_vocab_size(stop_words_removed:bool) -> int:
    file = get_vocab_file(stop_words_removed)
    length = int(file.readline())
    file.close()
    return length

# returns a dictionary of words and there amounts found in the document aka non-binary bag of words
def create_bag_of_words(sentence:str, remove_stop_words) -> {str, int}:
    input_string = normalizing(sentence, remove_stop_words)
    input_string = input_string.split()
    while input_string.count('') > 0: input_string.remove('')
    bag = {}
    for word in input_string:
        try:
            bag[word] = bag[word] + 1
        except KeyError:
            bag[word] = 1
    return bag

def split_documents(train_size:int, remove_stop_words:bool) -> ([(int, {str, int})], [(int, {str, int})]):
    training_set = []
    testing_set = []
    path = "RateMyProfessor_Data_Set/RateMyProfessor_SampleData/"
    list_of_files = os.listdir(path)
    documents = []
    for file in list_of_files:
        file_read = csv.reader(open(path + file, "r"))
        first = True
        for line in file_read:
            if first:
                first = False
            else:
                try:
                    documents.append((int(float(line[13]) * 2), create_bag_of_words(normalizing(line[22], remove_stop_words), remove_stop_words)))
                except Exception:
                    pass
    amount = len(documents)
    for i in range(amount):
        doc = documents[i]
        if (i+1)/amount <= (train_size/100):
            training_set.append(doc)
        if (i+1)/amount >= 0.80:
            testing_set.append(doc)
    #print(str(len(training_set))) used to check if they are changing size and they are
    #print(str(len(testing_set)))
    return training_set, testing_set




