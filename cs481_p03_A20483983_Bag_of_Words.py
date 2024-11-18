import nltk
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
import os
import csv
import string

path_bad_data = "RateMyProfessor_Data_Set/RateMyProfessor_Sample_Data.csv"
path_clean_data = "stored_data/clean_data_set.csv"
# note to self prep_data has a makedir with folder name so don't change folder name only here

def prep_data(force_overwrite:bool=False):
    if not force_overwrite:
        if os.path.isfile(path_clean_data):
            return
    os.makedirs("stored_data/", exist_ok=True)
    file_w = open(path_clean_data, 'w')
    write_to = csv.writer(file_w, lineterminator='\n')
    file_r = open(path_bad_data, "r")
    file_read = csv.reader(file_r)
    first = True
    for line in file_read:
        if first:
            first = False
            continue
        comment = line[22]
        if comment != "" and comment != "No Comments":
            comment = normalizing(comment)
            good_data = [int(float(line[14])), int(float(line[15])), comment]
            write_to.writerow(good_data)
    file_r.close()
    file_w.close()

def normalizing(sentence:str, remove_punctuation:bool=True, remove_stop_words:bool=True, lemmatization:bool=True) -> str:
    # lower casing
    output = sentence.lower()

    # removal of punctuation and symbols
    if remove_punctuation:
        for remove_word in string.punctuation:
            output = output.replace(remove_word, ' ')

    # removal of stop words
    if remove_stop_words:
        words = output.split(' ')
        all_stop_words = nltk.corpus.stopwords.words('english')
        output = ''
        for word in words:
            if word != '':
                if word not in all_stop_words:
                    output = output + word + ' '
        output = output[:-1]

    # lemmatization
    if lemmatization:
        words = output.split(' ')
        output = ''
        lem = WordNetLemmatizer()
        for word in words:
            output = output + lem.lemmatize(word) + ' '
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




