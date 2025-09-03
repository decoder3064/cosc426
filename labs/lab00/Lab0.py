# ----------------------------------------------------------

# Implement the functions below in order, uncommenting one at a time
# 
# Doing so you have only its tests to debug and clear before
# running the tests of the next function


##### STARTER FUNCTIONS DO NOT MODIFY #####

import os
import doctest
import math

# additional import
import json 

def get_filepaths(root_dir:str) -> list:

    file_paths = []
    for dir_, _, files in os.walk(root_dir):
        for file_name in files:
            if file_name.endswith('.txt'):  
                abs_dir = os.path.abspath(dir_)
                fpath = os.path.join(abs_dir, file_name)
                file_paths.append(fpath)

    return file_paths


def initialize_vocab(fpath:str)->dict:
    with open(fpath, 'r', encoding="utf-8") as f:
        words = f.read().split()

    vocab = {'UNK': 0}
    for word in words:
        vocab[word.strip()] = 0

    return vocab

####### END OF STARTER FUNCTIONS #####

def clean_words(words: list[str]) -> list: 
    """Keeps words with just letters and digits.
    Also removes punctuation and converts to lower case. 

    Params:
        words: list of original words

    Returns:
        New list of words that are cleaned

    >>> clean_words(["XyZ123@", "isn't", "a", "real    ", "word?" "!!!", "RIGHT"])
    ['xyz123', 'isnt', 'a', 'real', 'word', 'right']

    >>> clean_words(["!!", "?%"])
    []
    """
    cleaned_words = []
    for word in words:
        alnum_word = ''.join(c for c in word if c.isalnum()) # only alphanumeric characters
        if alnum_word: # omit empty string
            cleaned_word = alnum_word.lower()
            cleaned_words.append(cleaned_word)
    return cleaned_words

def update_frequencies(freq_dict: dict, fpath: str) -> dict:
    """Updates existing frequency dictionary with the words in fpath

    Params:
        freq_dict: dictionary of frequencies
        fpath: path to the file with new text

    Returns: 
        Modified original dictionary (makes it easy for doctest)

    >>> update_frequencies({'here': 0, 'is':0, 'a':0, 'sentence':0, 'UNK':0}, 'data/test.txt')
    {'here': 1, 'is': 1, 'a': 2, 'sentence': 2, 'UNK': 1}

    >>> update_frequencies({'here': 2, 'is':3, 'a':1, 'sentence':0, 'UNK':1}, 'data/test.txt')
    {'here': 3, 'is': 4, 'a': 3, 'sentence': 2, 'UNK': 2}
    """
    updated_dict:dict[str, int] = freq_dict.copy() # return val.
    if 'UNK' not in freq_dict:
        updated_dict['UNK'] = 0
    
    try: # read file ? proceed : error
        with open(fpath, 'r') as f:
            for line in f:
                words = clean_words(line.split())

                # in keys ? Word +1 : UNK +1
                for word in words: 
                    if word in freq_dict:
                        updated_dict[word] += 1
                    else:
                        updated_dict['UNK'] += 1
    except FileNotFoundError:
        print(f'{fpath} NOT FOUND')
    
    return updated_dict

def get_probabilities(freq_dict: dict) -> dict: 
    """ Converts frequencies to probabilities

    Params: 
        freq_dict: dictionary with frequencies

    Returns: 
        New dictionary where frequencies are used to compute probabilities

    >>> get_probabilities({'here': 1, 'is':1, 'a':1, 'sentence':0, 'UNK':2})
    {'here': 0.2, 'is': 0.2, 'a': 0.2, 'sentence': 0.0, 'UNK': 0.4}

    >>> get_probabilities({'here': 4, 'is':6, 'a':0, 'sentence':0, 'UNK':0})
    {'here': 0.4, 'is': 0.6, 'a': 0.0, 'sentence': 0.0, 'UNK': 0.0}

    >>> get_probabilities({'here': 0.0, 'is':0.0, 'a':0.0, 'sentence':0.0, 'UNK':10})
    {'here': 0.0, 'is': 0.0, 'a': 0.0, 'sentence': 0.0, 'UNK': 1.0}

    >>> get_probabilities({'here': 0, 'is': 0, 'a': 0, 'sentence': 0, 'UNK': 0})
    {'here': 0, 'is': 0, 'a': 0, 'sentence': 0, 'UNK': 0}
    """
    prob_dict: dict[str, float] = dict.fromkeys(freq_dict, 0) # return val. Init w/ 0 instead of 0.0 b/c test reqmt. 
    freq_sum: int = sum(freq_dict.values()) 
    
    if freq_sum:
        for key, val in freq_dict.items():
            prob_dict[key] = val / freq_sum

    return prob_dict

def get_logprob_text(text:str, prob_dict:dict, eps: float) -> float:
    """ Returns log probability of a text given some probability dictionary

    Params: 
        text: text to compute the probability of
        prob_dict: dictionary returned by get_probabilties
        eps: some small value to make sure we are not taking log of 0. 

    Returns: 
        The sum of the log probabilities of all the words in the text given the prob_dict. 
        Hint: You can get log of a number x with math.log(x). Remember to add epsilon before taking the log! 
    
    >>> get_logprob_text('here is a sentence!!', {'here': 0.3, 'is': 0.3, 'a': 0.3, 'sentence': 0.15, 'UNK': 0.05}, 0.0001)
    -5.507372119950168

    >>> get_logprob_text('here is a new sentence!!', {'here': 0.3, 'is': 0.3, 'a': 0.2, 'sentence': 0.15, 'UNK': 0.05}, 0.0001)
    -8.906404901698119

    >>> get_logprob_text('here is some new text!!', {'here': 0.3, 'is': 0.3, 'a': 0.2, 'sentence': 0.15, 'UNK': 0.05}, 0.0001)
    -11.388481865745586
    """
    log_prob:float = 0.0 # reutrn val. Init 0.0 to fool proof
    words = clean_words(text.split())
    
    for word in words:
        # word in dict ? word : unk prob
        prob = prob_dict[word] if word in prob_dict else prob_dict['UNK']
        log_prob += math.log(prob + eps)
    
    return log_prob

def classify(text: str, class_dicts:dict, eps: float) -> str: 
    """Classifies text based on log probability given different classes. 

    Params: 
        text: text to be classified

        class_dicts: nested dictionary where keys are the classes, and values are dictionaries that correspond to the probability dictionary for the class (as derived from get_probabilities)

        eps: some small value to make sure we are not taking log of 0 in get_logprob_text

    Returns:
        The class in class_dicts that assigns the highest log prob value to the text


    >>> classify('here is some text!!',{'A': {'here': 0.2, 'is': 0.2, 'a': 0.2, 'sentence': 0.2, 'UNK': 0.2}, 'B': {'here': 0.2, 'is': 0.15, 'a': 0.3, 'sentence': 0.3, 'UNK': 0.05}},0.0001)
    'A'

    >>> classify('a word or two',{'A': {'one': 0.4, 'word': 0.4, 'UNK': 0.2}, 'B': {'one': 0.4, 'word': 0.3, 'UNK': 0.3}, 'C': {'one': 0.1, 'word': 0.4, 'UNK': 0.5}},0.0001)
    'C'
    """
    highest_class = '' # return val.
    best_logprob = -math.inf # give them this line
    
    for cls, prob_dict in class_dicts.items():    
        curr_logprob = get_logprob_text(text,prob_dict,eps)
        if curr_logprob > best_logprob:
            best_logprob = curr_logprob
            highest_class = cls
    
    return highest_class


def train(train_dict: dict, freq_dict: dict):
    """For each class, trains a model for each class (i.e., creates a probability dictionary) given a list of files.

    Params:
        train_dict: A dictionary where keys are the labels and the values are a list of files associated with the label to train the model

        freq_dict: A dictionary where keys are the words in vocabulary and the values are the counts of the words (which keeps getting updated over training)

    Returns: 
        Nothing. But modifies freq_dict. 

    """
    prob_dict = {} # file content
    
    for label, fpaths in train_dict.items():
        for fpath in fpaths:
            freq_dict[label] = update_frequencies(freq_dict[label], fpath)
        
        prob_dict[label] = get_probabilities(freq_dict[label])
        
    with open('prob_dict.json', "w") as f:
        json.dump(prob_dict, f, indent=4)


def classify_texts(class_dicts:dict, fpaths: dict, outfpath: str) -> None: 
    """ Classifies texts for different classes. Creates a file with all the sentences and predictions.   

    Params: 
        class_dicts: A dictionary where keys are the words in vocabulary and the values are the probability of the word given the label

        fpaths: A dictionary where keys are the labels and the values are a list of files associated with the label to evaluate the model on

        outfpath: A filepath to save all the predictions to
    """

    
    eps = 0.0001
    headers = ["text", "gold", "predicted", "correct"]
    output = ["\t".join(headers)]
    
    for label, paths in fpaths.items(): # Note: gold = label
        for path in paths:                
                with open(path, "r") as rf:
                    next(rf) # skipping title
                    for line in rf:
                        cleaned_line = line.strip().replace('\t', ' ') # Note: text = cleaned_line
                        if cleaned_line:
                            predicted = classify(cleaned_line, class_dicts, eps)
                            correct = str(int(label == predicted))
                            row = '\t'.join([cleaned_line, label, predicted, correct])
                            output.append(row)
    with open(outfpath, 'w') as f:
        f.writelines(line + '\n' for line in output)
def main():

    train_dict = {
        'swift': get_filepaths('data/train/swift/'),
        'shakespeare': get_filepaths('data/train/shakespeare/')
    }

    test = {
        'swift': get_filepaths('data/test/swift/'),
        'shakespeare': get_filepaths('data/test/shakespeare/')
    }

    test_toy = {
        'swift': get_filepaths('data/test/swift_toy/'),
        'shakespeare': get_filepaths('data/test/shakespeare_toy/')
    }

    freq_dict = {
        'swift': initialize_vocab('data/glove_vocab.txt'),
        'shakespeare': initialize_vocab('data/glove_vocab.txt')
    }

    outfpath = 'predictions.txt'
    outfpath_toy = 'predictions_toy.txt'

    ## Implement the rest of the main function
    try:
        train(train_dict, freq_dict)
    except Exception as e:
        print(f"TRAIN FAILED: {e}")
        
    class_dicts ={}
    
    with open("prob_dict.json", "r") as f:
        class_dicts = json.load(f)
        
    # for toy
    classify_texts(class_dicts, test_toy, outfpath_toy)

    # for test
    classify_texts(class_dicts, test, outfpath)


### DO NOT DELETE THESE LINES
doctest.testmod()
main()












