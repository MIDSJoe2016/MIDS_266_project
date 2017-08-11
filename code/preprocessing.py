import glob, os, json, re, unicodedata
import numpy as np
from bs4 import BeautifulSoup
from sklearn.cross_validation import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
import nltk.data
from scipy import stats
from operator import itemgetter 

def get_data(sentence_max_threshold = 15000):
    load_verbose = 0
    loaded_labels = []
    loaded_text = []
    presidents = ["Barack Obama",
              "Donald J. Trump",
              "Dwight D. Eisenhower",
              "Franklin D. Roosevelt",
              "George Bush",
              "George W. Bush",
              "Gerald R. Ford",
              "Harry S. Truman",
              "Herbert Hoover",
              "Jimmy Carter",
              "John F. Kennedy",
              "Lyndon B. Johnson",
              "Richard Nixon",
              "Ronald Reagan",
              "William J. Clinton"]
    labels = {}
    for idx, name in enumerate(presidents):
        labels[name] = idx

    # load raw text files straight in, no parsing
    file_to_label = {"Obama": "Barack Obama", 
                         "Trump": "Donald J. Trump",
                         "Eisenhower": "Dwight D. Eisenhower",
                         "Roosevelt": "Franklin D. Roosevelt",
                         "Bush": "George Bush",
                         "WBush": "George W. Bush",
                         "Ford": "Gerald R. Ford",
                         "Truman": "Harry S. Truman",
                         "Hoover": "Herbert Hoover",
                         "Carter": "Jimmy Carter",
                         "Kennedy": "John F. Kennedy",
                         "Johnson": "Lyndon B. Johnson",
                         "Nixon": "Richard Nixon",
                         "Reagan": "Ronald Reagan",
                         "Clinton": "William J. Clinton"
                        }

    directory = "../data/processed/"
    for filename in glob.glob(os.path.join(directory, '*.txt')):
        arr = filename.replace(directory,'').split("_")
        if any(prefix in arr[0] for prefix in file_to_label.keys()):
            loaded_labels = loaded_labels + [labels[file_to_label[arr[0]]]]
            raw = open(filename).read().decode("UTF-8").encode("ascii","ignore")
            loaded_text = loaded_text + [raw] 
    #    else:
    #        print "Skipping ", filename

    print "Loaded", len(loaded_text), "speeches for", len(set(loaded_labels)), "presidents."
    # processed2 now contains files generated from unprocessed
    directory = "../data/processed3/"
    for filename in glob.glob(os.path.join(directory, '*.txt')):
        arr = filename.replace(directory,'').split("_")
        if any(prefix in arr[0] for prefix in file_to_label.keys()):
            loaded_labels = loaded_labels + [labels[file_to_label[arr[0]]]]
            raw = open(filename).read().decode("UTF-8").encode("ascii","ignore")
            loaded_text = loaded_text + [raw] 
    #    else:
    #        print "Skipping ", filename

    print "Loaded", len(loaded_text), "speeches for", len(set(loaded_labels)), "presidents."


    # In[2]:

    # summary stats & chop up into smaller
    #print "Loaded", len(input_text), "speeches for", len(set(input_labels)), "presidents."

    print "\nHow many speeches per president?"
    speech_freq = np.bincount(loaded_labels)
    for key, value in sorted(labels.iteritems()):
        print str(value).ljust(2), ":", key.ljust(20), "\t", speech_freq[value]

    print "\nApproximately many words of text per president?"
    vocab = set()
    for key, value in sorted(labels.iteritems()):
        label_set = [cnt for cnt, idx in enumerate(loaded_labels) if idx == value]
        label_speeches = [loaded_text[i] for i in label_set]
        print str(value).ljust(2), ":", key.ljust(20), "\t", sum(len(speech.split()) for speech in label_speeches)


    print "\nApproximately how many average words per speech per president?"
    for key, value in sorted(labels.iteritems()):
        label_set = [cnt for cnt, idx in enumerate(loaded_labels) if idx == value]
        label_speeches = [loaded_text[i] for i in label_set]
        print str(value).ljust(2), ":", key.ljust(20), "\t", (sum(len(speech.split()) for speech in label_speeches)) / speech_freq[value]


    # parse speeches into sentences and see what we have
    input_text = []
    input_labels = []

    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    for idx in range(0,len(loaded_text)):
        speech = loaded_text[idx]
        label = loaded_labels[idx]
        parsed_sentences = sent_detector.tokenize(speech.strip())
        input_text = input_text + parsed_sentences
        input_labels = input_labels + ([label]*len(parsed_sentences))

    print "Parsed ", len(input_text), "sentences, applying", len(input_labels), "labels."

    print "\nHow many sentences of text per president?"
    sentence_label_count = np.bincount(input_labels)
    for key, value in sorted(labels.iteritems()):
        print str(value).ljust(2), ":", key.ljust(20), "\t", sentence_label_count[value]

    print "\nSummary stats of sentence counts"
    print stats.describe(sentence_label_count)

    max_sentence_len_char = len(max(input_text, key=len))
    max_sentence_len_word = len(max(input_text, key=len).split())

    print "\nMaximum sentence length (characters):", max_sentence_len_char
    print "Maximum sentence length (words):", max_sentence_len_word
    print "\nLongest sentence:", max(input_text, key=len)


    # approach here is too simplistic but it suffices for now:
    #   If <= threshold, take all; else just pick first threshold # of sentences sentences

    trimmed_text = []
    trimmed_labels = []
    sentence_label_count = np.bincount(input_labels)

    for key, value in sorted(labels.iteritems()):
        # grab all values of a specific label
        subset_text = list(itemgetter(*[idx for idx, label in enumerate(input_labels) if label == value ])(input_text))
        subset_labels = list(itemgetter(*[idx for idx, label in enumerate(input_labels) if label == value ])(input_labels))

        if sentence_label_count[value] <= sentence_max_threshold:
            print str(value).ljust(2), ":", key.ljust(20), "\t", "copy", "\t", str(sentence_label_count[value]).ljust(6), "now at:", len(subset_text)
        else:
            subset_text = subset_text[0:sentence_max_threshold]
            subset_labels = subset_labels[0:sentence_max_threshold]
            print str(value).ljust(2), ":", key.ljust(20), "\t", "trim", "\t", str(sentence_label_count[value]).ljust(6), "now at:", len(subset_text)
        trimmed_text = trimmed_text + subset_text
        trimmed_labels = trimmed_labels + subset_labels

    # free up some memory
    subset_labels = None
    subset_text = None

    print "\nSentences trimmed from", len(input_text), "to", len(trimmed_text)
    print "\nLabels trimmed from", len(input_labels), "to", len(trimmed_labels)


    tokenizer = Tokenizer(num_words=None, #max_words, 
                          filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                          lower=True, split=" ", char_level=False)
    tokenizer.fit_on_texts(trimmed_text)
    tokenized_text = tokenizer.texts_to_sequences(trimmed_text)


    X = sequence.pad_sequences(tokenized_text, maxlen=max_sentence_len_word)
    y = to_categorical(trimmed_labels)

    train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.8, random_state=45)

    print "Prepared training (", len(train_X), "records) and test (", len(test_X), "records) data sets."
    return tokenizer, max_sentence_len_word, labels, train_X, test_X, train_y, test_y 

