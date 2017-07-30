import glob, os, json, re, unicodedata
from bs4 import BeautifulSoup


load_verbose = 0
loaded_labels = []
loaded_text = []
labels = {"Barack Obama": 0,
          "Donald J. Trump": 1,
          "Dwight D. Eisenhower": 2,
          "Franklin D. Roosevelt": 3,
          "George Bush": 4,
          "George W. Bush": 5,
          "Gerald R. Ford": 6,
          "Harry S. Truman": 7,
          "Herbert Hoover": 8,
          "Jimmy Carter": 9,
          "John F. Kennedy": 10,
          "Lyndon B. Johnson": 11,
          "Richard Nixon": 12,
          "Ronald Reagan": 13,
          "William J. Clinton": 14}

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
        loaded_labels = loaded_labels + [labels[file_to_label[arr[0]]]]
        raw = open(filename).read().decode("UTF-8").encode("ascii","ignore")
        loaded_text = loaded_text + [raw] 

print "Loaded", len(loaded_text), "speeches for", len(set(loaded_labels)), "presidents."
# processed2 now contains files generated from unprocessed
directory = "../data/processed3/"
for filename in glob.glob(os.path.join(directory, '*.txt')):
        arr = filename.replace(directory,'').split("_")
        loaded_labels = loaded_labels + [labels[file_to_label[arr[0]]]]
        raw = open(filename).read().decode("UTF-8").encode("ascii","ignore")
        loaded_text = loaded_text + [raw] 


print "Loaded", len(loaded_text), "speeches for", len(set(loaded_labels)), "presidents."




###################

import numpy as np
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

##################
import nltk.data
from scipy import stats

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

#######################
# adjust sentence volumes 
from operator import itemgetter 

# approach here is too simplistic but it suffices for now:
#   If <= threshold, take all; else just pick first threshold # of sentences sentences

sentence_max_threshold = 10000

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
#########################
## USE NLTK Tokenizer instead?

from sklearn.cross_validation import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical

#max_words = len(tokenizer.word_counts) #15000

tokenizer = Tokenizer(num_words=None, #max_words, 
                      filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                      lower=True, split=" ", char_level=False)
tokenizer.fit_on_texts(trimmed_text)
tokenized_text = tokenizer.texts_to_sequences(trimmed_text)


X = sequence.pad_sequences(tokenized_text, maxlen=max_sentence_len_word)
y = to_categorical(trimmed_labels)

train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.8, random_state=45)

print "Prepared training (", len(train_X), "records) and test (", len(test_X), "records) data sets."
print X.shape, y.shape, train_X.shape, train_y.shape, len(tokenizer.word_counts), len(tokenized_text)

###########################
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, SimpleRNN, Dropout

max_features = len(tokenizer.word_counts)+1 #15000
batch_size = 100

print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 100, input_length=max_sentence_len_word))
#model.add(Dropout(0.5))
model.add(SimpleRNN(100,input_dim=100,activation='tanh',return_sequences=True))
model.add(Dropout(0.5))
model.add(SimpleRNN(50,activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(15, activation='softmax'))
#model.add(Dense(64, input_dim=64,
#                kernel_regularizer=regularizers.l2(0.01),
#                activity_regularizer=regularizers.l1(0.01)))
model.compile(loss='categorical_crossentropy', optimizer='Adagrad',metrics=['categorical_accuracy'])
print(model.summary())

model.fit(train_X, y=train_y, batch_size=batch_size, nb_epoch=60, verbose=1)

from keras.models import load_model
model.save('model_60.h5')  # creates a HDF5 file 'my_model.h5'

model2 = load_model('model_60.h5')

############################
# Evaluate performance
print "Evaluating test data..."
loss_and_metrics = model.evaluate(test_X, test_y)
print model.metrics_names
print loss_and_metrics

# Make some predictions
print "\nPredicting using test data..."
pred_y = model.predict(test_X, batch_size=batch_size, verbose=1)
pred_y_collapsed = np.argmax(pred_y, axis=1)
test_y_collapsed = np.argmax(test_y, axis=1)

print "Done prediction."

