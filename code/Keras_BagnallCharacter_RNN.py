
# coding: utf-8

# In[1]:

# Include so results on different machines are (should be) the same.
from numpy.random import seed
seed(1)

from tensorflow import set_random_seed
set_random_seed(2)


# In[ ]:

get_ipython().system(u'jupyter nbconvert --to script Keras_BagnallCharacter_RNN.ipynb')


# In[ ]:

import glob, os, json, re, unicodedata
from bs4 import BeautifulSoup

load_verbose = 0
loaded_labels = []
loaded_text = []
presidents = [
#     "Barack Obama",
#     "Donald J. Trump",
#     "Dwight D. Eisenhower",
    "Franklin D. Roosevelt",
    "George Bush",
    "George W. Bush",
#     "Gerald R. Ford",
    "Harry S. Truman",
#     "Herbert Hoover",
#     "Jimmy Carter",
#     "John F. Kennedy",
#     "Lyndon B. Johnson",
#     "Richard Nixon",
#     "Ronald Reagan",
#     "William J. Clinton"
]

labels = {}
for idx, name in enumerate(presidents):
    labels[name] = idx

# load raw text files straight in, no parsing
file_to_label = {
#     "Obama": "Barack Obama",
#     "Trump": "Donald J. Trump",
#     "Eisenhower": "Dwight D. Eisenhower",
    "Roosevelt": "Franklin D. Roosevelt",
    "Bush": "George Bush",
    "WBush": "George W. Bush",
#     "Ford": "Gerald R. Ford",
    "Truman": "Harry S. Truman",
#     "Hoover": "Herbert Hoover",
#     "Carter": "Jimmy Carter",
#     "Kennedy": "John F. Kennedy",
#     "Johnson": "Lyndon B. Johnson",
#     "Nixon": "Richard Nixon",
#     "Reagan": "Ronald Reagan",
#     "Clinton": "William J. Clinton"
}

directory = "../data/processed/"
for filename in glob.glob(os.path.join(directory, '*.txt')):
    arr = filename.replace(directory,'').split("_")
    if any(prefix in arr[0] for prefix in file_to_label.keys()):
        loaded_labels = loaded_labels + [labels[file_to_label[arr[0]]]]
        raw = open(filename).read().decode("UTF-8").encode("ascii","ignore")
        loaded_text = loaded_text + [raw] 


print "Loaded", len(loaded_text), "speeches for", len(set(loaded_labels)), "presidents."
# processed2 now contains files generated from unprocessed
directory = "../data/processed3/"
for filename in glob.glob(os.path.join(directory, '*.txt')):
    arr = filename.replace(directory,'').split("_")
    if any(prefix in arr[0] for prefix in file_to_label.keys()):
        loaded_labels = loaded_labels + [labels[file_to_label[arr[0]]]]
        raw = open(filename).read().decode("UTF-8").encode("ascii","ignore")
        loaded_text = loaded_text + [raw] 

print "Loaded", len(loaded_text), "speeches for", len(set(loaded_labels)), "presidents."


# In[ ]:

#
# Bagnall 2015 text pre-processing
#
from string import maketrans
import re

chars_to_replace = "[]%!()>=*&_}+"
sub_chars = len(chars_to_replace) * " "
trantab = maketrans(chars_to_replace, sub_chars)
for x in range(0,len(loaded_text)):
    # "Various rare characters that seemed largely equivalent are mapped together..."
    loaded_text[x] = re.sub('`', '\'', loaded_text[x])
    # "...all digits in all languages are mapped to 7"
    loaded_text[x] = re.sub('[0-9]', '7', loaded_text[x])
    # "...any character with a frequency lower than 1 in 10,000 is discarded."
    loaded_text[x] = loaded_text[x].translate(trantab)
    # "Runs of whitespace are collapsed into a single space."
    loaded_text[x] = re.sub(' +', ' ', loaded_text[x])

print "Replacements complete."


# In[ ]:

#
# Join all speeches into one massive per president
#  for later processing
#
import numpy as np
from scipy import stats
from operator import itemgetter
from collections import defaultdict

compressed_text = [None]*(len(labels))
for key, value in sorted(labels.iteritems()):
    compressed_text[value] = ""
    for idx in range(0,len(loaded_text)):
        if (loaded_labels[idx] == value):
            compressed_text[value] = compressed_text[value] + loaded_text[idx] + " "
            
print "How many characters of text per president?"
for key, value in sorted(labels.iteritems()):
    print str(value).ljust(2), ":", key.ljust(20), "\t", len(compressed_text[value])

label_min_chars = len(min(compressed_text, key=len))
print "\nMinimum number of characters per president?"
print label_min_chars


# In[ ]:

#
# Tokenize words into chars
#
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

# Tokenize into characters
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(compressed_text)
tokenized_text = tokenizer.texts_to_sequences(compressed_text)

# there's an oddity in the encoding for some reason where a len+1 character occurs
unique_chars = len(tokenizer.word_counts)+1

print "Unique char count:", unique_chars
print "\nChars w/ counts:"
print sorted(((v,k) for k,v in tokenizer.word_counts.iteritems()), reverse=True)


# In[ ]:

#
# Split speeches into subsequences 
#
from collections import Counter

def splits(_list, _split_size):
    output_list = []
    for idx in range(0, len(_list), _split_size):
        if (idx + _split_size) <= len(_list):
            output_list.append(_list[idx:idx + _split_size])
    return output_list

max_seq_len = 50

# create new speech/label holders
split_text = []
split_labels = []

for idx in range(0, len(tokenized_text)):
    current_label = idx
    current_speech = tokenized_text[idx]#[:label_min_chars]
    current_splits = splits(current_speech, max_seq_len)
    split_text.extend(current_splits)
    split_labels.extend([current_label] * len(current_splits))

print "Sample splits, labels:", len( split_text ), len( split_labels )
print "\nOriginal total chars:", len( split_text ) * max_seq_len


# In[ ]:

#
# split amongst speaker samples, not the whole population of samples
#
def split_test_train(input_text, input_labels, labels, train_pct=0.8):
    train_text = []
    train_labels = []
    test_text = []
    test_labels = []

    for key, value in sorted(labels.iteritems()):
        # grab all values of a specific label
        subset_text = list(itemgetter(*[idx for idx, label in enumerate(input_labels) if label == value ])(input_text))
        subset_labels = list(itemgetter(*[idx for idx, label in enumerate(input_labels) if label == value ])(input_labels))
        
        cut_pos = int(train_pct * len(subset_text))
        train_text = train_text + subset_text[:cut_pos]
        train_labels = train_labels + subset_labels[:cut_pos]
        test_text = test_text + subset_text[cut_pos:]
        test_labels = test_labels + subset_labels[cut_pos:]
        
    return train_text,train_labels,test_text,test_labels


# In[ ]:

#
# Prep test/train
#
from sklearn.preprocessing import OneHotEncoder
from keras.utils import to_categorical
from sklearn.utils import class_weight

# compute class weights to account for imbalanced classes
y_weights = (class_weight.compute_class_weight('balanced', np.unique(split_labels), split_labels)).tolist()
print "Class weights:\n", y_weights

# split data smartly
train_X, train_y, test_X, test_y = split_test_train(split_text, split_labels, 
                                                    labels, train_pct=0.8)

print "Splits:\n Test = ", len(train_X), "\n Train = ", len(test_X)

# one-hot encode classes
train_y = to_categorical(train_y)
test_y = to_categorical(test_y)


# In[ ]:

#
# One-hot encoding samples
#
from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder(sparse=False,n_values=unique_chars)

train_X = np.array(train_X)
print "Encoding train_X with dimensions ", train_X.shape
train_X = enc.fit_transform(train_X)
print "...to ", train_X.shape
train_X = np.reshape(train_X,(train_X.shape[0],max_seq_len,unique_chars))
print "...and reshaping to ", train_X.shape

test_X = np.array(test_X)
print "\nEncoding test_X with dimensions ", test_X.shape
test_X = enc.transform(test_X)
print "...to ", test_X.shape
test_X = np.reshape(test_X,(test_X.shape[0],max_seq_len,unique_chars))
print "...and reshaping to ", test_X.shape


# In[ ]:

# custom activation from Bagnall 2015
#  we were never able to get this to work; either nan'ed or never converged
import tensorflow as tf

def ReSQRT(x):
    cond = tf.less_equal(x, 0)
    result = tf.where(cond, x + 0.0, tf.sqrt(x+1)-1)
    return result


# Bagnall proposes that the following possible values contribute to the success of the model:
# 
# | meta-parameter                  	| typical value                      	|
# |---------------------------------	|------------------------------------	|
# | initial adagrad learning scale  	| 0.1, 0.14, 0.2, 0.3                	|
# | initial leakage between classes 	| 1/4N to 5/N                        	|
# | leakage decay (per sub-epoch)   	| 0.67 to 0.9                        	|
# | hidden neurons                  	| 79, 99, 119, 139                   	|
# | presynaptic noise σ             	| 0, 0.1, 0.2, 0.3, 0.5              	|
# | sub-epochs                      	| 6 to 36                            	|
# | text direction                  	| forward or backward                	|
# | text handling                   	| sequential, concatenated, balanced 	|
# | initialisation                  	| gaussian, zero                     	|

# In[ ]:

from keras.models import Model
from keras.layers import Input, Dense, SimpleRNN, Dropout, Bidirectional, LSTM
from keras.optimizers import Adagrad, adam
from keras.callbacks import ReduceLROnPlateau, CSVLogger

# define operating vars
batch_size = 100
epochs = 50

# define optimizer
optimizer = Adagrad(lr=0.01)

# define any callbacks
reduce_lr = ReduceLROnPlateau(monitor='categorical_accuracy', factor=0.5,
              patience=1, verbose=1)
csv_logger = CSVLogger('Keras_BagnallCharacterRNN_training.log')

# assemble & compile model
print('Build model...')
main_input = Input(shape=(max_seq_len,unique_chars,))
rnn = Bidirectional(SimpleRNN(units=100,activation='relu'))(main_input)
drop = Dropout(0.5)(rnn)
main_output = Dense(len(labels),activation='softmax')(rnn)
model = Model(inputs=[main_input], outputs=[main_output])

model.compile(loss='categorical_crossentropy', 
              optimizer=optimizer, 
              metrics=['categorical_accuracy'])
print(model.summary())

# train
model.fit([np.array(train_X)],
          [np.array(train_y)],
          batch_size=batch_size,
          epochs=epochs,
          class_weight = y_weights,
          callbacks=[reduce_lr, csv_logger],
          verbose=1)

model.save('Keras_BagnallCharacterRNN_training.h5')  
del model


# In[ ]:

# Load computed model
from keras.models import load_model
# returns a compiled model
# identical to the previous one
model = load_model('Keras_BagnallCharacterRNN_training.h5')


# In[ ]:

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


# In[ ]:

# Plot confusion matrix
#   from scikit-learn examples @
#   http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html 
get_ipython().magic(u'matplotlib inline')
import itertools
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
        print(np.sum(cm,axis=0))

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    #plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(test_y_collapsed, pred_y_collapsed)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure(figsize=(10,10))
plot_confusion_matrix(cnf_matrix, classes=(sorted(labels, key=labels.get)),
                      title='Confusion matrix, without normalization')

#Plot normalized confusion matrix
plt.figure(figsize=(10,10))
plot_confusion_matrix(np.round(cnf_matrix,2), classes=(sorted(labels, key=labels.get)), normalize=True,
                      title='Normalized confusion matrix')

plt.show()


# In[ ]:



