
# coding: utf-8

# In[1]:


# Include so results on different machines are (should be) the same.
from numpy.random import seed
seed(1)

from tensorflow import set_random_seed
set_random_seed(2)


# In[16]:


get_ipython().system(u'jupyter nbconvert --to script Keras_Character_MultiHeadRNN.ipynb')


# In[2]:


import glob, os, json, re, unicodedata
from bs4 import BeautifulSoup

load_verbose = 0
loaded_labels = []
loaded_text = []
presidents = [
    "Barack Obama",
    "Donald J. Trump",
#     "Dwight D. Eisenhower",
#     "Franklin D. Roosevelt",
#     "George Bush",
    "George W. Bush",
#     "Gerald R. Ford",
#     "Harry S. Truman",
#     "Herbert Hoover",
#     "Jimmy Carter",
#     "John F. Kennedy",
#     "Lyndon B. Johnson",
#     "Richard Nixon",
#     "Ronald Reagan",
    "William J. Clinton"
]

labels = {}
for idx, name in enumerate(presidents):
    labels[name] = idx

# load raw text files straight in, no parsing
file_to_label = {
    "Obama": "Barack Obama",
    "Trump": "Donald J. Trump",
#     "Eisenhower": "Dwight D. Eisenhower",
#     "Roosevelt": "Franklin D. Roosevelt",
#     "Bush": "George Bush",
    "WBush": "George W. Bush",
#     "Ford": "Gerald R. Ford",
#     "Truman": "Harry S. Truman",
#     "Hoover": "Herbert Hoover",
#     "Carter": "Jimmy Carter",
#     "Kennedy": "John F. Kennedy",
#     "Johnson": "Lyndon B. Johnson",
#     "Nixon": "Richard Nixon",
#     "Reagan": "Ronald Reagan",
    "Clinton": "William J. Clinton"
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


# In[3]:


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
    loaded_text[x] = re.sub('--', '-', loaded_text[x])
    loaded_text[x] = re.sub('\n\n', '\n', loaded_text[x])
    # "...all digits in all languages are mapped to 7"
    loaded_text[x] = re.sub('[0-9]+', '7', loaded_text[x])
    # "...any character with a frequency lower than 1 in 10,000 is discarded."
    loaded_text[x] = loaded_text[x].translate(trantab)
    # "Runs of whitespace are collapsed into a single space."
    loaded_text[x] = re.sub(' +', ' ', loaded_text[x])
     
    # REPLACE WORD IN ALL CAPS with <space>; headers
    loaded_text[x] = re.sub('[A-Z]{2,}','', loaded_text[x])

print "Character clean-up complete."


# In[4]:


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


# In[5]:


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


# In[6]:


#
# Split speeches into subsequences 
#
from collections import Counter

def splits(_list, _split_size, window=False):
    output_list = []
    if (window):
        for idx in range(0, len(_list)-_split_size):
            output_list.append(_list[idx:idx + _split_size])
    else:
        for idx in range(0, len(_list), _split_size):
            if (idx + _split_size) <= len(_list):
                output_list.append(_list[idx:idx + _split_size])
    return output_list

max_seq_len = 25

# create new speech/label holders
split_text = []
split_labels = []

for idx in range(0, len(tokenized_text)):
    current_label = idx
    current_speech = tokenized_text[idx]#[:label_min_chars]
    current_splits = splits(current_speech, max_seq_len)
    split_text.extend(current_splits)
    split_labels.extend([current_label] * len(current_splits))

print "Subsequence total count; subsequence label total count:", len( split_text ), len( split_labels )
print "\nTotal characters:", len( split_text ) * max_seq_len


# In[7]:


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


# In[8]:


#
# Prep test/train
#
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import class_weight

# split data smartly
train_X, train_y, test_X, test_y = split_test_train(split_text, split_labels, 
                                                    labels, train_pct=0.8)
print "Splits:\n Test = ", len(train_X), "\n Train = ", len(test_X)##

# compute class weights to account for imbalanced classes
y_weights = (class_weight.compute_class_weight('balanced', np.unique(train_y), train_y)).tolist()
y_weights = dict(zip(sorted(labels.values()), y_weights))
print "Class weights:\n", y_weights


# In[9]:


#
# One-hot encoding classes & samples
#
from keras.utils import to_categorical

# one-hot encode classes
train_y = np.array(to_categorical(train_y))
test_y = np.array(to_categorical(test_y))

# one-hot encode samples
train_X = np.array(train_X)
orig_train_X_size=train_X.shape[0]
print "Encoding train_X with dimensions ", train_X.shape
train_X = to_categorical(train_X, num_classes=unique_chars)
print "...to ", train_X.shape
train_X = np.reshape(train_X,(orig_train_X_size,max_seq_len,unique_chars))
print "...and reshaping to ", train_X.shape

test_X = np.array(test_X)
orig_test_X_size=test_X.shape[0]
print "\nEncoding test_X with dimensions ", test_X.shape
test_X = to_categorical(test_X, num_classes=unique_chars)
print "...to ", test_X.shape
test_X = np.reshape(test_X,(orig_test_X_size,max_seq_len,unique_chars))
print "...and reshaping to ", test_X.shape


# In[10]:


# custom activation from Bagnall 2015
#  does not appear to perform as well as a ReLU
import tensorflow as tf
from keras.utils.generic_utils import get_custom_objects

def ReSQRT(x):
    cond = tf.less_equal(x, 0.0)
    result = tf.where(cond, x * 0.0, tf.sqrt(x+1)-1)
    return result

get_custom_objects().update({'ReSQRT': ReSQRT})


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


##
## MODEL OPTIMIZATION
##
from keras.layers import Input, Dense, SimpleRNN, Bidirectional
from keras.layers.merge import Maximum, Add, Concatenate
from keras.callbacks import ReduceLROnPlateau, CSVLogger
from keras.layers.merge import Average, Maximum
from keras.optimizers import Adagrad, adam
from keras.models import Model

from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

# define operating vars
# activation = "relu" #"ReSQRT" 
# units = 150# 50
# dropout = 0.0 #0.7646166765488501
# batch_size = 50# 100
# epochs = 100
# optimizer='adamax'#'rmsprop'
# shuffle=True #False

def create_model(optimizer='rmsprop', learn_rate=0.01,
                 init_mode1='glorot_uniform', init_mode2='glorot_uniform', 
                 merge_mode='ave', activation='relu', 
                 dropout_rate=0.0, neuron_count=50):

    # assemble & compile model
    input = Input(shape=(max_seq_len,unique_chars,))
    rnn = Bidirectional(SimpleRNN(units=neuron_count,
                                  activation=activation,
                                  recurrent_dropout=dropout_rate,
                                  kernel_initializer=init_mode1),
                        merge_mode=merge_mode)(input)
    
    soft_out = []
    for idx in range(0,len(labels)):
        soft_out.append(Dense(len(labels),
                              activation='softmax', 
                              kernel_initializer=init_mode2)(rnn))
    final_out = Add()(soft_out)

    model = Model(inputs=[input], outputs = final_out) 

    model.compile(loss='categorical_crossentropy', 
                  optimizer=optimizer, 
                  metrics=['accuracy'])

    return model


# instantiate model
model = KerasClassifier(build_fn=create_model, verbose=1, epochs=5)

# define the grid search parameters
epoch = [3]
batch_sizes = [25, 50, 75, 100, 200]
optimizers = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
init_modes1 = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
init_modes2 = init_modes1
merge_modes = ['sum', 'mul', 'concat', 'ave', None]
activations = ['ReSQRT','relu','sigmoid','tanh']
dropout_rates = [0.0,0.2,0.4,0.6,0.8]
neuron_counts = [25,50,75,100,150,200]
learn_rates = [0.001, 0.01, 0.1, 0.2, 0.3]  #currently ignored
param_grid = dict(batch_size=batch_sizes,
                  epochs=epoch,
                  optimizer=optimizers,
                  learn_rate=learn_rates,
                  init_mode1=init_modes1,
                  init_mode2=init_modes2,
                  merge_mode=merge_modes,
                  activation=activations,
                  dropout_rate=dropout_rates,
                  neuron_count=neuron_counts)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)
grid_result = grid.fit(train_X, train_y)

# summarize results
# from http://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# In[ ]:



