import numpy as np
import operator
"""
CMSC723 / INST725 / LING723 -- Fall 2016
Project 1: Implementing Word Sense Disambiguation Systems
"""


"""
 read one of train, dev, test subsets

 subset - one of train, dev, test

 output is a tuple of three lists
 	labels: one of the 6 possible senses <cord, division, formation, phone, product, text >
 	targets: the index within the text of the token to be disambiguated
 	texts: a list of tokenized and normalized text input (note that there can be multiple sentences)

"""
import nltk
#### added dev_manual to the subset of allowable files
def read_dataset(subset):
	labels = []
	texts = []
	targets = []
	if subset in ['train', 'dev', 'test','dev_manual']:
		with open('data/wsd_'+subset+'.txt') as inp_hndl:
			for example in inp_hndl:
				label, text = example.strip().split('\t')
				text = nltk.word_tokenize(text.lower().replace('" ','"'))
				if 'line' in text:
					ambig_ix = text.index('line')
				elif 'lines' in text:
					ambig_ix = text.index('lines')
				else:
					ldjal
				targets.append(ambig_ix)
				labels.append(label)
				texts.append(text)
		return (labels, targets, texts)
	else:
		print '>>>> invalid input !!! <<<<<'

"""
computes f1-score of the classification accuracy

gold_labels - is a list of the gold labels
predicted_labels - is a list of the predicted labels

output is a tuple of the micro averaged score and the macro averaged score

"""
import sklearn.metrics
#### changed method name from eval because of naming conflict with python keyword
def eval_performance(gold_labels, predicted_labels):
	return ( sklearn.metrics.f1_score(gold_labels, predicted_labels, average='micro'),
			 sklearn.metrics.f1_score(gold_labels, predicted_labels, average='macro') )


"""
a helper method that takes a list of predictions and writes them to a file (1 prediction per line)
predictions - list of predictions (strings)
file_name - name of the output file
"""
def write_predictions(predictions, file_name):
	with open(file_name, 'w') as outh:
		for p in predictions:
			outh.write(p+'\n')

"""
Trains a naive bayes model with bag of words features and computes the accuracy on the test set

train_texts, train_targets, train_labels are as described in read_dataset above
The same thing applies to the reset of the parameters.
"""
def run_bow_naivebayes_classifier(train_texts, train_targets, train_labels,
				dev_texts, dev_targets,dev_labels, test_texts, test_targets, test_labels):

	"""
	**Your final classifier implementation of part 2 goes here**
	"""
	pass

## extract all the distinct words from a set of texts
## return a dictionary {word:index} that maps each word to a unique index
def extract_all_words(texts,prev_set=set()):
    all_words = prev_set
    for t in texts:
        for w in t:
            all_words.add(w)

    all_words_idx = {}
    for i,w in enumerate(all_words):
        all_words_idx[w] = i

    return all_words_idx

## extract all distinct labels from a dataset
## return a dictionary {label:index} that maps each label to a unique index
def extract_all_labels(labels):
    distinct_labels = list(set(labels))

    all_labels_idx = {}
    for i,l in enumerate(distinct_labels):
        all_labels_idx[l] = i

    return all_labels_idx

## compute the feature vector for a set of words and a given label
## the features are computed as described in Slide #19 of:
## http://www.cs.umd.edu/class/fall2016/cmsc723/slides/slides_02.pdf
def compute_features_for_label(all_words_idx,all_labels_idx,instance,label):
    feature_vector = np.zeros((len(all_words_idx.keys()),len(all_labels_idx.keys())))
    for word in instance:
        if all_words_idx.get(word,None) is None:
            continue
        feature_vector[all_words_idx[word]][all_labels_idx[label]] += 1
    return np.append(feature_vector.flatten(),1)

## get the predicted label for a given instance
## the predicted label is the one with the highest dot product of theta*feature_vector
## return the predicted label, the dot product scores for all labels and the features computed for all labels for that instance
def get_predicted_label(inst,class_labels,theta):
  all_labels_scores = {}
  all_labels_features = {}
  for lbl in class_labels:
      feat_vec = get_features_for_label(inst,lbl,class_labels)
      #print len(feat_vec),len(theta)
      #exit()
      assert len(feat_vec) == len(theta)
      all_labels_scores[lbl] = np.dot(feat_vec,theta)

  predicted_label = max(all_labels_scores.iteritems(), key=operator.itemgetter(1))[0]
  return predicted_label

## train the perceptron by iterating over the entire training dataset
## the algorithm is an implementation of the pseudocode from Slide #23 of:
## http://www.cs.umd.edu/class/fall2016/cmsc723/slides/slides_03.pdf
def train_perceptron(train_features,train_labels,class_labels,NUM_PARAMS):
  theta = np.zeros(NUM_PARAMS)

  print '# Training Instances:',len(train_features)

  np.random.seed(1)
  shuffled_indices = np.arange(len(train_features))
  np.random.shuffle(shuffled_indices)

  num_iterations = 0
  cnt_updates = 0
  m = np.zeros(NUM_PARAMS)
  for piter in range(50):
      for i in shuffled_indices:
          inst = train_features[i]
          actual_label = train_labels[i]
          predicted_label = get_predicted_label(inst,class_labels,theta)

          if predicted_label != actual_label:
              cnt_updates += 1
              theta = theta + get_features_for_label(inst,actual_label,class_labels) - get_features_for_label(inst,predicted_label,class_labels)
              m = m + theta

          num_iterations += 1

  theta = m/cnt_updates
  print '# Updates:',cnt_updates
  return theta

## return the predictions of the perceptron on a test set
def test_perceptron(theta,test_features,test_labels,class_labels):
  predictions = []
  for inst in test_features:
      predicted_label = get_predicted_label(inst,class_labels,theta)
      predictions.append(predicted_label)
  return predictions


def extract_features(all_words_idx,all_labels_idx,texts):
    NUM_FEATURES = len(all_words_idx.keys())*len(all_labels_idx.keys())
    NUM_INSTANCES = len(texts)
    features_matrix = np.zeros((NUM_INSTANCES,NUM_FEATURES),dtype=np.int16)
    for i,instance in enumerate(texts):
        feature_vector = np.zeros((len(all_words_idx.keys()),len(all_labels_idx.keys())))
        for lbl in all_labels_idx.keys():
            for word in instance:
                if all_words_idx.get(word,None) is None:
                    continue
                feature_vector[all_words_idx[word]][all_labels_idx[lbl]] += 1
        feature_vector = feature_vector.flatten().tolist()
        features_matrix[i] = feature_vector
    return features_matrix

def get_features_for_label(instance,label,class_labels):
    num_labels = len(class_labels)
    num_feats = len(instance)/num_labels
    feats = np.zeros(len(instance)+1)
    feats[num_feats*label:num_feats*label+num_feats] = instance[num_feats*label:num_feats*label+num_feats]
    return feats

def extract_features_insane(all_words_idx,all_labels_idx,texts):
    features_matrix = []
    for i,instance in enumerate(texts):
        feature_vector = [None]*len(all_labels_idx.keys())
        for lbl in all_labels_idx.keys():
            label_feature_vector = np.zeros((len(all_words_idx.keys()),len(all_labels_idx.keys())))
            for word in instance:
                if all_words_idx.get(word,None) is None:
                    continue
                label_feature_vector[all_words_idx[word]][all_labels_idx[lbl]] += 1
            label_features = np.append(label_feature_vector.flatten(),1)
            feature_vector[all_labels_idx[lbl]] = label_features

        features_matrix.append(feature_vector)
    return features_matrix


"""
Trains a perceptron model with bag of words features and computes the accuracy on the test set

train_texts, train_targets, train_labels are as described in read_dataset above
The same thing applies to the reset of the parameters.

"""
def run_bow_perceptron_classifier(train_texts, train_targets,train_labels,
				dev_texts, dev_targets,dev_labels, test_texts, test_targets, test_labels):
  all_words_idx = extract_all_words(train_texts)
  all_labels_idx = extract_all_labels(train_labels)

  NUM_PARAMS = len(all_words_idx.keys())*len(all_labels_idx.keys())+1
  class_labels = all_labels_idx.values()

  #train_features = extract_features(all_words_idx,all_labels_idx,train_texts)
  #np.save('train_features.npy',train_features)
  train_features = np.load('train_features.npy')
  train_labels = map(lambda e: all_labels_idx[e],train_labels)

  test_features = extract_features(all_words_idx,all_labels_idx,test_texts)
  test_labels = map(lambda e: all_labels_idx[e],test_labels)

  #print train_features.shape,test_features.shape
  #exit()

  theta = train_perceptron(train_features,train_labels,class_labels,NUM_PARAMS)
  test_predictions = test_perceptron(theta,test_features,test_labels,class_labels)
  eval_test = eval_performance(test_labels,test_predictions)
  return ('test-micro=%d%%, test-macro=%d%%' % (int(eval_test[0]*100),int(eval_test[1]*100)))


"""
Trains a naive bayes model with bag of words features  + two additional features
and computes the accuracy on the test set

train_texts, train_targets, train_labels are as described in read_dataset above
The same thing applies to the reset of the parameters.

"""
def run_extended_bow_naivebayes_classifier(train_texts, train_targets,train_labels,
				dev_texts, dev_targets,dev_labels, test_texts, test_targets, test_labels):
	"""
	**Your final implementation of Part 4 with perceptron classifier**
	"""
	pass


"""
Trains a perceptron model with bag of words features  + two additional features
and computes the accuracy on the test set

train_texts, train_targets, train_labels are as described in read_dataset above
The same thing applies to the reset of the parameters.

"""
def run_extended_bow_perceptron_classifier(train_texts, train_targets,train_labels,
				dev_texts, dev_targets,dev_labels, test_texts, test_targets, test_labels):
	"""
	**Your final implementation of Part 4 with perceptron classifier**
	"""
	pass

# Part 1.1
def run_most_frequent_class_classifier(train_texts, train_targets,train_labels,
        dev_texts, dev_targets,dev_labels, test_texts, test_targets, test_labels):
  labels_freq = {}
  for l in train_labels:
      if labels_freq.get(l,None) is None:
          labels_freq[l] = 0
      labels_freq[l] += 1


  most_frequent_label = max(labels_freq.iteritems(), key=operator.itemgetter(1))[0]

  train_pred = [most_frequent_label]*len(train_labels)
  dev_pred = [most_frequent_label]*len(dev_labels)
  assert train_pred[2] == train_labels[2]
  eval_train = eval_performance(train_labels,train_pred)
  eval_dev = eval_performance(dev_labels,dev_pred)
  return ('training-micro=%d%%, training-macro=%d%%, dev-micro=%d%%, dev-macro=%d%%' % (int(eval_train[0]*100),int(eval_train[1]*100),int(eval_dev[0]*100),int(eval_dev[1]*100)))

# Part 1.2
def run_inner_annotator_agreement(train_texts, train_targets,train_labels,
        dev_texts, dev_targets,dev_labels, test_texts, test_targets, test_labels):
    dev_labels_manual, dev_targets_manual, dev_texts_manual = read_dataset('dev_manual')

    return '%.2f' % sklearn.metrics.cohen_kappa_score(dev_labels[:20],dev_labels_manual)

if __name__ == "__main__":
    # reading, tokenizing, and normalizing data
    train_labels, train_targets, train_texts = read_dataset('train')
    dev_labels, dev_targets, dev_texts = read_dataset('dev')
    test_labels, test_targets, test_texts = read_dataset('test')

    #running the classifier
    test_scores = run_bow_perceptron_classifier(train_texts, train_targets, train_labels,
				dev_texts, dev_targets, dev_labels, test_texts, test_targets, test_labels)

    print test_scores
