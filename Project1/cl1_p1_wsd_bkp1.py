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

    return all_words_idx,all_words

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
def compute_features(all_words_idx,all_labels_idx,instance,label):
    feature_vector = np.zeros((len(all_words_idx.keys()),len(all_labels_idx.keys())))
    for word in instance:
        if all_words_idx.get(word,None) is None:
            #assert False
            continue
        feature_vector[all_words_idx[word]][all_labels_idx[label]] += 1
    return np.append(feature_vector.flatten(),1)

## get the predicted label for a given instance
## the predicted label is the one with the highest dot product of theta*feature_vector
## return the predicted label, the dot product scores for all labels and the features computed for all labels for that instance
def get_predicted_label(all_words_idx,all_labels_idx,inst,theta,precomputed_features=None):
  all_labels_scores = {}
  all_labels_features = {}
  for lbl in all_labels_idx.keys():
      if precomputed_features is not None:
          feat_vec = precomputed_features[lbl]
      else:
          feat_vec = compute_features(all_words_idx,all_labels_idx,inst,lbl)
      assert len(feat_vec) == len(theta)
      all_labels_scores[lbl] = np.dot(feat_vec,theta)
      all_labels_features[lbl] = feat_vec

  predicted_label = max(all_labels_scores.iteritems(), key=operator.itemgetter(1))[0]
  return predicted_label,all_labels_scores,all_labels_features

## train the perceptron by iterating over the entire training dataset
## the algorithm is an implementation of the pseudocode from Slide #23 of:
## http://www.cs.umd.edu/class/fall2016/cmsc723/slides/slides_03.pdf
def train_perceptron(all_words_idx, all_labels_idx, train_texts, train_targets,train_labels,dev_texts, dev_targets,dev_labels,test_texts, test_targets, test_labels):
  theta = np.zeros(len(all_words_idx.keys())*len(all_labels_idx.keys())+1)


  train_set_features = {}
  for i,inst in enumerate(train_texts):
      _,_,feat = get_predicted_label(all_words_idx,all_labels_idx,inst,theta)
      train_set_features[i] = feat

  dev_set_features = {}
  for i,inst in enumerate(dev_texts):
      _,_,feat = get_predicted_label(all_words_idx,all_labels_idx,inst,theta)
      dev_set_features[i] = feat

  test_set_features = {}
  for i,inst in enumerate(test_texts):
      _,_,feat = get_predicted_label(all_words_idx,all_labels_idx,inst,theta)
      test_set_features[i] = feat

  print '# Training Instances:',len(train_texts)

  np.random.seed(1)
  shuffled_indices = np.arange(len(train_texts))
  np.random.shuffle(shuffled_indices)

  num_iterations = 0
  cnt_updates = 0
  m = np.zeros(len(all_words_idx.keys())*len(all_labels_idx.keys())+1)
  for i in shuffled_indices:
      inst = train_texts[i]
      actual_label = train_labels[i]
      predicted_label,_,all_labels_features = get_predicted_label(all_words_idx,all_labels_idx,inst,theta,precomputed_features=train_set_features[i])

      if predicted_label != actual_label:
          cnt_updates += 1
          theta = theta + all_labels_features[actual_label] - all_labels_features[predicted_label]
          m = m + theta
      '''
      if num_iterations % 10 == 0: #test_set_features

          crt_avg_theta = m/cnt_updates
          #dev_predictions = test_perceptron(crt_avg_theta,all_words_idx,all_labels_idx,dev_texts,precomputed_features=dev_set_features)
          #eval_dev = eval_performance(dev_labels,dev_predictions)
          #print ('%d - test-micro=%d%%, test-macro=%d%%' % (num_iterations,int(eval_dev[0]*100),int(eval_dev[1]*100)))

          test_predictions = test_perceptron(crt_avg_theta,all_words_idx,all_labels_idx,test_texts,precomputed_features=test_set_features)
          eval_test = eval_performance(test_labels,test_predictions)
          print ('%d - test-micro=%d%%, test-macro=%d%%' % (num_iterations,int(eval_test[0]*100),int(eval_test[1]*100)))
          #train_predictions = test_perceptron(theta,all_words_idx,all_labels_idx,train_texts)
          #eval_train = eval_performance(train_labels,train_predictions)
          #print ('test-micro=%d%%, test-macro=%d%%' % (int(eval_train[0]*100),int(eval_train[1]*100)))
      '''
      num_iterations += 1
      #print np.where(theta != 0)[0]
      #if num_iterations == 200:
      #    break
  theta = m/cnt_updates
  print '# Updates:',cnt_updates
  return theta

## return the predictions of the perceptron on a test set
def test_perceptron(theta, all_words_idx, all_labels_idx, test_texts,precomputed_features=None):
  predictions = []
  for i in range(len(test_texts)):
      inst = test_texts[i]
      predicted_label,_,_ = get_predicted_label(all_words_idx,all_labels_idx,inst,theta,precomputed_features)
      predictions.append(predicted_label)
  return predictions

"""
Trains a perceptron model with bag of words features and computes the accuracy on the test set

train_texts, train_targets, train_labels are as described in read_dataset above
The same thing applies to the reset of the parameters.

"""
def run_bow_perceptron_classifier(train_texts, train_targets,train_labels,
				dev_texts, dev_targets,dev_labels, test_texts, test_targets, test_labels):
  all_words_idx,all_words_train = extract_all_words(train_texts)
  #_,all_words_test = extract_all_words(test_texts,prev_set=all_words_train)
  #all_words_idx,_ = extract_all_words(dev_texts,prev_set=all_words_test)

  all_labels_idx = extract_all_labels(train_labels)
  theta = train_perceptron(all_words_idx,all_labels_idx,train_texts, train_targets,train_labels,dev_texts, dev_targets,dev_labels,test_texts, test_targets, test_labels)
  test_predictions = test_perceptron(theta,all_words_idx,all_labels_idx,test_texts)
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