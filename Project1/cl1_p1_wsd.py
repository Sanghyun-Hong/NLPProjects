import numpy as np
import operator

# SHHONG: custom modules imported
import json
import random
import itertools
from math import pow, log
from collections import Counter


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
	if subset in ['train', 'dev', 'test', 'dev_manual']:
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
      dev_texts, dev_targets, dev_labels, test_texts, test_targets, test_labels,
      improved=True, alpha=0.5, silent=False):

  # Part 2.1 (c_s/c_sw)
  c_s  = dict.fromkeys(set(train_labels), 0)
  multiples = list(itertools.product(c_s.keys(), ['time', 'loss', 'export']))
  c_sw = dict.fromkeys(multiples, 0)
  t_w  = [each_word for each_text in train_texts for each_word in each_text]
  multiples = list(itertools.product(c_s.keys(), t_w))
  t_sw = dict.fromkeys(multiples, 0)
  for idx, label in enumerate(train_labels):
    cur_text = train_texts[idx]
    # compute c_s
    c_s[label] += len(cur_text)
    # compute c_sw
    time_cnt = cur_text.count('time')
    loss_cnt = cur_text.count('loss')
    export_cnt = cur_text.count('export')
    c_sw[(label, 'time')] += time_cnt
    c_sw[(label, 'loss')] += loss_cnt
    c_sw[(label, 'export')] += export_cnt
    # compute t_sw (total occurances): of (label, word): occurances
    for each_word in cur_text:
      t_sw[(label, each_word)] += 1

  # total # of distinct words: will be used for smoothing
  t_dw = Counter(t_w)

  if not silent:
    print '{:<11} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10} |'.\
          format('s', 'cord', 'division', 'formation', 'phone', 'product', 'text')
    print '------------------------------------------------------------------------------------------'
    print '{:<11} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10} |'.\
          format('c(s)', c_s['cord'], c_s['division'], c_s['formation'], c_s['phone'], c_s['product'], c_s['text'])
    print '{:<11} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10} |'.\
          format('c(s,time)', c_sw[('cord', 'time')], c_sw[('division', 'time')], c_sw[('formation', 'time')], \
                 c_sw[('phone', 'time')], c_sw[('product', 'time')], c_sw[('text', 'time')])
    print '{:<11} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10} |'.\
          format('c(s,loss)', c_sw[('cord', 'loss')], c_sw[('division', 'loss')], c_sw[('formation', 'loss')], \
                 c_sw[('phone', 'loss')], c_sw[('product', 'loss')], c_sw[('text', 'loss')])
    print '{:<11} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10} |'.\
          format('c(s,export)', c_sw[('cord', 'export')], c_sw[('division', 'export')], c_sw[('formation', 'export')], \
                 c_sw[('phone', 'export')], c_sw[('product', 'export')], c_sw[('text', 'export')])
    print '------------------------------------------------------------------------------------------'
    print ' total distinct words: %d ' % (len(t_dw.keys()))

  # Part 2.2 (p_s/p_ws)
  total_occurances = float(sum(c_s.values()))
  label_count      = Counter(train_labels)
  p_s  = {key: (value / float( sum( label_count.values() )) ) for key, value in label_count.iteritems()}
  if improved:
    p_ws = {key: ( (value + alpha) / \
                   (float(c_s[key[0]]) + alpha*len(t_dw.keys())) ) \
                   for key, value in c_sw.iteritems()}
    t_ws = {key: ( (value + alpha) / \
                   (float(c_s[key[0]]) + alpha*len(t_dw.keys())) ) \
                   for key, value in t_sw.iteritems()}
  else:
    p_ws = {key: (value / float(c_s[key[0]])) for key, value in c_sw.iteritems()}
    t_ws = {key: (value / float(c_s[key[0]])) for key, value in t_sw.iteritems()}

  # normalization steps
  norm_denominators = {
      'time': 0.0,
      'loss': 0.0,
      'export': 0.0
    }
  for key, value in p_ws.iteritems():
    norm_denominators[key[1]] += value
  p_ws_norm = {key: (value / norm_denominators[key[1]]) for key, value in p_ws.iteritems()}
  p_ws = p_ws_norm

  if not silent:
    print '------------------------------------------------------------------------------------------'
    print '{:<11} | {:<10.8f} | {:<10.8f} | {:<10.8f} | {:<10.8f} | {:<10.8f} | {:<10.8f} |'.\
          format('p(s)', p_s['cord'], p_s['division'], p_s['formation'], p_s['phone'], p_s['product'], p_s['text'])
    print '{:<11} | {:<10.8f} | {:<10.8f} | {:<10.8f} | {:<10.8f} | {:<10.8f} | {:<10.8f} |'.\
          format('p(time|s)', p_ws[('cord', 'time')], p_ws[('division', 'time')], p_ws[('formation', 'time')], \
                 p_ws[('phone', 'time')], p_ws[('product', 'time')], p_ws[('text', 'time')])
    print '{:<11} | {:<10.8f} | {:<10.8f} | {:<10.8f} | {:<10.8f} | {:<10.8f} | {:<10.8f} |'.\
          format('p(loss|s)', p_ws[('cord', 'loss')], p_ws[('division', 'loss')], p_ws[('formation', 'loss')], \
                 p_ws[('phone', 'loss')], p_ws[('product', 'loss')], p_ws[('text', 'loss')])
    print '{:<11} | {:<10.8f} | {:<10.8f} | {:<10.8f} | {:<10.8f} | {:<10.8f} | {:<10.8f} |'.\
          format('p(export|s)', p_ws[('cord', 'export')], p_ws[('division', 'export')], p_ws[('formation', 'export')], \
                 p_ws[('phone', 'export')], p_ws[('product', 'export')], p_ws[('text', 'export')])

  # Part 2.3 (p_sxd, on the 1st line on test set)
  p_sxd  = dict.fromkeys(c_s.keys(), 0.0)
  lp_sxd = dict.fromkeys(c_s.keys(), 0.0)
  cur_text = dev_texts[0]
  for key in p_sxd.keys():
    # compute p for each class
    if improved:
      tp_sxd  = p_s[key]
      tlp_sxd = log(p_s[key])
    else:
      tp_sxd  = p_s[key]
    # compute for each word
    for each_word in cur_text:
      if t_ws.has_key((key, each_word)):
        if improved:
          tp_sxd  *= t_ws[(key, each_word)]
          tlp_sxd += log(t_ws[(key, each_word)])
        else:
          tp_sxd  *= t_ws[(key, each_word)]
    # add to the dict
    if improved:
      p_sxd[key]  = tp_sxd
      lp_sxd[key] = tlp_sxd
    else:
      p_sxd[key]  = tp_sxd

  if not silent:
    print '------------------------------------------------------------------------------------------'
    print ' %s | %s | %s | %s | %s | %s | %s |' % \
          ('p(s|X)', p_sxd['cord'], p_sxd['division'], p_sxd['formation'], \
                     p_sxd['phone'], p_sxd['product'], p_sxd['text'])
    print '------------------------------------------------------------------------------------------'
    print ' 1st label in dev    : %s ' % (dev_labels[0])
    print ' 1st text  in dev[:5]: %s ' % (dev_texts[0][:5])
    if improved:
      print '------------------------------------------------------------------------------------------'
      print '{:<11} | {:<10.8f} | {:<10.8f} | {:<10.8f} | {:<10.8f} | {:<10.8f} | {:<10.8f} |'.\
            format('log(p(s|X))', lp_sxd['cord'], lp_sxd['division'], lp_sxd['formation'], \
                                  lp_sxd['phone'], lp_sxd['product'], lp_sxd['text'])

  # Part 2.4: compute all the prob on the test dataset
  p_sx = list()
  for idx, text in enumerate(test_texts):
    t_prob = dict.fromkeys(c_s.keys(), 0.0)
    for key in t_prob.keys():
      # compute p for each class
      if improved:
        tp_sxt  = log(p_s[key])
      else:
        tp_sxt  = p_s[key]
      for each_word in text:
        if t_ws.has_key((key, each_word)):
          if improved:
            tp_sxt += log(t_ws[(key, each_word)])
          else:
            tp_sxt *= t_ws[(key, each_word)]
      # add to the dict
      t_prob[key] = tp_sxt
    # add dict to the entire list
    p_sx.append(t_prob)

  # Part 2.4 (run the classifier for all)
  labels_predicted = list()
  for idx, label in enumerate(test_labels):
    maximum_probs    = max(p_sx[idx].values())
    label_prediction = [key for key, value in p_sx[idx].iteritems() if value == maximum_probs]
    label_prediction = random.choice(label_prediction)
    # based on the prob
    labels_predicted.append(label_prediction)
  naivebayes_performance = eval_performance(test_labels, labels_predicted)

  # Part 2.5 (do more tuning for the classifier)
  #  - Laplace smoothing
  #  - Log likelihoods
  if not silent:
    print '------------------------------------------------------------------------------------------'
  return 'Naive Bayes: micro/macro = [%.2f, %.2f] @ (alpha: %s)' % \
          (naivebayes_performance[0]*100, naivebayes_performance[1]*100, alpha)


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

## construct a bow feature matrix for a set of instances
## the returned matrix has the size NUM_INSTANCES X NUM_FEATURES
def extract_features(all_words_idx,all_labels_idx,texts):
    NUM_FEATURES = len(all_words_idx.keys())
    NUM_INSTANCES = len(texts)
    features_matrix = np.zeros((NUM_INSTANCES,NUM_FEATURES))
    for i,instance in enumerate(texts):
        for word in instance:
            if all_words_idx.get(word,None) is None:
                continue
            features_matrix[i][all_words_idx[word]] += 1
    return features_matrix

## compute the feature vector for a set of words and a given label
## the features are computed as described in Slide #19 of:
## http://www.cs.umd.edu/class/fall2016/cmsc723/slides/slides_02.pdf
def get_features_for_label(instance,label,class_labels):
    num_labels = len(class_labels)
    num_feats = len(instance)
    feats = np.zeros(len(instance)*num_labels+1)
    assert len(feats[num_feats*label:num_feats*label+num_feats]) == len(instance)
    feats[num_feats*label:num_feats*label+num_feats] = instance
    return feats

## get the predicted label for a given instance
## the predicted label is the one with the highest dot product of theta*feature_vector
## return the predicted label, the dot product scores for all labels and the features computed for all labels for that instance
def get_predicted_label(inst,class_labels,theta):
  all_labels_scores = {}
  all_labels_features = {}
  for lbl in class_labels:
      feat_vec = get_features_for_label(inst,lbl,class_labels)
      assert len(feat_vec) == len(theta)
      all_labels_scores[lbl] = np.dot(feat_vec,theta)

  predicted_label = max(all_labels_scores.iteritems(), key=operator.itemgetter(1))[0]
  return predicted_label

## train the perceptron by iterating over the entire training dataset
## the algorithm is an implementation of the pseudocode from Slide #23 of:
## http://www.cs.umd.edu/class/fall2016/cmsc723/slides/slides_03.pdf
def train_perceptron(train_features,train_labels,class_labels,num_features):
  NO_MAX_ITERATIONS = 20
  np.random.seed(0)

  theta = np.zeros(num_features)

  print '# Training Instances:',len(train_features)

  num_iterations = 0
  cnt_updates_total = 0
  cnt_updates_prev = 0
  m = np.zeros(num_features)
  print '# Total Updates / # Current Iteration Updates:'
  for piter in range(NO_MAX_ITERATIONS):
      shuffled_indices = np.arange(len(train_features))
      np.random.shuffle(shuffled_indices)
      cnt_updates_crt = 0
      for i in shuffled_indices:
          inst = train_features[i]
          actual_label = train_labels[i]
          predicted_label = get_predicted_label(inst,class_labels,theta)

          if predicted_label != actual_label:
              cnt_updates_total += 1
              cnt_updates_crt += 1
              theta = theta + get_features_for_label(inst,actual_label,class_labels) - get_features_for_label(inst,predicted_label,class_labels)
              m = m + theta

          num_iterations += 1

      print cnt_updates_total,'/',cnt_updates_crt
      if cnt_updates_crt == 0:
          break

  theta = m/cnt_updates_total
  print '# Iterations:',piter
  print '# Iterations over instances:',num_iterations
  print '# Total Updates:',cnt_updates_total
  return theta

## return the predictions of the perceptron on a test set
def test_perceptron(theta,test_features,test_labels,class_labels):
  predictions = []
  for inst in test_features:
      predicted_label = get_predicted_label(inst,class_labels,theta)
      predictions.append(predicted_label)
  return predictions

"""
Trains a perceptron model with bag of words features and computes the accuracy on the test set

train_texts, train_targets, train_labels are as described in read_dataset above
The same thing applies to the reset of the parameters.

"""
def run_bow_perceptron_classifier(train_texts, train_targets,train_labels,
				dev_texts, dev_targets,dev_labels, test_texts, test_targets, test_labels):
  all_words_idx = extract_all_words(train_texts)
  all_labels_idx = extract_all_labels(train_labels)

  num_features = len(all_words_idx.keys())*len(all_labels_idx.keys())+1
  class_labels = all_labels_idx.values()

  train_features = extract_features(all_words_idx,all_labels_idx,train_texts)
  train_labels = map(lambda e: all_labels_idx[e],train_labels)
  test_features = extract_features(all_words_idx,all_labels_idx,test_texts)
  test_labels = map(lambda e: all_labels_idx[e],test_labels)

  for l in class_labels:
        inst = train_features[0]
        ffl = get_features_for_label(inst,l,class_labels)
        assert False not in (inst == ffl[l*len(inst):(l+1)*len(inst)])

  theta = train_perceptron(train_features,train_labels,class_labels,num_features)
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


## this feature is just a random number generated for each instance
def get_feature_A(train_texts, train_targets,train_labels,
        dev_texts, dev_targets,dev_labels, test_texts, test_targets, test_label):

    np.random.seed(0)

    train_feature_vector = np.random.random_sample((len(train_texts),))
    dev_feature_vector = np.random.random_sample((len(dev_texts),))
    test_feature_vector = np.random.random_sample((len(test_texts),))

    return train_feature_vector,dev_feature_vector,test_feature_vector

## this feature encodes the number of distinct words in each instance
def get_feature_B(train_texts, train_targets,train_labels,
        dev_texts, dev_targets,dev_labels, test_texts, test_targets, test_label):
    train_feature_vector = np.zeros(len(train_texts))
    dev_feature_vector = np.zeros(len(dev_texts))
    test_feature_vector = np.zeros(len(test_texts))

    for i,text in enumerate(train_texts):
        nw = len(set(text))
        train_feature_vector[i] = nw

    for i,text in enumerate(dev_texts):
        nw = len(set(text))
        dev_feature_vector[i] = nw

    for i,text in enumerate(test_texts):
        nw = len(set(text))
        test_feature_vector[i] = nw

    return train_feature_vector,dev_feature_vector,test_feature_vector


"""
Trains a perceptron model with bag of words features  + two additional features
and computes the accuracy on the test set

train_texts, train_targets, train_labels are as described in read_dataset above
The same thing applies to the reset of the parameters.

"""
def run_extended_bow_perceptron_classifier(train_texts, train_targets,train_labels,
        dev_texts, dev_targets,dev_labels, test_texts, test_targets, test_labels):

  RUN_EXP_A = False # set to False for running on feature B

  if RUN_EXP_A:
      train_new_feature_vector,dev_new_feature_vector,test_new_feature_vector = get_feature_A(train_texts, train_targets,train_labels, dev_texts, dev_targets,dev_labels, test_texts, test_targets, test_labels)
  else:
      train_new_feature_vector,dev_new_feature_vector,test_new_feature_vector = get_feature_B(train_texts, train_targets,train_labels, dev_texts, dev_targets,dev_labels, test_texts, test_targets, test_labels)

  all_words_idx = extract_all_words(train_texts)
  all_labels_idx = extract_all_labels(train_labels)

  num_features = (len(all_words_idx.keys())+1)*len(all_labels_idx.keys())+1
  class_labels = all_labels_idx.values()

  train_features = extract_features(all_words_idx,all_labels_idx,train_texts)
  train_labels = map(lambda e: all_labels_idx[e],train_labels)
  test_features = extract_features(all_words_idx,all_labels_idx,test_texts)
  test_labels = map(lambda e: all_labels_idx[e],test_labels)

  train_features = np.c_[train_features, train_new_feature_vector]
  test_features = np.c_[test_features, test_new_feature_vector]

  for l in class_labels:
        inst = train_features[0]
        ffl = get_features_for_label(inst,l,class_labels)
        assert False not in (inst == ffl[l*len(inst):(l+1)*len(inst)])

  theta = train_perceptron(train_features,train_labels,class_labels,num_features)
  test_predictions = test_perceptron(theta,test_features,test_labels,class_labels)
  eval_test = eval_performance(test_labels,test_predictions)
  return ('test-micro=%d%%, test-macro=%d%%' % (int(eval_test[0]*100),int(eval_test[1]*100)))



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


"""
    Main (able to change the classifier to other ones)
"""
if __name__ == "__main__":
    # reading, tokenizing, and normalizing data
    train_labels, train_targets, train_texts = read_dataset('train')
    dev_labels, dev_targets, dev_texts = read_dataset('dev')
    test_labels, test_targets, test_texts = read_dataset('test')

    #running the classifier
    test_scores = run_extended_bow_perceptron_classifier(train_texts, train_targets, train_labels,
        dev_texts, dev_targets, dev_labels, test_texts, test_targets, test_labels)

    print test_scores

