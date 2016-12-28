## Project 3 - Semantic Textual Similarity

### Instructions:

 1. The project is to be done in groups of 3.
 2. In addition to the provided training datasets, you are allowed to use any available resources except for additional labeled data. Specifically, this means you cannot use examples of sentence pairs with a STS score beyond those given in this project. But you can use e.g. pretrained word embeddings, pretrained models for sentence representations, or unlabeled texts. If you are not sure whether you can use a certain resource or not, please ask on Piazza. 
 3. You are not allowed to reuse an existing implementation of an STS system. You are welcome to implement your own version of published systems, use NLP toolkits such as NLTK, CoreNLP, SpaCy, .. etc and ML toolkits such as scikit-learn, theano, .. etc
 4. Your submission will include 1 zip file that includes: (only one member of the team should make the submission)
  1. Your implementation.
  2. downloads.sh file that downloads all the resources you have decided to use for your methods (e.g. pretrained word embeddings)
  3. STS output files One file per subset. The scores are to be normalized to the range 0-5.
  4. team.txt that includes the names of the group members
  5. writeup.pdf describing your methods, result analysis and  implementation decisions  (Please see the expected format below)

### Overview

In this project you are going to work on the Semantic Textual Similarity task where you will develop and analyze methods that predict the similarity score (in the range 0-5) between sentence pairs. For example

```
sentence 1: render one language in another language
sentence 2: restate (words) from one language into another language.
score: 5.0
```
```
sentence 1: nations unified by shared interests, history or institutions
sentence 2: a group of nations having common interests.
score: 3.25
```
```
sentence 1: travel in or be shaped like a zigzag
sentence 2: bend into the shape of a crank.
score: 0.75
```

We will focus on the monolingual settings where both sentences are in English. You can also work on an optional part (for extra credit) where the input is one sentence in English and the other is in Spanish.

**Background**: [The semantic textual similarity (STS) task](http://alt.qcri.org/semeval2017/task1/) is a shared task that is being organized each year since 2012 as part of [SemEval workshop.](http://alt.qcri.org/semeval2017/) Each year, researchers are invited to build new systems for comparing sentence-level meanings, and the systems are evaluated on new datasets each year. In this project, we are using the test sets of 2016 for our evaluation. See (Agirre *et al.*, 2016) for a report about the submissions made for the 2016 task. 


**Provided Datasets**: The attached [zip file](https://myelms.umd.edu/courses/1199410/files/45441346/download?wrap=1) includes:

 1. Training dataset (under train) which consists of the training set of 2012 and all the test sets of 2012-2015. Each subset of sentence pairs has a corresponding input file (tab separated sentence pair) and a gold scores file.
 2. Test sets (under test) which consists of 5 subsets of sentence pairs (the gold scores are not provided). These are the subsets that will be used for evaluation.
 3. Crosslingual test sets (under xling_test) which consists of 2 subset of crosslingual sentence pairs that are used for the part 3.
 4. Crosslingual dev set (under xling_dev) which is a small subset of 103 sentence pairs that you can use as a dev set.
 5. correlation-noconfidence.pl which is the script we will be using for evaluating your predictions. The evaluation is based on the correlation between the scores you will produce and the gold scores. We will use the average of the 5 correlation scores that correspond to the 5 test subsets for part 1 and part 2, and the average of the 2 correlation scores that correspond to the 2 crosslingual test subsets for part 3.


### Part 1: Establish and Analyze a Baseline (15 Points)

Our baseline will compare the 2 sentence in the input by embedding each sentence into a single embedding, and computing cosine similarity between the two embeddings. We will use Paragram embeddings (Wieting et al., 2016) , which computes sentence-level vector representations by averaging the embeddings of the individual tokens. the pretrained embeddings are available on the [author's website](http://ttic.uchicago.edu/~wieting/).

Your implementation of that part should be inside a file called predict_baseline.py that can be run as

```shell
python predict_baseline.py input_sentence_pairs.txt output_scores.txt
```

where *input_sentence_pairs.txt* has the same format as the input sentences provided in the training and testing sets and *output_scores.txt* is an output file of the scores that you will produce. *output_scores.txt* should be formatted as one score for each sentence pair per line.

Next, examine the output on a dev set (that you can sample from the provided training data). Identify 2 categories of examples for which the baseline is accurate, Identify 4 categories of examples for which the baseline is incorrect. For each of the error categories, explain why the baseline is unable to handle them better.provide an error analysis of the kind of mistakes that the method makes. Based on your understanding of how sentences are being represented,  provide explanations about why the method is making each kind of mistake. 

Implementing the baseline correctly (your average correlation should be at least 0.73) is worth 5 points, and the analysis part is worth 10 points (2 points for correct predictions, 4 points for each error category, 4 points for explanation for each error category).


### Part 2: Develop your Own Method (25 Points)

Develop your own method for solving the STS task. 10 points of your grade will be based on how much you improve over the baseline (2 points for each 0.01 of improvement to the average correlation up to 10 points), and the other 15 points will be based on the description of what you did.

What can you do to improve performance?  You might want to consider using the provided training data (the baseline does not use it) You might want to look at the literature on monolingual word alignment (e.g. Yao *et al.*, 2013) and the descriptions of SemEval 2016 systems (e.g., Sultan *et al.*) There is a recent literature (with potentially available code) on using deep learning models to directly compute text similarity (e.g. Tai *et al.*, 2015) or learn sentence representations (e.g. Hill *et al.*, 2016).  You are welcome to try your own ideas as well!


### [Optional] Part 3: Crosslingual Extension 

Propose and implement a method for estimating the similarity score for a Spanish-English sentence pair. You might want to consider estimating the scores based on parallel text alignment methods you have seen in class (e.g. [the Berkeley Aligner](https://code.google.com/archive/p/berkeleyaligner/)) and/or bilingual word embeddings. See (Upadhyay *et al.*, 2016) for a survey. You can also reduce the task to a monolingual task by automatically translating the Spanish sentence into English. You are not allowed to do manual translation. You can also come up with totally different methods.

You will get one point of extra credit for each 0.01 of improvement you achieve over 0.8


### What to Submit:

**For part 1**, provide the predict_baseline.py script and its output for the test set.

**For parts 2 and part 3**, provide

 1. training code/scripts. If your training takes a significant amount of resources (time,memory, ..etc), please provide your trained model and any intermediate data structures (e.g. word alignments) required to reproduce your results. 
 2. prediction scripts that can be run as
```shell
python predict_fancy_monolingual.py input_sentence_pairs.txt output_scores.txt
```
```shell
python predict_fancy_crosslingual.py input_sentence_pairs.txt output_scores.txt
```

**Writeup**: You are required to submit a short report (at most 4 pages) describing the STS methods you have implemented and their results. The writeup should include:

 1. Any data preprocessing decision you made
 2. Your own description of the baseline and he decision you made to implement it
 3. Your baseline results with the details of the analysis you did for part 1
 4. The description of your method(s) and the results you have obtained
 5. An error analysis of your best performing results
 6. What else you could have tried if you had more time 


#### References

 - Agirre, Eneko, et al. "Semeval-2016 task 1: Semantic textual similarity, monolingual and cross-lingual evaluation." SemEval, 2016.
 - Wieting, John, et al. "Towards universal paraphrastic sentence embeddings." ICLR, 2016.
 - Yao, Xuchen, et al. "A Lightweight and High Performance Monolingual Word Aligner." ACL, 2013.
 - Sultan, Md Arafat, Steven Bethard, and Tamara Sumner. "DLS@ CU at SemEval-2016 Task 1: Supervised Models of Sentence Similarity." Proceedings of SemEval, 2016.
 - Hill, Felix, Kyunghyun Cho, and Anna Korhonen. "Learning distributed representations of sentences from unlabelled data." NAACL, 2016.
 - Tai, Kai Sheng, Richard Socher, and Christopher D. Manning. "Improved semantic representations from tree-structured long short-term memory networks." ACL, 2015. 
 - Upadhyay, Shyam, et al. "Cross-lingual Models of Word Embeddings: An Empirical Comparison." ACL, 2016.
