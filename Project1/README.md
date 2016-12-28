## Project 1 - Implementing Word Sense Disambiguation System.


### Instructions:

 - The project is to be done in groups of 3.
 - Your submission will include 1 zip file that includes:
  - your implementation inside the provided code skeleton,
  - your answers to the questions below (using the provided answers template),
  - predicted labels using your naive bayes implementation (Q4 in part 2). The file name should be q4p2.txt,
  - predicted labels using your perceptron implementation (Q3 in part 3). The file name should be q3p3.txt,
  - predicted labels using each of your naive bayes and perceptron implementations with the two additional features (Q4 in part 4). The file names should be q4p4_nb.txt and q4p4_pn.txt, respectively.

The format of the submitted files should be one label per line. So for the provided test set, each predictions fie should have 623 lines. In total, make sure that your zip file contains 6 files (4 for predictions + python implementation + answers to the questions).

Please do not change the provided interfaces and answer formats as your code will be tested automatically, and your answers will be processed (semi)-automatically. You will not get credit if we cannot grade your submission due to incorrect formatting and modifications to the expected interface. The submission should be made by only one member from each group. All members of each group will get the same grade.

 - In order to get the provided partial implementation to work, you need to add nltk and sklearn to your python installation. Just run:
 ```shell
 pip install nltk
 pip install sklearn
 ```
 - For the questions that ask you to write your own implementation of Naive Bayes and the perceptron, you are expected to write your own implementation :) this means that you cannot directly use existing implementations of these classifiers (e.g. from nltk, or sklearn). However, you can use data structures, such as collection.Counter.

 
### Project Files:

 - Dataset: wsd_line_dataset.zip
 - Code Skeleton: cl1_p1_wsd.py
 - Answers Template: answers_template.txt
 

#### Part 1: How difficult is the problem we are trying to solve? Establishing a baseline and an upper bound 

 1. Report  the accuracy of the “most frequent class” baseline on the “line” training set and dev set? **[1 point]**
 2. Have one team member annotate the first 20 examples in the dev set (without looking at the gold annotations provided). Using Cohen’s Kappa, report the inter annotator agreement with the reference annotation. **[1 point]**

#### Part 2: Implement a naïve Bayes classifier to predict sense s given example X

 1. Write code to read in the training sentences and collect counts c(s) (number of occurrences  in class s) and c(s,w) for s ∈ S and all context words in w. Report c(s) and c(s,w) for s ∈{cord, division, formation, phone, product, text} and w ∈ {time, loss, export}. **[1 point]**
 2. Write code to compute the probabilities p(s) and p(w∣s) for all s,w. Train on the training set. Report s ∈ {cord, division, formation, phone, product, text} and w ∈ {time, loss, export}. **[1 point]**
 3. Write code to read in a test sentence and compute the probabilities P(s∣x) for each s. Report these probabilities for the first line of the dev set **[1 point]**
 4. Run the classifier on the test set and report your accuracy, **[2 points]** which should be at least 77% (in terms of the micro-averaged f1-score) **[3 points]**
 5. Describe any implementation choices you had to make (e.g., smoothing, log-probabilities). **[2 points]**

#### Part 3: Implement the perceptron algorithm

 1. Report what weight updates you would make (starting from all zero weights) on the first line of on the training set. Each weight corresponds to a feature.Only report the subset of weights that got updated together with their new values after the update. Your answer should look like: "export:2.41, game: -3.23, .. etc". which means that the weight that corresponds to the feature 'has export as a token' got updated from zero to 2.41, ad the weight that corresponds to the feature 'has game as a token' got updated from zero to -3.23. **[1 point]**
 2. Now run the trainer on the whole training set. At each iteration, report your accuracy on the training set. **[2 points]**
 3. Run the classifier on the test set and report your accuracy, **[2 points]** which should be at least 80%. **[3 points]**
 4. Describe any implementation choices you had to make (e.g., random shuffling of examples, learning rate, number of iterations, weight averaging). **[2 points]** 

#### Part 4: Experiment with (at least) two new kinds of features

 1. Extend the code you wrote to construct a bag of words to include an additional kind of feature besides words. You can try anything you like (your feature does not have to depend on the identity of the tokens). Describe your new features. **[2 points]**
 2. Report your new accuracy for both naïve Bayes and perceptron **[1 point]**
 3. Briefly write down your conclusions from this experiment. **[1 point]**
 4. Do the same thing for another kind of feature. **[4 points]**
 
