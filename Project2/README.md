## Project 2 - Dependency Parsing

### Introduction:

This project is about understanding data-driven dependency parsing for English. We will work with data annotated and released as part of the CoNLL evaluations. You can download everything as [a tar file](https://myelms.umd.edu/courses/1199410/files/45175834/download?wrap=1).

**Submission**: All the required files (described below) should be uploaded as a single zip file. Include a file team.txt that includes the names of the team members and a contact email. The project is to be done in groups of 3.

**Evaluation**: Your code will be autograded for technical correctness. Please do not change the names of any provided functions or classes within the code, or you will wreak havoc on the autograder. However, the correctness of your implementation -- not the autograder's output -- will be the final judge of your score. If necessary, we will review and grade assignments individually to ensure that you receive due credit for your work.

**Academic Dishonesty**: We will be checking your code against other submissions in the class for logical redundancy. If you copy someone else's code and submit it with minor changes, we will know. These cheat detectors are quite hard to fool, so please don't try. We trust you all to submit your own work only; please don't let us down. If you do, we will pursue the strongest consequences available to us.

**Getting Help**: You are not alone! If you find yourself stuck on something, contact the course staff for help. Office hours, class time, and Piazza are there for your support; please use them. We want these projects to be rewarding and instructional, not frustrating and demoralizing. But, we don't know when or how to help unless you ask. One more piece of advice: if you don't know what a variable is, print it out.

### Learning a graph-based dependency parser (5 points)

As a warm-up, we will walk through the key steps needed to train a graph-based dependency parser. To make things simpler, we produce an undirected dependency graph (rather than a directed graph). This lets us use the [networkx](http://networkx.github.io/documentation/latest/install.html) package to represent and manipulate our graphs.

First, make sure you are running Python2.7 and have networkx installed.

Open the **graphparser.py** file. You will see that we define a simple sentence that will we use as a training instance: "the hairy monster ate tasty little children." We store the sentence words and POS tags, as well as the correct graph representing dependency arcs.

Your task is to complete all the **TODO** items, so that you can run through a complete pass of training for the example sentence.

Once you're done, the first way to check whether your code is doing something reasonable is to make repeated updates on the example sentence. The number of errors should go down, and the predicted graph should improve.

The 2nd check is to run your code in a slightly more realistic setting, and actually use it to train on a larger number of training examples. You can do that by running, for instance:

```python
weights = Weights()
>>> for interation in range(5):
         totalErr = 0.
     for G in iterCoNLL('en.tr100'): 
              totalErr += runOneExample(weights, G, quiet=True)
     print totalErr
```
As you go through more iterations over the data, the error should (roughly) keep dropping.

What should you hand in: **graphparser.py**, after filling in your code for all the TODO items.


### Transition-based dependency parsing (15 Points)

In this section, your task is to implement a transition-based dependency parser and train it on the provided training data.

Inputs you are given:

 - **depeval.py** which will be used to evaluate the performance of the parser. It computes the percentage of words that have the correct head, a quantity that is sometimes called the unlabelled attachment score in CoNLL shared tasks.
 - **en.tr100**, **en.dev** and **en.tst**, which are training, dev and test sets in the CoNLL format we saw in class. Data files contain sentences annotated with dependency arcs, represented with one word per line, and using blank lines to separate sentences. Each word entry contains several tab separated fields. The first one is the position of the word in the sentence, followed by the word itself. We don't use lemmas in this project, so the next useful fields are the 4th and 5th fields which contain two versions of the POS tag of the current word. Finally, the last useful field for this project is the 7th field which, for each word, gives us the position of its head.

What you should hand in: **transparser.py** and **en.tst.out**. The first file is a python script that can be run as follows:

```shell
transparser.py en.tr100 en.tst en.tst.out
```

where

 - "en.tr100" is the training file used to train the underlying classifier
 - "en.tst" is the test file for which we need to predict dependency relations. It is a blind test set. The correct head for each word is not given to you (so the 7th field is "_" for each word).
 - "en.tst.out" is your output file. It should be in the same format as "en.tst", but the 7th field should now contain the position of the head predicted by your parser.

Parser specification: as we have seen in class, there are \*many\* ways to define a transition-based parser. In this problem, you are asked to implement the following:

 - the arc-standard transition sytem
 - the averaged perceptron algorithm to learn weights for predicting the next parsing action
 - a set of basic features to represent each configuration. For now, let's use a small set of feature templates:
  - identity of word at top of the stack
  - identity of word at head of buffer
  - coarse POS of word at top of the stack
  - coarse POS of word at head of buffer
  - pair of words at top of stack and head of buffer
  - pair of coarse POS at top of stack and head of buffer

### Improving performance (10 Points)

Now that you have implemented a parser, your task is to improve its performance. You are welcome to be creative and come up with your own strategies for improvement, as long as (1) you use only the data provided for this assignment, and (2) you do not use existing syntactic parsers beyond what you implement for this assignment.

Here are a few potential directions to think about:

 - Use more training examples (check out the 'en.tr' file as opposed to 'en.tr100')
 - Use more/better features to predict the next parsing action. You can look at the lecture slides and readings for inspiration, as well as the work of [Zhang and Nivre](http://dl.acm.org/citation.cfm?id=2002777).
 - Modify the transition system and/or oracle (e.g., [Goldberg & Nivre, 2012](http://www.aclweb.org/anthology/C12-1059),[2013](https://www.aclweb.org/anthology/Q/Q13/Q13-1033.pdf)).

What you should hand in:

 1. **fancydep.py** your final parser, which should expect the same arguments as **transparser.py**.
 2. **en.tst.fancy.out** The output of your parser on the test file.
 3. **whatwedid.txt**: a text file explaining (briefly) what you tried, why you thought it was worth trying, and what the impact on devset accuracy was (positive or negative.)

5 points of your grade will account for how well you are doing on the test set (proportional to how much improvement you achieve over your baseline accuracy of part 2). And the remaining 5 points will reward the ideas described in the write-up, whether they improved performance or not. 
