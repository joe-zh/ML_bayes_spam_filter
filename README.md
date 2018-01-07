# ML_bayes_spam_filter

This project is an improvement based on the naive Bayes Classifer, that detects spam emails with a 99.25% accuracy over test and training files combined (99.5% over training set, and 99% over testing set).

Read more about the paradigm here: https://en.wikipedia.org/wiki/Naive_Bayes_classifier

In short - The original algorithm expands on the Naive Bayes Theorem, which is a probabilistic classifier that assumes the following: Each feature x_i is conditionally independent of every other feature x_j for j not equal to i, given the category C. In implementing the chain rule, this algorithm also implements Laplace smoothing to avoid the likely event of a test word occuring 0 times in the training set, which may cause the total probability to be 0 if not taken into account. 

This project builds upon such naive model, which natively has ~96% accuracy, by implementing several other spam-detection features.

Spam-detection features experimented:
  1.	Preprocessed tokenized method
  2.	Unigram feature categorization
  3.	Bigram feature categorization
  4.	Length feature detection
  5.	Experimentation on different smoothing constants
  
  Training & Test Files provided upon request.
  
