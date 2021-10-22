import os, math


class NaiveBayesClassifier:
    """Code for a bag-of-words Naive Bayes classifier.
    """

    def __init__(self, train_dir='data/train', REMOVE_STOPWORDS=False):
        self.REMOVE_STOPWORDS = REMOVE_STOPWORDS
        self.stopwords = set([l.strip() for l in open('english.stop')])
        self.classes = os.listdir(train_dir)
        self.train_data = {c: os.path.join(train_dir, c) for c in self.classes}
        self.vocabulary = set([])
        self.logprior = {}
        self.loglikelihood = {}  # keys should be tuples in the form (w, c)

    def train(self):
        """Train the Naive Bayes classification model, following the pseudocode for
        training given in Figure 4.2 of SLP Chapter 4. 

        Note that self.train_data contains the paths to training data files. 
        To get all the documents for a given training class c in a list, you can use:
            c_docs = open(self.train_data[c]).readlines()

        You can get words with simply `words = doc.split()`

        Remember to account for whether the self.REMOVE_STOPWORDS flag is set or not;
        if it is True then the stopwords in self.stopwords should be removed whenever
        they appear.

        When converting from the pseudocode, consider how many loops over the data you
        will need to properly estimate the parameters of the model, and what intermediary
        variables you will need in this function to store the results of your computations.

        Parameters
        ----------
        None (reads training data from self.train_data)

        Returns
        -------
        None (updates class attributes self.vocabulary, self.logprior, self.loglikelihood)
        """
        # >>> YOUR ANSWER HERE

        fake_docs = []
        fake_words = []
        fake_words_freq = {}
        real_docs = []
        real_words = []
        real_words_freq = {}

        # load fake data of the training dataset, store the docs and words
        fake_data = open(self.train_data['fake']).readlines()
        for sentence in fake_data:
            preprocess_sentence = sentence.strip()
            fake_docs.append(preprocess_sentence)
            fake_words.extend(preprocess_sentence.split())

        # load real data of the training dataset, store the docs, words and word frequencies.
        real_data = open(self.train_data['real']).readlines()
        for sentence in real_data:
            preprocess_sentence = sentence.strip()
            real_docs.append(preprocess_sentence)
            real_words.extend(preprocess_sentence.split())

        # remove stop words if necessary
        if self.REMOVE_STOPWORDS:
            fake_words = [word for word in fake_words if word not in self.stopwords]
            real_words = [word for word in real_words if word not in self.stopwords]

        # calculate all words' frequency
        for word in fake_words:
            self.vocabulary.add(word)
            fake_words_freq[word] = fake_words_freq.get(word, 0) + 1
        for word in real_words:
            self.vocabulary.add(word)
            real_words_freq[word] = real_words_freq.get(word, 0) + 1

        # pre-calculate the number of all docs, the number of docs per class and words frequency per class for
        # calculation in the training loop.
        n_doc = len(fake_docs) + len(real_docs)
        n_class = {'fake': len(fake_docs), 'real': len(real_docs)}
        big_doc_dict = {'fake': fake_words_freq, 'real': real_words_freq}
        fake_words_num = 0
        real_words_num = 0
        for w in self.vocabulary:
            fake_words_num += fake_words_freq.get(w, 0)
            real_words_num += real_words_freq.get(w, 0)
        words_frequency_per_class = {'fake': fake_words_num, 'real': real_words_num}

        # Training
        for c in self.classes:
            self.logprior[c] = math.log(n_class[c] / n_doc)
            for w in self.vocabulary:
                count_w_c = big_doc_dict[c].get(w, 0)
                log_likelihood = math.log((count_w_c + 1) / (len(self.vocabulary) + words_frequency_per_class[c]))
                self.loglikelihood[(w, c)] = log_likelihood
        # >>> END YOUR ANSWER

    def score(self, doc, c):
        """Return the log-probability of a given document for a given class,
        using the trained Naive Bayes classifier. 

        This is analogous to the inside of the for loop in the TestNaiveBayes
        pseudocode in Figure 4.2, SLP Chapter 4.

        Parameters
        ----------
        doc : str
            The text of a document to score.
        c : str
            The name of the class to score it against.

        Returns
        -------
        float
            The log-probability of the document under the model for class c.
        """
        # >>> YOUR ANSWER HERE
        # the inner loop in the TEST NAIVE BAYES, sum up the logprior of the class and all words' loglikelihood
        sum = self.logprior[c]
        words = doc.split()
        for w in words:
            if w in self.vocabulary:
                sum += self.loglikelihood[(w, c)]
        return sum
        # >>> END YOUR ANSWER

    def predict(self, doc):
        """Return the most likely class for a given document under the trained classifier model.
        This should be only a few lines of code, and should make use of your self.score function.

        Consider using the `max` built-in function. There are a number of ways to do this:
           https://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary

        Parameters
        ----------
        doc : str
            A text representation of a document to score.

        Returns
        -------
        str
            The most likely class as predicted by the model.
        """
        # >>> YOUR ANSWER HERE
        # For each class c, calculate the corresponding score of the doc
        scores = [(self.score(doc, c), c) for c in self.classes]
        # after the sort by score, return the most likely class
        scores.sort(key=lambda x: x[0])
        return scores[-1][1]
        # >>> END YOUR ANSWER

    def evaluate(self, test_dir='data/dev', target='real'):
        """Calculate a precision, recall, and F1 score for the model
        on a given test set.

        Not the 'target' parameter here, giving the name of the class
        to calculate relative to. So you can consider a True Positive
        to be an instance where the gold label for the document is the
        target and the model also predicts that label; a False Positive
        to be an instance where the gold label is *not* the target, but
        the model predicts that it is; and so on.

        Parameters
        ----------
        test_dir : str
            The path to a directory containing the test data.
        target : str
            The name of the class to calculate relative to.

        Returns
        -------
        (float, float, float)
            The model's precision, recall, and F1 score relative to the
            target class.
        """
        test_data = {c: os.path.join(test_dir, c) for c in self.classes}
        if not target in test_data:
            print('Error: target class does not exist in test data.')
            return
        outcomes = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
        # >>> YOUR ANSWER HERE
        data = []
        for c in test_data:
            docs = open(test_data[c]).readlines()
            for doc in docs:
                preprocess_doc = doc.strip()
                data.append((c, preprocess_doc))
        for item in data:
            predict_ans = self.predict(item[1])
            if item[0] == 'real':
                if predict_ans == 'real':
                    outcomes['TP'] += 1
                else:
                    outcomes['FN'] += 1
            else:
                if predict_ans == 'real':
                    outcomes['FP'] += 1
                else:
                    outcomes['TN'] += 1
        precision = outcomes['TP'] / (outcomes['TP'] + outcomes['FP'])  # replace with equation for precision
        recall = outcomes['TP'] / (outcomes['TP'] + outcomes['FN'])  # replace with equation for recall
        f1_score = 2 * ((precision * recall) / (precision + recall))  # replace with equation for f1
        # >>> END YOUR ANSWER
        return precision, recall, f1_score

    def print_top_features(self, k=10):
        results = {c: {} for c in self.classes}
        for w in self.vocabulary:
            for c in self.classes:
                ratio = math.exp(self.loglikelihood[w, c] - min(
                    self.loglikelihood[w, other_c] for other_c in self.classes if other_c != c))
                results[c][w] = ratio

        for c in self.classes:
            print(f'Top features for class <{c.upper()}>')
            for w, ratio in sorted(results[c].items(), key=lambda x: x[1], reverse=True)[0:k]:
                print(f'\t{w}\t{ratio}')
            print('')


if __name__ == '__main__':
    target = 'real'

    clf = NaiveBayesClassifier(train_dir='data/train')
    clf.train()
    print(f'Performance on class <{target.upper()}>, keeping stopwords')
    precision, recall, f1_score = clf.evaluate(test_dir='data/train', target=target)
    print(f'\tPrecision: {precision}\t Recall: {recall}\t F1: {f1_score}\n')

    clf = NaiveBayesClassifier(train_dir='data/train', REMOVE_STOPWORDS=True)
    clf.train()
    print(f'Performance on class <{target.upper()}>, removing stopwords')
    precision, recall, f1_score = clf.evaluate(test_dir='data/dev', target=target)
    print(f'\tPrecision: {precision}\t Recall: {recall}\t F1: {f1_score}\n')

    clf.print_top_features()
