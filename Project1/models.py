# models.py

from sentiment_data import Counter, List
import torch
import torch.nn as nn
from torch import optim  
import numpy as np
import random
from typing import List
from sentiment_data import *
from utils import *
from collections import Counter
import tqdm
from nltk.corpus import stopwords


stop_words = set(stopwords.words('english'))


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str]) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]]) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :return:
        """
        return [self.predict(ex_words) for ex_words in all_ex_words]


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str]) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1


class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """

    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param sentence: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return: A feature vector. We suggest using a Counter[int], which can encode a sparse feature vector (only
        a few indices have nonzero value) in essentially the same way as a map. However, you can use whatever data
        structure you prefer, since this does not interact with the framework code.
        """
        raise Exception("Don't call me, call my subclasses")


class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.

    :param indexer: contains the vocabulary of the FeatureExtractor. It can be grown or fixed depending on usage. 
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer
        self.stop_words = stop_words

    def get_indexer(self):
        return self.indexer
    
    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        counter = Counter()
        for word in sentence:
            if self.process_unigram(word) is not None:
                idx = self.indexer.add_and_get_index(word, add_to_indexer)
                counter[idx] += 1
        return counter

    """
    Returns preprocessed unigram token and returns None if token is not a valid feature.
    Only stop words are considered invalid at this time. 
    """
    def process_unigram(self, word: str) -> str:
        result = word.lower()
        return result if result not in self.stop_words else None


class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """

    def __init__(self, indexer: Indexer):
        self.indexer = indexer
        self.stop_words = set(stopwords.words('english'))

    def get_indexer(self):
        return self.indexer
    
    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        counter = Counter()
        sentence = list(filter(lambda token: token != None, map(
                lambda token: self.process_unigram(token), 
                sentence
        )))
        for i in range(len(sentence) - 1):
            idx = self.indexer.add_and_get_index(frozenset((sentence[i], sentence[i + 1])), add_to_indexer)
            counter[idx] += 1
        return counter
    
    def process_unigram(self, word: str) -> str:
        result = word.lower()
        return result if result not in self.stop_words else None


class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    """

    def __init__(self, indexer: Indexer):
        raise Exception("Must be implemented")


class LogisticRegressionClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, weights: np.ndarray, feat_extractor: FeatureExtractor):
        self.weights = weights
        self.feat_extractor = feat_extractor
    
    def predict(self, ex_words: List[str]) -> int:
        feature_vector = self.feat_extractor.extract_features(ex_words)
        dot = sum([self.weights[i]*feature_vector[i] for i in feature_vector])
        exp = np.exp(dot)
        probability = exp/(1 + exp)
        return 1 if probability >= 0.5 else 0
    
    def predict_all(self, all_ex_words: List[List[str]]) -> List[int]:
        return super().predict_all(all_ex_words)
        


def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """
    # random.seed(1)
    num_epochs = 10
    num_samples = len(train_exs)
    step_size = 0.01
    tau = 1
    schedule_rate = 2
    feature_vectors: List[Counter] = [feat_extractor.extract_features(sample.words, True) for sample in train_exs]
    indexer: Indexer = feat_extractor.get_indexer()
    vocab_size = len(indexer)

    #zero weight initialization
    weights = np.zeros(vocab_size)

    for epoch in tqdm.tqdm(range(num_epochs)):
        sample_order = list(range(num_samples))
        random.shuffle(sample_order)
        cost = 0
        for idx in sample_order:
            dot = sum([weights[i]*feature_vectors[idx][i] for i in feature_vectors[idx]])
            exp = np.exp(dot)
            probability = exp/(1 + exp)
            probability = train_exs[idx].label - probability
            for i in feature_vectors[idx]:
                weights[i] += probability*feature_vectors[idx][i]*step_size
        

        if schedule_rate != 0 and epoch % schedule_rate == 0:
            step_size /= tau


    
    model = LogisticRegressionClassifier(weights, feat_extractor)
    return model




def train_linear_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your linear model. You may modify this, but do not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Add additional preprocessing code here
        feat_extractor = UnigramFeatureExtractor(Indexer())
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        feat_extractor = BetterFeatureExtractor(Indexer())
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    # Train the model
    model = train_logistic_regression(train_exs, feat_extractor)
    return model


class SentimentClassifierNN(nn.Module):
    def __init__(self, embedding_layer, inp, hid, out):
        super(SentimentClassifierNN, self).__init__()
        self.embedding_layer = embedding_layer
        linear_1 = nn.Linear(inp, hid)
        linear_out = nn.Linear(hid, out)
        nn.init.xavier_uniform_(linear_1.weight)
        nn.init.xavier_uniform_(linear_out.weight)


        self.linear = nn.Sequential(
            linear_1, 
            nn.ReLU(),
            linear_out, 
            nn.LogSoftmax(dim=0)
        )
    
    #x: list of indices that can be converted into an embedding
    def forward(self, x_indices):
        x = torch.sum(self.embedding_layer(x_indices), 0)
        x /= len(x_indices)
        return self.linear(x)
    

def process_token(word: str) -> str:
    result = word.lower()
    return result if result not in stop_words else None

class NeuralSentimentClassifier(SentimentClassifier):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.)
    """
    def __init__(self, network, word_embeddings):

        self.model = network
        self.word_embeddings = word_embeddings
        self.indexer = self.word_embeddings.word_indexer
    def predict(self, ex_words: List[str]) -> int: 
        x = torch.zeros(1, self.word_embeddings.get_embedding_length()).float()
        tokens = list(filter(lambda token: token != None, map(
                lambda token: process_token(token), 
                ex_words
            )))
        token_indices = set()
        for token in tokens:
            if self.indexer.contains(token):
                token_indices.add(self.indexer.index_of(token))
        
        x = torch.tensor(list(token_indices), dtype=torch.long)
        y_hat = self.model.forward(x)
        return torch.argmax(y_hat)
    
    def predict_all(self, all_ex_words: List[List[str]]) -> List[int]:
        return super().predict_all(all_ex_words)
    


def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample], word_embeddings: WordEmbeddings) -> NeuralSentimentClassifier:
    """
    Main entry point for your deep averaging network model.
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :return: A trained NeuralSentimentClassifier model
    """
    
    num_samples = len(train_exs)
    embeddings: nn.Embedding = word_embeddings.get_initialized_embedding_layer(frozen=True)
    indexer = word_embeddings.word_indexer
    print("Num Samples: ", num_samples)
    num_epochs = 6
    num_classes = 2
    hid = 60
    input_dim = word_embeddings.get_embedding_length()
    #Create input vectors
    loss_function = torch.nn.CrossEntropyLoss()
    x = []
    labels = torch.zeros(num_samples)
    for idx in range(num_samples):
        sample = train_exs[idx]
        tokens = list(filter(lambda token: token != None, map(
                lambda token: process_token(token), 
                sample.words
            )))        
        token_indices = set() 
        for token in tokens:
            if indexer.contains(token):
                token_indices.add(indexer.index_of(token))
        x.append(torch.tensor(list(token_indices)))
        labels[idx] = sample.label


    model = SentimentClassifierNN(embeddings, input_dim, hid, num_classes)
    initial_learning_rate = 0.0005
    optimizer = optim.Adam(model.parameters(), lr=initial_learning_rate)

    for epoch in range(num_epochs):
        sample_order = list(range(num_samples))
        random.shuffle(sample_order)
        total_loss = 0.0

        for idx in sample_order:
            y = labels[idx]
            y_onehot = torch.zeros(num_classes)
            y_onehot.scatter_(0, torch.from_numpy(np.asarray(y, dtype=np.int64)), 1)

            model.zero_grad()
            y_hat = model.forward(x[idx])

            loss = loss_function(y_hat, y_onehot)
            total_loss += loss

            loss.backward()
            optimizer.step()
        if not epoch % 1:
            print("Average Loss on epoch %i: %f" % (epoch, total_loss/num_samples))

    classifier = NeuralSentimentClassifier(model, word_embeddings)
    return classifier
