# -*- coding: utf-8 -*-
# -*- created_time: 2019/4/17 -*-

import pandas as pd
import numpy as np
import re
import nltk
import time

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import GridSearchCV

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.neural_network import MLPClassifier


# 1. -*-  load fake news data (from Kaggle) -*-

def load_data(file="fake_news_data.csv"):
    data = pd.read_csv(file,encoding="utf-8")

    # omit all rows with null values
    data = data.dropna()

    # convert all non-string columns into string-type columns
    data = data.astype({"URLs":"str","Headline":"str","Body":"str"})

    # data_0 = data.loc[data["Label"]==0]
    # data_1 = data.loc[data["Label"]==1]
    # print("True news",data_0.shape)
    # print("Fake news",data_1.shape)
    return data


# 2. -*-  Output accuracy/classification report/confusion matrix. Output the results of cross-validation -*-

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
def evaluation_predictions(true_labels,predictions):
    print("\nAccuracy score:",accuracy_score(true_labels,predictions))
    print("Classification report:\n",classification_report(true_labels,predictions))
    print("Confusion matrix:\n",confusion_matrix(true_labels,predictions))

# output the results of cross-validation
def print_cv_scores_summary(cv_scores):
    def cv_score_summary(name,scores):
        print("{}: mean = {:.2f}%, sd = {:.2f}%, min = {:.2f}%, max = {:.2f}%".format(name, scores.mean() * 100,
                                                                                    scores.std() * 100,
                                                                                    scores.min() * 100,
                                                                                    scores.max() * 100))

    cv_score_summary("Accuracy", cv_scores['test_accuracy'])
    cv_score_summary("Precision", cv_scores['test_precision_weighted'])
    cv_score_summary("Recall", cv_scores['test_recall_weighted'])
    cv_score_summary("F1", cv_scores['test_f1_weighted'])



import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

#  A couple of methods for showing classifier results
def confusion_matrix_heatmap(true_labels,predictions,labels):
    confu_matrix = confusion_matrix(true_labels,predictions)
    cmdf = pd.DataFrame(confu_matrix, index = labels, columns=labels)
    dims = (10, 10)
    fig, ax = plt.subplots(figsize=dims)
    sns.heatmap(cmdf, annot=True, cmap="coolwarm", center=0)
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    plt.show()

def confusion_matrix_percent_heatmap(true_labels,predictions,labels):
    confu_matrix = confusion_matrix(true_labels,predictions)
    cmdf = pd.DataFrame(confu_matrix, index = labels, columns=labels)
    percents = cmdf.div(cmdf.sum(axis=1), axis=0)*100
    dims = (10, 10)
    fig, ax = plt.subplots(figsize=dims)
    sns.heatmap(percents, annot=True, cmap="coolwarm", center=0, vmin=0, vmax=100)
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([0, 25, 50, 75, 100])
    cbar.set_ticklabels(['0%', '25%', '50%', '75%', '100%'])


# 3. -*- Customising preprocessing -*-
import ftfy

def preprocess(text):
    # preprocess the multi-spaces, newlines and urls in strings

    multispace_re = re.compile(r"\s{2,}")
    newline_re = re.compile(r"\n+")
    url_re = re.compile(r"(?:https?://)?(?:[-\w]+\.)+[a-zA-Z]{2,9}[-\w/#~:;.?+=&%@~]*")
    # emoji_re = re.compile("([\U00010000-\U0010ffff]|[\uD800-\uDBFF][\uDC00-\uDFFF])")

    preprocessed_text = multispace_re.sub("",text)
    preprocessed_text = newline_re.sub("",preprocessed_text)
    preprocessed_text = url_re.sub("url", preprocessed_text)
    preprocessed_text = ftfy.fix_text(preprocessed_text)
    return preprocessed_text



# 4. -*- Customising word tokenization and word lemmatization -*-

def nltk_word_tokenize(text,advanced_tokenize=True):
    if advanced_tokenize:
        advanced_text_tokenize = nltk.tokenize.TweetTokenizer()
        return advanced_text_tokenize.tokenize(text)
    else:
        return nltk.word_tokenize(text)

# utilize WordNetLemmatizer class to creat the lemma n-gram
from nltk.stem import WordNetLemmatizer

# pos parameter could be "n", "v", "a", "r"
def word_lemmatizer(word, pos = "v"):
    lemmatizer = WordNetLemmatizer()
    return lemmatizer.lemmatize(word,pos=pos)


# 5. -*- Creating a document class to extract different features for each row of fake news data -*-
from collections import Counter
from nltk.util import ngrams,skipgrams
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

class Document():
    def __init__(self,meta = {"Web_source":""}):
        self.meta = meta

        # extract web source of news as a feature
        self.web_source_dict = Counter()
        self.web_source_dict.update(self.meta["Web_source"])

        # storing each token and corresponding pos-tag into lists
        self.tokens_list = []
        self.pos_tag_list = []
        # count the number of tokens and assign the text to the variable self.text
        self.num_tokens = 0
        self.text = ""

        # storing the frequencies of tokens into a statistical Counter class
        self.token_frequencies_dict = Counter()
        # storing the frequencies of pos-tags into a statistical Counter class
        self.pos_tag_frequencies_dict = Counter()

        # storing the frequencies of bi-gram into a statistical Counter class
        self.bi_gram_frequencies_dict  = Counter()
        # storing the frequencies of pos-tag bi-gram into a statistical Counter class
        self.postag_bi_gram_frequencies_dict = Counter()
        # storing the frequencies of lemma bi-gram into a statistical Counter class
        self.lemma_bi_gram_frequencies_dict = Counter()
        # storing the frequencies of bi-skipgram into a statistical Counter class
        self.bi_skipgram_frequencies_dict = Counter()

        # storing the frequencies of tri-gram into a statistical Counter class
        self.tri_gram_frequencies_dict = Counter()
        # storing the frequencies of pos-tag tri-gram into a statistical Counter class
        self.postag_tri_gram_frequencies_dict = Counter()
        # storing the frequencies of lemma tri-gram into a statistical Counter class
        self.lemma_tri_gram_frequencies_dict = Counter()
        # storing the frequencies of tri-skipgram into a statistical Counter class
        self.tri_skipgram_frequencies_dict = Counter()

    def extract_features_from_text(self,text):

        preprocessed_text = preprocess(text)
        self.text = preprocessed_text

        # tokenization
        tokens = nltk_word_tokenize(preprocessed_text)
        self.tokens_list.extend(tokens)
        self.num_tokens += len(tokens)
        self.token_frequencies_dict.update(tokens)

        # get the pos-tags of tokens
        tokens_postag = nltk.pos_tag(tokens)
        pos_tags = [pos_tag[1] for pos_tag in tokens_postag]
        self.pos_tag_list.extend(pos_tags)
        self.pos_tag_frequencies_dict.update(pos_tags)

        # get the verb lemmatization forms of tokens
        lemma_tokens = [word_lemmatizer(word,pos="v") for word in self.tokens_list]

        bi_gram_list = ["_".join(bigram) for bigram in ngrams(self.tokens_list,2)]
        self.bi_gram_frequencies_dict.update(bi_gram_list)

        postag_bi_gram_list = ["_".join(postag_bigram) for postag_bigram in ngrams(self.pos_tag_list,2)]
        self.postag_bi_gram_frequencies_dict.update(postag_bi_gram_list)

        lemma_bi_gram_list = ["_".join(lemma_bigram) for lemma_bigram in ngrams(lemma_tokens, 2)]
        self.lemma_bi_gram_frequencies_dict.update(lemma_bi_gram_list)

        # creat bi-skipgrams,and the distance of shipped tokens is 2 (containing bi-gram)
        bi_skipgram_list = ["_".join(bi_skipgram) for bi_skipgram in skipgrams(self.tokens_list,2,2)]
        self.bi_skipgram_frequencies_dict.update(bi_skipgram_list)


        tri_gram_list = ["_".join(trigram) for trigram in ngrams(self.tokens_list, 3)]
        self.tri_gram_frequencies_dict.update(tri_gram_list)

        postag_tri_gram_list = ["_".join(postag_trigram) for postag_trigram in ngrams(self.pos_tag_list, 3)]
        self.postag_tri_gram_frequencies_dict.update(postag_tri_gram_list)

        lemma_tri_gram_list = ["_".join(lemma_trigram) for lemma_trigram in ngrams(lemma_tokens, 3)]
        self.lemma_tri_gram_frequencies_dict.update(lemma_tri_gram_list)

        # creat tri-skipgrams,and the distance of shipped tokens is 2 (containing tri-gram)
        tri_skipgram_list = ["_".join(bi_skipgram) for bi_skipgram in skipgrams(self.tokens_list,3,2)]
        self.tri_skipgram_frequencies_dict.update(tri_skipgram_list)


    def extract_features_from_texts(self, texts):
        for text in texts:
            self.extract_features_from_text(text)

    def average_token_length(self):
        sum_character_length = 0
        for word,frequency in self.token_frequencies_dict.items():
            sum_character_length += len(word) * frequency
        return sum_character_length / self.num_tokens


# 6. -*- Creating training data and labels, and test data and labels -*-

def corpus_creation(data_df):
    for row in range(data_df.shape[0]):
        # use regular expression to extract web source
        web_source = re.search(r"((?<=(www.|api.))[a-zA-Z0-9]{1,}|[a-zA-Z0-9]{1,}(?=(.com|.tv|.net|.it)))",
                                            data_df.iloc[row, 0]).group()
        # store web source and label, of fake news, into the meta dictionary variable of Document class
        doc = Document(meta={"Web_source": web_source, "label": data_df.iloc[row, 3]})

        # combine news headline and news body
        text = data_df.iloc[row, 1] + " " + data_df.iloc[row, 2]
        doc.extract_features_from_text(text)
        yield doc



def train_text_data_creation(data_partition=True):
    # creat corpus and labels
    data_df = load_data()

    corpus = []
    corpus.extend(corpus_creation(data_df))
    Y = [doc.meta["label"] for doc in corpus]

    if data_partition:
        X_train, X_test, Y_train, Y_test = train_test_split(corpus, Y, test_size=0.3, random_state=0)
        return X_train, X_test, Y_train, Y_test
    else:
        return corpus,Y


# 7. -*-
# Now, the train/test instances are created, a custom Transformer is needed to be constructed, which takes in one dataset
# and returns a new dataset. Here we need to take in a list of Document objects and transform it into a set of features.
# We build a simple class for this, which overrides the transform method. The intention is for a list of Document objects
# to be passed into the transformer, and parameter-defined (callable) method is used to extract featuress. -*-
from sklearn.base import BaseEstimator, TransformerMixin

class DocumentProcessor(BaseEstimator,TransformerMixin):
    def __init__(self,process_method):
        self.process_method = process_method

    # no fitting necessary, although could use this to build a vocabulary for all documents, and then limit to set (e.g. top 1000).
    def fit(self,X,y=None):
        return self

    def transform(self,documents):
        for document in documents:
            yield self.process_method(document)

# Below are some process methods for returning the extracted features from the Document. These can be edited and added to as needed.
def get_token_frequencies_dict(document):
    return document.token_frequencies_dict

def get_pos_tag_frequencies_dict(document):
    return document.pos_tag_frequencies_dict

def get_bi_gram_frequencies_dict(document):
    return document.bi_gram_frequencies_dict

def get_postag_bi_gram_frequencies_dict(document):
    return document.postag_bi_gram_frequencies_dict

def get_lemma_bi_gram_frequencies_dict(document):
    return document.lemma_bi_gram_frequencies_dict

def get_bi_skipgram_frequencies_dict(document):
    return document.bi_skipgram_frequencies_dict

def get_tri_gram_frequencies_dict(document):
    return document.tri_gram_frequencies_dict

def get_postag_tri_gram_frequencies_dict(document):
    return document.postag_tri_gram_frequencies_dict

def get_lemma_tri_gram_frequencies_dict(document):
    return document.lemma_tri_gram_frequencies_dict

def get_tri_skipgram_frequencies_dict(document):
    return document.tri_skipgram_frequencies_dict


# 8. -*- Construct different pipeline models based on combination of different ngram features -*-

# pipeline model 1 is based on tokens features (bag of words)
model_1 = Pipeline([
    ("union",FeatureUnion(transformer_list=[
        ("tokens_features",Pipeline([
            ("tokens_processor",DocumentProcessor(process_method=get_token_frequencies_dict)),
            ("tokens_vectorizer",DictVectorizer())
        ])),
    ])),
    ("clf",LinearSVC(random_state=0))
])

# pipeline model 2 is based on pos-tag features (bag of pos-tags)
model_2 = Pipeline([
    ("union",FeatureUnion(transformer_list=[
        ("pos_tag_features",Pipeline([
            ("pos_tag_processor",DocumentProcessor(process_method=get_pos_tag_frequencies_dict)),
            ("pos_tag_vectorizer",DictVectorizer())
        ])),
    ])),
    ("clf",LinearSVC(random_state=0))
])

# pipeline model 3 is based on bi-gram and/or tri-gram
model_3 = Pipeline([
    ("union",FeatureUnion(transformer_list=[
        ("bi_gram_features",Pipeline([
            ("bi_gram_processor",DocumentProcessor(process_method=get_bi_gram_frequencies_dict)),
            ("bi_gram_vectorizer",DictVectorizer())
        ])),
        # ("tri_gram_features",Pipeline([
        #     ("tri_gram_processor",DocumentProcessor(process_method=get_tri_gram_frequencies_dict)),
        #     ("tri_gram_vectorizer",DictVectorizer())
        # ])),
    ])),
    ("clf",LinearSVC(random_state=0))
])

# pipeline model 4 is based on pos-tag bi-gram and/or pos-tag tri-gram
model_4 = Pipeline([
    ("union",FeatureUnion(transformer_list=[
        ("postag_bi_gram_features",Pipeline([
            ("postag_bi_gram_processor",DocumentProcessor(process_method=get_postag_bi_gram_frequencies_dict)),
            ("postag_bi_gram_vectorizer",DictVectorizer())
        ])),
        # ("postag_tri_gram_features",Pipeline([
        #     ("postag_tri_gram_processor",DocumentProcessor(process_method=get_postag_tri_gram_frequencies_dict)),
        #     ("postag_tri_gram_vectorizer",DictVectorizer())
        # ])),
    ])),
    ("clf",LinearSVC(random_state=0))
])

# pipeline model 5 is based on lemma bi-gram and/or lemma tri-gram
model_5 = Pipeline([
    ("union",FeatureUnion(transformer_list=[
        ("lemma_bi_gram_features",Pipeline([
            ("lemma_bi_gram_processor",DocumentProcessor(process_method=get_lemma_bi_gram_frequencies_dict)),
            ("lemma_bi_gram_vectorizer",DictVectorizer())
        ])),
        # ("lemma_tri_gram_features",Pipeline([
        #     ("lemma_tri_gram_processor",DocumentProcessor(process_method=get_lemma_tri_gram_frequencies_dict)),
        #     ("lemma_tri_gram_vectorizer",DictVectorizer())
        # ])),
    ])),
    ("clf",LogisticRegression(random_state=0,solver="lbfgs"))
])

# pipeline model 6_1 is based on bi-skipgram (containing bi-gram )
model_6_1 = Pipeline([
    ("union",FeatureUnion(transformer_list=[
        ("bi_skipgram_features",Pipeline([
            ("bi_skipgram_processor",DocumentProcessor(process_method=get_bi_skipgram_frequencies_dict)),
            ("bi_skipgram_vectorizer",DictVectorizer())
        ]))
    ])),
    ("clf",LinearSVC(random_state=0))
])

# pipeline model 6_2 is based on tri-skipgram (containing tri-gram)
model_6_2 = Pipeline([
    ("union",FeatureUnion(transformer_list=[
        ("tri_skipgram_features",Pipeline([
            ("tri_skipgram_processor",DocumentProcessor(process_method=get_tri_skipgram_frequencies_dict)),
            ("tri_skipgram_vectorizer",DictVectorizer())
        ]))
    ])),
    ("clf",LinearSVC(random_state=0))
])

# pipeline model 7_1 is based on postag bi-gram and bi-gram
model_7_1 = Pipeline([
    ("union",FeatureUnion(transformer_list=[
        ("postag_bi_gram_features", Pipeline([
            ("postag_bi_gram_processor", DocumentProcessor(process_method=get_postag_bi_gram_frequencies_dict)),
            ("postag_bi_gram_vectorizer", DictVectorizer())
        ])),
        ("bi_gram_features", Pipeline([
            ("bi_gram_processor", DocumentProcessor(process_method=get_bi_gram_frequencies_dict)),
            ("bi_gram_vectorizer", DictVectorizer())
        ])),
    ])),
    ("clf",LinearSVC(random_state=0))
])

# pipeline model 7_2 is based on postag tri-gram and tri-gram
model_7_2 = Pipeline([
    ("union",FeatureUnion(transformer_list=[
        ("postag_tri_gram_features", Pipeline([
            ("postag_tri_gram_processor", DocumentProcessor(process_method=get_postag_tri_gram_frequencies_dict)),
            ("postag_tri_gram_vectorizer", DictVectorizer())
        ])),
        ("tri_gram_features", Pipeline([
            ("tri_gram_processor", DocumentProcessor(process_method=get_tri_gram_frequencies_dict)),
            ("tri_gram_vectorizer", DictVectorizer())
        ])),
    ])),
    ("clf",LinearSVC(random_state=0))
])

# pipeline model 8_1 is based on lemma bi-gram and bi-gram
model_8_1 = Pipeline([
    ("union",FeatureUnion(transformer_list=[
        ("lemma_bi_gram_features",Pipeline([
            ("lemma_bi_gram_processor",DocumentProcessor(process_method=get_lemma_bi_gram_frequencies_dict)),
            ("lemma_bi_gram_vectorizer",DictVectorizer())
        ])),
        ("bi_gram_features", Pipeline([
            ("bi_gram_processor", DocumentProcessor(process_method=get_bi_gram_frequencies_dict)),
            ("bi_gram_vectorizer", DictVectorizer())
        ])),
    ])),
    ("clf",LinearSVC(random_state=0))
])

# pipeline model 8_2 is based on lemma tri-gram and tri-gram
model_8_2 = Pipeline([
    ("union",FeatureUnion(transformer_list=[
        ("lemma_tri_gram_features",Pipeline([
            ("lemma_tri_gram_processor",DocumentProcessor(process_method=get_lemma_tri_gram_frequencies_dict)),
            ("lemma_tri_gram_vectorizer",DictVectorizer())
        ])),
        ("tri_gram_features", Pipeline([
            ("tri_gram_processor", DocumentProcessor(process_method=get_tri_gram_frequencies_dict)),
            ("tri_gram_vectorizer", DictVectorizer())
        ])),
    ])),
    ("clf",LinearSVC(random_state=0))
])

# # Give the previous model name and run the function train_model_by_cv(), the results of four metrics would be outputed.
def train_model_by_cv(model_name):
    time_start = time.time()

    X,Y = train_text_data_creation(data_partition=False)

    # just return different cv scores based on cross-validation test sets
    cv_scores_model = cross_validate(model_name, X, Y, cv=StratifiedKFold(n_splits=10,random_state=0), return_train_score=False,
                                   scoring=["accuracy","precision_weighted","recall_weighted","f1_weighted"])
    time_end = time.time()
    print("Time cost: ",time_end - time_start,"seconds")
    print_cv_scores_summary(cv_scores_model)





# 9. -*- Based on Word2Vec model (CBOW) in gensim package, convert all fake news sentences into vector representations to train models-*-
from gensim.models import word2vec

# After running the function fake_news_to_vectors(), the trained word2vec CBOW model would be saved.
# Just load the saved CBOW model in function vectorized_data_creation(), and use it directly.

def fake_news_to_vectors():
    # not to part the corpus X and target Y
    corpus,Y = train_text_data_creation(data_partition=False)
    # get all tokens in each fake news sentences
    fake_news_sen_tokens = [doc.tokens_list for doc in corpus]

    # train a word2vec CBOW model to convert all tokens in every fake news sentences, into vector representations
    # word dimensionality parameter size is set to 300
    # Here, in output layer of CBOW neural netword, it would use negatve sampling method to do the gradient ascent (sg=0 and hs=0),
    # and the amount of noise word is set to 10.
    word2vec_model = word2vec.Word2Vec(sentences=fake_news_sen_tokens, size=300, window=10, min_count=1, sg=0, hs=0,
                                       negative=10,seed=0,workers=4)

    # The model would not be trained any further. Hence here, use init_sims() function to make model more read-effective and memory-effective
    word2vec_model.init_sims(replace=True)
    # Svae the trained word2vec CBOW model. Later, for convenience, it can load saved model and get word vectors directly.
    word2vec_model.save("fake_news_word2vec_model.model")
    print("fake_news_word2vec_model is saved successfully")


def vectorized_data_creation(mean_vector = True):
    # load saved word2vec CBOW model and get word vectors directly.
    word2vec_model = word2vec.Word2Vec.load("fake_news_word2vec_model.model")

    # not to part the corpus X and target Y
    corpus, Y = train_text_data_creation(data_partition=False)
    # get all tokens in each fake news sentences
    fake_news_sen_tokens = [doc.tokens_list for doc in corpus]

    # initialize a 300-dimensionality numpy array to store all vectors of fake news sentences (each row represents the vector
    # of one fake news sentence )
    X_vectors = np.zeros((len(fake_news_sen_tokens),300))


    for num in range(len(fake_news_sen_tokens)):
        each_news_tokens = fake_news_sen_tokens[num]
        each_news_vectors = None

        if mean_vector:
            # # initialize a 300-dimensionality numpy array to store a vector of one fake news sentence
            # each_news_vectors = np.zeros((len(each_news_tokens), 300))
            #
            # for i in range(len(each_news_tokens)):
            #     # one raw in zero array each_news_vectors, would be replaced by a 300-dims vector array of one token
            #     each_news_vectors[i] = word2vec_model.wv[each_news_tokens[i]]
            each_news_vectors = np.vstack((word2vec_model.wv[token] for token in each_news_tokens))

            # Now, the numpy array contains massive rows, and rach row represents a 300-dimensionality vector of a token in each fake news sentence
            # However here, it is needed to calculate the mean vector of all tokens' vectors in one fake news sentence, and use this mean vector to
            # represent the vector of one fake news sentence (sentence vector)
            each_news_vectors = np.mean(each_news_vectors, axis=0)

        else:
            each_news_vectors = np.vstack((word2vec_model.wv[token] for token in each_news_tokens))
            each_news_vectors = np.sum(each_news_vectors, axis=0)


        # Then, store the 300-dimensionality vector of each fake news sentences into each row of numpy array X_vectors
        X_vectors[num] = each_news_vectors

    return X_vectors,Y

# Based on all vectors of fake news sentences, train a model
def train_model_by_vectors():
    time_start = time.time()

    X_vectors, Y = vectorized_data_creation(mean_vector=False)
    linear_svc = LinearSVC(random_state=0)

    cv_scores_model = cross_validate(linear_svc, X_vectors, Y, cv=StratifiedKFold(n_splits=10,random_state=0),return_train_score=False,
                                   scoring=["accuracy","precision_weighted","recall_weighted","f1_weighted"])

    time_end = time.time()
    print("Time cost: ", time_end - time_start, "seconds")
    print_cv_scores_summary(cv_scores_model)


if __name__ == "__main__":
    # the names of models could be model_3, model_4, model_5, model_6_1, model_6_2, model_7_1, model_7_2, model_8_1, model_8_2
    # train_model_by_cv(model_7_2)

    # After running the function fake_news_to_vectors(), the trained word2vec CBOW model would be saved.
    fake_news_to_vectors()
    # Perform the fake news classification by word embedding
    train_model_by_vectors()


