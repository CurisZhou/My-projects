# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
from scipy import sparse
import re
import nltk
import time

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_selection import SelectKBest, chi2

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score,classification_report,recall_score, precision_score,confusion_matrix



# 1. -*-  load data  -*-

def load_data(data_file="review_data.csv"):
    data = pd.read_csv(data_file,encoding="utf-8",lineterminator='\n')

    # omit all rows with null values
    data =data.dropna()


    # Encode the label "Positive" as 1, and encode the label "Negative" as 0
    data["label"] = data["label"].replace(["Positive","Negative"],[1,0])

    # data_0 = data.loc[data["label"]==1]
    # data_1 = data.loc[data["label"]==0]
    # print("Positive",data_0.shape)
    # print("Negative",data_1.shape)
    # print("All data: ",data.shape[0],"instances")

    # return data
    return data
    # print(train_data.head(300))


# 2. -*-  Output accuracy/classification report/confusion matrix. Output the results of cross-validation -*-


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



# 3. -*- Customising preprocessing -*-
# import ftfy

def preprocess(text):
    # preprocess the multi-spaces, newlines and urls in strings

    multispace_re = re.compile(r"\s{2,}")
    hashtag_re = re.compile(r"#\w+")
    mention_re = re.compile(r"@\w+")
    url_re = re.compile(r"(?:https?://)?(?:[-\w]+\.)+[a-zA-Z]{2,9}[-\w/#~:;.?+=&%@~]*")
    # emoji_re = re.compile("([\U00010000-\U0010ffff]|[\uD800-\uDBFF][\uDC00-\uDFFF])")

    preprocessed_text = multispace_re.sub("",text)
    preprocessed_text = hashtag_re.sub("",preprocessed_text)
    preprocessed_text = mention_re.sub("",preprocessed_text)
    preprocessed_text = url_re.sub("url", preprocessed_text)
    # preprocessed_text = ftfy.fix_text(preprocessed_text)
    return preprocessed_text



# 4. -*- Customising word tokenization  -*-

def nltk_word_tokenize(text,advanced_tokenize=True):
    if advanced_tokenize:
        advanced_text_tokenize = nltk.tokenize.TweetTokenizer()
        return advanced_text_tokenize.tokenize(text)
    else:
        return nltk.word_tokenize(text)


# 5. -*- Creating a document class to extract different features for each row of review data -*-
from collections import Counter
from nltk.util import ngrams,skipgrams
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

class Document():
    def __init__(self,meta = {}):
        self.meta = meta

        # storing each token and corresponding pos-tag into lists
        self.tokens_list = []
        # self.pos_tag_list = []
        # count the number of tokens and assign the text to the variable self.text
        self.num_tokens = 0
        self.text = ""

        # storing the frequencies of tokens into a statistical Counter class
        self.token_frequencies_dict = Counter()

        # # storing the frequencies of pos-tags into a statistical Counter class
        # self.pos_tag_frequencies_dict = Counter()
        # storing the frequencies of bi-gram into a statistical Counter class
        # self.bi_gram_frequencies_dict  = Counter()

        # storing the frequencies of emojis into a statistical Counter class
        self.emoji_frequencies_dict = Counter()


    def extract_features_from_text(self,text):

        preprocessed_text = preprocess(text)
        self.text = preprocessed_text

        # tokenization
        tokens = nltk_word_tokenize(preprocessed_text)
        self.tokens_list.extend(tokens)
        self.num_tokens += len(tokens)
        self.token_frequencies_dict.update(tokens)

        # # get the pos-tags of tokens
        # tokens_postag = nltk.pos_tag(tokens)
        # pos_tags = [pos_tag[1] for pos_tag in tokens_postag]
        # self.pos_tag_list.extend(pos_tags)
        # self.pos_tag_frequencies_dict.update(pos_tags)

        # bi_gram_list = ["_".join(bigram) for bigram in ngrams(self.tokens_list,9)]
        # self.bi_gram_frequencies_dict.update(bi_gram_list)

        # It would extract all emojis from text and regard them as new features
        emojis = re.findall("([\U00010000-\U0010ffff]|[\uD800-\uDBFF][\uDC00-\uDFFF])", preprocessed_text)
        self.emoji_frequencies_dict.update(emojis)


    def average_token_length(self):
        sum_character_length = 0
        for word,frequency in self.token_frequencies_dict.items():
            sum_character_length += len(word) * frequency
        return sum_character_length / self.num_tokens



# 6. -*- Create data and labels. Extract different linguistic features from each row of text data and store them into Document classes -*-

def corpus_creation(data_df):

    for row in range(data_df.shape[0]):
        # store sentiment label of review, into the meta dictionary variable of Document class
        doc = Document(meta={"label": data_df.iloc[row, 2]})

        # get review body
        text = data_df.iloc[row, 1]
        doc.extract_features_from_text(text)
        yield doc



# Below are some process methods for returning the extracted features from the Document. These can be edited and added to as needed.
def get_token_frequencies_dict(document):
    return document.token_frequencies_dict

# def get_pos_tag_frequencies_dict(document):
#     return document.pos_tag_frequencies_dict

# def get_bi_gram_frequencies_dict(document):
#     return document.bi_gram_frequencies_dict

def get_emoji_frequencies_dict(document):
    return document.emoji_frequencies_dict



def train_text_data_creation():
    # return data
    data = load_data()

    # Extract different linguistic features from each row of text data and store these features into Document classes.
    # Store all Document classes into corpus class
    corpus = []
    corpus.extend(corpus_creation(data))
    # Store all labels of data into list Y
    Y = np.array([doc.meta["label"] for doc in corpus])


    # Creat tf-idf linguistic features based on bag of words in corpus
    corpus_texts = [doc.text for doc in corpus]
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_features = tfidf_vectorizer.fit_transform(corpus_texts)

    # Select the most important top k extracted features, based on the function SelectKBest(chi2, k=number)
    feature_selector = SelectKBest(chi2, k=10000)
    tfidf_features = feature_selector.fit_transform(tfidf_features, Y)
    print("The selection of tf-idf features, is finished")


    # Creat emoji features based on corpus
    emoji_frequencies_dict = [get_emoji_frequencies_dict(doc) for doc in corpus]
    emoji_vectorizer = DictVectorizer(sparse=False)
    emoji_frequencies_vectorized = emoji_vectorizer.fit_transform(emoji_frequencies_dict)


    # # Creat bi-grams linguistic features based on bag of words in corpus
    # bigram_frequencies_dict = [get_bi_gram_frequencies_dict(doc) for doc in corpus]
    # bigram_vectorizer = DictVectorizer(sparse=False)
    # bigram_frequencies_vectorized = bigram_vectorizer.fit_transform(train_bigram_frequencies_dict)


    # An exception occurs when numpy np.hstack() function is used to merge multiple sparse matrices (containing a large number of zeros)
    # Now, it should use the sparse.hstack().toarray() function in the scipy package.
    # The sparse.hstack().toarray() function can horizontally combine a variety of different linguistic features extracted
    # from the text corpus such as word frequency, tf-idf, emojis.
    X = sparse.hstack((tfidf_features,emoji_frequencies_vectorized)).toarray()

    return X, Y


# 7. -*- Construct a stacked generalization model to do the sentiment analysis  -*-
class stacked_generalization():
    def __init__(self,data,target):
        self.data = data
        if len(target.shape) == 2:
            # Convert 2-dim target array into 1-dim target array
            self.target = target.reshape(target.shape[0])
        else:
            self.target = target

        self.training_data = None
        self.training_target = None
        self.test_data = None
        self.test_target = None

        # Construct 3 Tier-1 (base) classifiers
        self.Tier1_classifier1 = LogisticRegression(solver="lbfgs")
        self.Tier1_classifier2 = MultinomialNB()
        self.Tier1_classifier3 = LinearSVC(penalty="l2")
        self.Tier1_classifier4 = ExtraTreeClassifier()
        # self.Tier1_classifier5 = SGDClassifier(max_iter=1000, tol=1e-3)

        # Construct Tier-2 (meta) classifier
        # self.meta_classifier = LogisticRegression(solver="lbfgs")
        # self.meta_classifier = MultinomialNB()
        # self.meta_classifier = LinearSVC(penalty = "l2")
        self.meta_classifier = ExtraTreeClassifier()
        # self.meta_classifier = XGBClassifier()
        # self.meta_classifier = RandomForestClassifier(n_estimators=100)


    # Divide training data into different n_split training blocks and evaluation blocks
    # Create T Tier-1 classifiers, C1,..,CT, based on a cross-validation partition of the training data. To do so,
    # the entire training dataset is divided into B blocks, and each Tier-1 classifier is first trained on (a different set of)
    # B-1 blocks of the training data. Each classifier is then evaluated on the Bth (pseudo-test) block
    def TrainingData_Stratified_KFold_split(self,n_split = 5, shuffle = False):
        # Blocks of training data Partition. n_splits cannot be greater than the number of members in each class
        skf_blocks = StratifiedKFold(n_splits=n_split,shuffle=shuffle)

        # Creat the indexes of blocks of training data. The number of blocks is n_split
        training_blocks_index =[]
        evaluation_blocks_index = []

        for trainingBlock_index, evaluationBlock_index in skf_blocks.split(self.training_data, self.training_target):
            training_blocks_index.append(trainingBlock_index)
            evaluation_blocks_index.append(evaluationBlock_index)

        training_blocks_data = [self.training_data[index,:] for index in training_blocks_index]
        training_blocks_target = [self.training_target[index] for index in training_blocks_index]

        evaluation_blocks_data = [self.training_data[index,:] for index in evaluation_blocks_index]
        evaluation_blocks_target = [self.training_target[index] for index in evaluation_blocks_index]

        return training_blocks_data, training_blocks_target, evaluation_blocks_data, evaluation_blocks_target


    def train_meta_classifier(self):
        training_blocks_data, training_blocks_target, evaluation_blocks_data, evaluation_blocks_target = self.TrainingData_Stratified_KFold_split()

        # The classification outputs of all Tier-1 classifiers on each training data block (5 blocls now) are saved in list Tier1_outputs
        Tier1_outputs = []

        for block in range(len(training_blocks_data)):
            # all Tier-1 base classifiers fit n-1 training data blocks (n blocks totally)
            self.Tier1_classifier1.fit(training_blocks_data[block],training_blocks_target[block])
            self.Tier1_classifier2.fit(training_blocks_data[block],training_blocks_target[block])
            self.Tier1_classifier3.fit(training_blocks_data[block],training_blocks_target[block])
            self.Tier1_classifier4.fit(training_blocks_data[block],training_blocks_target[block])
            # self.Tier1_classifier5.fit(training_blocks_data[block],training_blocks_target[block])

            # All Tier-1 base classifiers fit nth training data blocks (n blocks totally).The outputs of all Tier-1 base
            # classifiers on each training data block (5 blocls now) are saved in list Tier1_outputs
            output_C1 = self.Tier1_classifier1.predict(evaluation_blocks_data[block])
            output_C1 = output_C1.reshape(output_C1.shape[0],1)

            output_C2 = self.Tier1_classifier2.predict(evaluation_blocks_data[block])
            output_C2 = output_C2.reshape(output_C2.shape[0],1)

            output_C3 = self.Tier1_classifier3.predict(evaluation_blocks_data[block])
            output_C3 = output_C3.reshape(output_C3.shape[0],1)

            output_C4 = self.Tier1_classifier4.predict(evaluation_blocks_data[block])
            output_C4 = output_C4.reshape(output_C4.shape[0],1)

            # output_C5 = self.Tier1_classifier5.predict(evaluation_blocks_data[block])
            # output_C5 = output_C5.reshape(output_C5.shape[0],1)

            # The classification outputs of all Tier-1 classifiers on each training data block (5 blocls now) are saved in list Tier1_outputs
            block_outputs = np.hstack((output_C1,output_C2,output_C3,output_C4))  # horizontally combined
            Tier1_outputs.append(block_outputs)

        # Vertically combine all training data blocks' classification outputs of all Tier-1 classifiers.
        # The function np.vstack() can be given a list
        Tier1_outputs = np.vstack(Tier1_outputs)
        # Combine all training data blocks' real labels
        evaluation_blocks_target = np.concatenate([eva_block_target for eva_block_target in evaluation_blocks_target])

        # Using all training data blocks' classification outputs of all Tier-1 classifiers and all training data blocks'
        # real labels to train the meta classifier
        self.meta_classifier.fit(Tier1_outputs,evaluation_blocks_target)

        print("The training of meta classifier is finished")
        # return accuracy, recall and precision of test data


    # Train stacked generalization by cross-validation partition.
    def train_stacked_generalization_CV(self,n_split = 5, shuffle = False):
        # Cross-validation Partition.  n_splits cannot be greater than the number of members in each class
        skf_cv = StratifiedKFold(n_splits=n_split, shuffle=shuffle)

        # Creat the indexes of training data and test data
        training_sets_index = []
        test_sets_index = []

        for training_index, test_index in skf_cv.split(self.data, self.target):
            training_sets_index.append(training_index)
            test_sets_index.append(test_index)

        training_sets_data = [self.data[index,:] for index in training_sets_index]
        training_sets_target = [self.target[index] for index in training_sets_index]

        test_sets_data = [self.data[index,:] for index in test_sets_index]
        test_sets_target = [self.target[index] for index in test_sets_index]

        # Store all metrics of cross-validation in different lists
        test_cv_accuracy = []
        test_cv_recall = []
        test_cv_precision = []

        time_start = time.time() # start time

        for cv_time in range(n_split):
            self.training_data = training_sets_data[cv_time]
            self.training_target = training_sets_target[cv_time]
            self.test_data = test_sets_data[cv_time]
            self.test_target = test_sets_target[cv_time]

            # train the meta classifier
            self.train_meta_classifier()

            # Using all training data to retrain the all Tier-1 base classifiers
            self.Tier1_classifier1.fit(self.training_data,self.training_target)
            self.Tier1_classifier2.fit(self.training_data,self.training_target)
            self.Tier1_classifier3.fit(self.training_data,self.training_target)
            self.Tier1_classifier4.fit(self.training_data,self.training_target)
            # self.Tier1_classifier5.fit(self.training_data,self.training_target)

            # All retrained Tier-1 base classifiers are utilized to predict the test data
            testset_output_C1 = self.Tier1_classifier1.predict(self.test_data)
            testset_output_C1 = testset_output_C1.reshape(testset_output_C1.shape[0],1)

            testset_output_C2 = self.Tier1_classifier2.predict(self.test_data)
            testset_output_C2 = testset_output_C2.reshape(testset_output_C2.shape[0],1)

            testset_output_C3 = self.Tier1_classifier3.predict(self.test_data)
            testset_output_C3 = testset_output_C3.reshape(testset_output_C3.shape[0],1)

            testset_output_C4 = self.Tier1_classifier4.predict(self.test_data)
            testset_output_C4 = testset_output_C4.reshape(testset_output_C4.shape[0],1)

            # testset_output_C5 = self.Tier1_classifier5.predict(self.test_data)
            # testset_output_C5 = testset_output_C5.reshape(testset_output_C5.shape[0],1)

            # Horizontally combine all Tier-1 base classifiers' predictions on test data
            testset_outputs_Tier1 = np.hstack((testset_output_C1,testset_output_C2,testset_output_C3,testset_output_C4))

            # Based on predictions on test data, of all Tier-1 base classifiers , it would use the meta classifier to predict labels of test data
            testset_outputs_meta = self.meta_classifier.predict(testset_outputs_Tier1)
            # Round all predictions of meta classifier xgboost
            testset_outputs_meta = np.round(testset_outputs_meta)

            # Store all metrics of cross-validation in different lists
            test_cv_accuracy.append(accuracy_score(self.test_target,testset_outputs_meta))
            test_cv_recall.append(recall_score(self.test_target,testset_outputs_meta))
            test_cv_precision.append(precision_score(self.test_target,testset_outputs_meta))

        # Convert lists into numpy arrays, since only numpy arrays can be used to calculate mean values, min values, max values and std values
        test_cv_accuracy = np.array(test_cv_accuracy)
        test_cv_recall = np.array(test_cv_recall)
        test_cv_precision = np.array(test_cv_precision)

        time_end = time.time() # end time
        print("\nTime cost: ", time_end - time_start, "seconds")

        cv_scores = {"test_accuracy":test_cv_accuracy,"test_precision_weighted":test_cv_recall,"test_recall_weighted":test_cv_precision}
        return cv_scores




# 8. -*- Construct different models, and perform the cross-validation for sentiment classification (sentiment analysis)  -*-
def sentiment_classification_single_classifier_by_cv():
    # return data X and labels Y
    X, Y = train_text_data_creation()

    time_start = time.time()
    # model = LogisticRegression(solver="lbfgs")   # Logistic Regression model
    # model =  MultinomialNB()   # Naive Bayes model
    # model = LinearSVC(penalty = "l2")
    # model = ExtraTreeClassifier()
    # model = XGBClassifier()
    model = RandomForestClassifier(n_estimators=100)
    # model = SGDClassifier(max_iter=1000, tol=1e-3)

    # just return different cv scores based on cross-validation test sets (5-fold cross-validation)
    cv_scores_model = cross_validate(model, X, Y, cv=StratifiedKFold(n_splits=5, random_state=0),
                                     return_train_score=False,
                                     scoring=["accuracy", "precision_weighted", "recall_weighted"])

    time_end = time.time()
    print("\nTime cost: ", time_end - time_start, "seconds")
    print_cv_scores_summary(cv_scores_model)




if __name__ == "__main__":
    # return data X and labels Y
    X, Y = train_text_data_creation()
    sg = stacked_generalization(data=X, target=Y)
    cv_scores = sg.train_stacked_generalization_CV()
    print_cv_scores_summary(cv_scores)

    # sentiment_classification_single_classifier_by_cv()

