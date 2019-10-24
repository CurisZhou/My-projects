1. The packages that need to be installed:
pandas, numpy, nltk, sklearn, seaborn, matplotlib, ftfy, gensim
(collections, re and time packages are included by Python)

2. Since, the code file reads fake news data from relative path. Please put the code file and the csv file of data in a same folder. The name of csv data file is "fake_news_data.csv".

3. In the code file, you just need to run the function "train_model_by_cv()", the function "fake_news_to_vectors()" and the function "train_model_by_vectors()". After that, the results of model performances by using different linguistic n-grams features or word embedding to do the fake news classification, could be outputed. 

4. After running the function fake_news_to_vectors(), the trained word2vec CBOW model would be saved. Since, the function "train_model_by_vectors()" in code file would load trained CBOW model from relative path. Please put the code file and the saved CBOW model file in a same folder. The name of saved CBOW model would be "fake_news_word2vec_model.model".

5. There are two types of code files, one is the Pyhton file, and another is the jupyter notebook file. The names of these two code files are "code_fake_news_detection".