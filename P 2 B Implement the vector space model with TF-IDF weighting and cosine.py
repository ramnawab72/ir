from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
# Import necessary libraries
import nltk  # Import NLTK to download stopwords
from nltk.corpus import stopwords  # Import stopwords from NLTK
import numpy as np  # Import NumPy library
from numpy.linalg import norm  # Import norm function from NumPy's linear algebra module

# Define the training and test sets of text documents
train_set = ["The sky is blue.", "The sun is bright."]  # Documents
test_set = ["The sun in the sky is bright."]  # Query

# Get the stopwords for English language from NLTK
nltk.download('stopwords')
stopWords = stopwords.words('english')

# Initialize CountVectorizer and TfidfTransformer objects
vectorizer = CountVectorizer(stop_words=stopWords)  # CountVectorizer to convert text to matrix of token counts
transformer = TfidfTransformer()  # TfidfTransformer to convert matrix of token counts to TF-IDF representation

# Convert the training and test sets to arrays of TF-IDF features
trainVectorizerArray = vectorizer.fit_transform(train_set).toarray()  # Fit-transform training set
testVectorizerArray = vectorizer.transform(test_set).toarray()  # Transform test set

# Display the TF-IDF arrays for training and test sets
print('Fit Vectorizer to train set', trainVectorizerArray)
print('Transform Vectorizer to test set', testVectorizerArray)

# Define a lambda function to calculate cosine similarity between vectors
cx = lambda a, b: round(np.inner(a, b) / (norm(a) * norm(b)), 3)

# Iterate through each vector in the training set
for vector in trainVectorizerArray:
    print(vector)  # Display each vector in the training set

# Iterate through each vector in the test set
for testV in testVectorizerArray:
    print(testV)  # Display each vector in the test set
    cosine = cx(vector, testV)  # Calculate cosine similarity between vectors
    print(cosine)  # Display the cosine similarity

# Fit the transformer to the training set and transform it to TF-IDF representation
transformer.fit(trainVectorizerArray)
print()
print(transformer.transform(trainVectorizerArray).toarray())

# Fit the transformer to the test set and transform it to TF-IDF representation
transformer.fit(testVectorizerArray)
print()
tfidf = transformer.transform(testVectorizerArray)
print(tfidf.todense())
