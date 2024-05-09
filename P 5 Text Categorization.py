#A) Implement a text classification algorithm (e.g., Naive Bayes or Support Vector Machines).
#B) Train the classifier on a labelled dataset and evaluate its performance.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load the CSV file
df = pd.read_csv(r'C:\Users\admin\Desktop\sem 6\IR\datasets.csv')
data = df["covid"] + "" + df["fever"]
X = data.astype(str)  # Test data
y = df['flu']  # Labels

# Splitting the data into training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Converting data into bag-of-data format to train the model
vectorizer = CountVectorizer()

# initializing the converter
X_train_counts = vectorizer.fit_transform(X_train)

# converting the training data
X_test_counts = vectorizer.transform(X_test)

# converting the test data

# using and training the multinomial model of naive bayes algorithm
classifier = MultinomialNB()  # initializing the classifier
classifier.fit(X_train_counts, y_train)  # training the classifier

# loading another dataset to test if the model is working properly
data1 = pd.read_csv(r"C:\Users\Administrator\Documents\Sem 6\IR\Test.csv")
new_data = data1["covid"] + "" + data1["fever"]
new_data_counts = vectorizer.transform(new_data.astype(str))  # converting the new data

# making the model to predict the results for new dataset
predictions = classifier.predict(new_data_counts)

# Output the results
new_data = predictions
print(new_data)

# retrieving the accuracy and classification report
accuracy = accuracy_score(y_test, classifier.predict(X_test_counts))
print(f"\nAccuracy: {accuracy:.2f}")
print("Classification Report: ")
print(classification_report(y_test, classifier.predict(X_test_counts)))

# Convert the predictions to a DataFrame
predictions_df = pd.DataFrame(predictions, columns=['flu_prediction'])

# concatenate the original DataFrame with the predictions DataFrame
data1 = pd.concat([data1, predictions_df], axis=1)

# write the DataFrame back to CSV
data1.to_csv(r"C:\Users\admin\Desktop\sem 6\IR\datasets.csv", index=False)
