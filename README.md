# Document Analysis for the Federalist Papers

I implemented an authorship attribution algorithm for the Federalist Papers dataset which contains historically important papers written by Hamilton and Madison, and I wanted to classify the author from the text. 

Fist, I preprocessed the data, such as removing non-letter characters, removing stopwords, and stemming words. 

Then, I made a dictionary that contained all the unique words along with the number of times they appeared in the corpus. 

Then, I removed words that are irrelevant to classification using feature selection. 

After that, I used the dictionary to do classification. I tried a few approaches, like Naive Bayes, Decision Trees, Ridge Regression and Lasso.
