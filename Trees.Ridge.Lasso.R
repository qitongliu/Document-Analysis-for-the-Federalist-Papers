#################
# Preprocess data
#################
library(tm)
preprocess.directory = function(dirname){
  
  # the directory must have all the relevant text files
  ds = DirSource(dirname)
  # Corpus will make a tm document corpus from this directory
  fp = Corpus( ds )
  # inspect to verify
  # inspect(fp[1])
  # another useful command
  # identical(fp[[1]], fp[["Federalist01.txt"]])
  # now let us iterate through and clean this up using tm functionality
  # make all words lower case
  fp = tm_map( fp , content_transformer(tolower));
  # remove all punctuation
  fp = tm_map( fp , removePunctuation);
  # remove stopwords like the, a, and so on.	
  fp = tm_map( fp, removeWords, stopwords("english"));
  # remove stems like suffixes
  fp = tm_map( fp, stemDocument)
  # remove extra whitespace
  fp = tm_map( fp, stripWhitespace)	
  # now write the corpus out to the files for our future use.
  # MAKE SURE THE _CLEAN DIRECTORY EXISTS
  writeCorpus( fp , sprintf('%s_clean',dirname) )
}
preprocess.directory("fp_hamilton_train")
preprocess.directory("fp_hamilton_test")
preprocess.directory("fp_madison_train")
preprocess.directory("fp_madison_test")

read.directory <- function(dirname) {
  # Store the infiles in a list
  infiles = list();
  # Get a list of filenames in the directory
  filenames = dir(dirname,full.names=TRUE);
  for (i in 1:length(filenames)){
    infiles[[i]] = scan(filenames[i],what="",quiet=TRUE);
  }
  return(infiles)
}
hamilton.train <- read.directory('fp_hamilton_train_clean')
hamilton.test <- read.directory('fp_hamilton_test_clean')
madison.train <- read.directory('fp_madison_train_clean')
madison.test <- read.directory('fp_madison_test_clean')

make.sorted.dictionary.df <- function(infiles){
  # This returns a dataframe that is sorted by the number of times 
  # a word appears
  
  # List of vectors to one big vetor
  dictionary.full <- unlist(infiles) 
  # Tabulates the full dictionary
  tabulate.dic <- tabulate(factor(dictionary.full)) 
  # Find unique values
  dictionary <- unique(dictionary.full) 
  # Sort them alphabetically
  dictionary <- sort(dictionary)
  dictionary.df <- data.frame(word = dictionary, count = tabulate.dic)
  sort.dictionary.df <- dictionary.df[order(dictionary.df$count,decreasing=TRUE),];
  return(sort.dictionary.df)
}
dictionary <- make.sorted.dictionary.df(c(hamilton.train,hamilton.test,madison.train,madison.test))

make.document.term.matrix <- function(infiles,dictionary){
  # This takes the text and dictionary objects from above and outputs a 
  # document term matrix
  num.infiles <- length(infiles);
  num.words <- nrow(dictionary);
  # Instantiate a matrix where rows are documents and columns are words
  dtm <- mat.or.vec(num.infiles,num.words); # A matrix filled with zeros
  for (i in 1:num.infiles){
    num.words.infile <- length(infiles[[i]]);
    infile.temp <- infiles[[i]];
    for (j in 1:num.words.infile){
      ind <- which(dictionary == infile.temp[j])[[1]];
      # print(sprintf('%s,%s', i , ind))
      dtm[i,ind] <- dtm[i,ind] + 1;
      #print(c(i,j))
    }
  }
  return(dtm);
}

dtm.hamilton.train <- make.document.term.matrix(hamilton.train,dictionary)
dtm.hamilton.test <- make.document.term.matrix(hamilton.test,dictionary)
dtm.madison.train <- make.document.term.matrix(madison.train,dictionary)
dtm.madison.test <- make.document.term.matrix(madison.test,dictionary)

# For each term matrix, create a vector of 0's and 1's to indicate the author of each document
train.response = c(rep(1, nrow(dtm.hamilton.train)), rep(0, nrow(dtm.madison.train)))
test.response = c(rep(1, nrow(dtm.hamilton.test)), rep(0, nrow(dtm.madison.test)))
# Combine the document term matrices and vectors of 0's and 1's to create two data frames
# one that includes all training data and one that includes all testing data
train = cbind(rbind(dtm.hamilton.train, dtm.madison.train), train.response)
test = cbind(rbind(dtm.hamilton.test, dtm.madison.test), test.response)
colnames(train) = c(as.vector(dictionary$word), "y")
colnames(test) = c(as.vector(dictionary$word), "y")
train = as.data.frame(train)
test = as.data.frame(test)

#################
# Use tree classification with Gini impurity coeficient splits to predict the author
#################
library(rpart)
tree.gini = rpart(y~., data = train, method = "class")
fit.gini = predict(tree.gini, test, type = "class")
correct.gini = sum(fit.gini == test.response) / length(test.response)
false.negative.gini = sum(fit.gini == 0 & test.response == 1) / nrow(dtm.hamilton.test)
false.positive.gini = sum(fit.gini == 1 & test.response == 0) / nrow(dtm.madison.test)
correct.gini
false.negative.gini
false.positive.gini
plot(tree.gini, margin = 0.2, main = "Gini Tree")
text(tree.gini, use.n = TRUE)

#################
# Use tree classification with information gain splits to predict the author
#################
tree.info = rpart(y~., data = train, method = "class", parms = list(split = "information"))
fit.info = predict(tree.info, test, type = "class")
correct.info = sum(fit.info == test.response) / length(test.response)
false.negative.info = sum(fit.info == 0 & test.response == 1) / nrow(dtm.hamilton.test)
false.positive.info = sum(fit.info == 1 & test.response == 0) / nrow(dtm.madison.test)
correct.info
false.negative.info
false.positive.info
plot(tree.info, margin = 0.2, main = "Information Tree")
text(tree.info, use.n = TRUE)

#################
# Create centered and scaled versions of document term matrices (not center and scale the labels)
#################
dtm.train = rbind(dtm.hamilton.train, dtm.madison.train)
train.scaled = scale(dtm.train)
train.center = colMeans(dtm.train)
train.sd = apply(dtm.train, 2, sd)
test.scaled = scale(rbind(dtm.hamilton.test, dtm.madison.test), center = train.center, scale = train.sd)
colnames(train.scaled) = as.vector(dictionary$word)
colnames(test.scaled) = as.vector(dictionary$word)
train.scaled[is.na(train.scaled)] = 0
test.scaled[is.na(test.scaled)] = 0

#################
# Use Ridge regression to predict the author
#################
library(glmnet)
cv.fit.ridge = cv.glmnet(train.scaled, train.response, family="binomial", alpha = 0)
y.pred.ridge = predict(cv.fit.ridge, test.scaled, type = "class")
correct.ridge = sum(y.pred.ridge == test.response) / length(test.response)
false.negative.ridge = sum(y.pred.ridge == 0 & test.response == 1) / nrow(dtm.hamilton.test)
false.positive.ridge = sum(y.pred.ridge == 1 & test.response == 0) / nrow(dtm.madison.test)
correct.ridge
false.negative.ridge
false.positive.ridge
ind = which(cv.fit.ridge$lambda == cv.fit.ridge$lambda.min)
fit.ridge = glmnet(train.scaled, train.response, family="binomial", alpha = 0)
fit.ridge$beta[,ind][order(abs(fit.ridge$beta[,ind]), decreasing = T)[1:10]]

#################
# Use Lasso regression to predict the author
#################
set.seed(2)
cv.fit.lasso = cv.glmnet(train.scaled, train.response, family="binomial", alpha = 1)
y.pred.lasso = predict(cv.fit.lasso, test.scaled, type = "class")
correct.lasso = sum(y.pred.lasso == test.response) / length(test.response)
false.negative.lasso = sum(y.pred.lasso == 0 & test.response == 1) / nrow(dtm.hamilton.test)
false.positive.lasso = sum(y.pred.lasso == 1 & test.response == 0) / nrow(dtm.madison.test)
correct.lasso
false.negative.lasso
false.positive.lasso
ind = which(cv.fit.lasso$lambda == cv.fit.lasso$lambda.min)
fit.lasso = glmnet(train.scaled, train.response, family="binomial", alpha = 0)
fit.lasso$beta[,ind][order(abs(fit.lasso$beta[,ind]), decreasing = T)[1:10]]

## use feature selection to remove features that are irrelevant to classification
#################
# Compute the mutual information for all features
# Use this to select the top n features as a dictionary, n={200, 500, 1000, 2500}
#################
make.pvec <- function(dtm,mu){
  # Sum up the number of instances per word
  pvec.no.mu <- colSums(dtm)
  # Sum up number of words
  n.words <- sum(pvec.no.mu)
  # Get dictionary size
  dic.len <- length(pvec.no.mu)
  # Incorporate mu and normalize
  pvec <- (pvec.no.mu + mu) / (mu*dic.len + n.words)
  return(pvec)
}

n = c(200, 500, 1000, 2500)
correct.gini = NULL
false.negative.gini = NULL
false.positive.gini = NULL
correct.info = NULL
false.negative.info = NULL
false.positive.info = NULL
correct.ridge = NULL
false.negative.ridge = NULL
false.positive.ridge = NULL
correct.lasso = NULL
false.negative.lasso = NULL
false.positive.lasso = NULL
for(i in 1:4){
  mu = 1 / n[i]
  prob.hamilton = make.pvec(rbind(dtm.hamilton.train, dtm.hamilton.test), mu)
  prob.madison = make.pvec(rbind(dtm.madison.train, dtm.madison.test), mu)
  prob = make.pvec(rbind(dtm.hamilton.train, dtm.hamilton.test, dtm.madison.train, dtm.madison.test), mu)
  total = nrow(dtm.hamilton.train) + nrow(dtm.hamilton.test) + nrow(dtm.madison.train) + nrow(dtm.madison.test)
  prior.hamilton = (nrow(dtm.hamilton.train) + nrow(dtm.hamilton.test)) / total
  prior.madison = (nrow(dtm.madison.train) + nrow(dtm.madison.test)) / total
  mutual.info = prob.hamilton * prior.hamilton * log(prob.hamilton/prob) + (1-prob.hamilton) * prior.hamilton * log((1-prob.hamilton)/(1-prob))
  mutual.info = mutual.info + prob.madison * prior.madison * log(prob.madison/prob) + (1-prob.madison) * prior.madison * log((1-prob.madison)/(1-prob))
  
  ind = order(mutual.info, decreasing = T)[1:n[i]]
  
  ### Gini
  train.sub = train[,c(ind, ncol(train))]
  test.sub = test[,c(ind, ncol(test))]
  tree.gini = rpart(y~., data = train.sub, method = "class")
  fit.gini = predict(tree.gini, test.sub, type = "class")
  correct.gini[i] = sum(fit.gini == test.response) / length(test.response)
  false.negative.gini[i] = sum(fit.gini == 0 & test.response == 1) / nrow(dtm.hamilton.test)
  false.positive.gini[i] = sum(fit.gini == 1 & test.response == 0) / nrow(dtm.madison.test)
  
  ### Info
  tree.info = rpart(y~., data = train.sub, method = "class", parms = list(split = "information"))
  fit.info = predict(tree.info, test.sub, type = "class")
  correct.info[i] = sum(fit.info == test.response) / length(test.response)
  false.negative.info[i] = sum(fit.info == 0 & test.response == 1) / nrow(dtm.hamilton.test)
  false.positive.info[i] = sum(fit.info == 1 & test.response == 0) / nrow(dtm.madison.test)
  
  ### Ridge
  train.scaled.sub = train.scaled[,ind]
  test.scaled.sub = test.scaled[,ind]
  cv.fit.ridge = cv.glmnet(train.scaled.sub, train.response, family="binomial", alpha = 0)
  y.pred.ridge = predict(cv.fit.ridge, test.scaled.sub, type = "class")
  correct.ridge[i] = sum(y.pred.ridge == test.response) / length(test.response)
  false.negative.ridge[i] = sum(y.pred.ridge == 0 & test.response == 1) / nrow(dtm.hamilton.test)
  false.positive.ridge[i] = sum(y.pred.ridge == 1 & test.response == 0) / nrow(dtm.madison.test)
  
  ### Lasso
  set.seed(2)
  cv.fit.lasso = cv.glmnet(train.scaled.sub, train.response, family="binomial", alpha = 1)
  y.pred.lasso = predict(cv.fit.lasso, test.scaled.sub, type = "class")
  correct.lasso[i] = sum(y.pred.lasso == test.response) / length(test.response)
  false.negative.lasso[i] = sum(y.pred.lasso == 0 & test.response == 1) / nrow(dtm.hamilton.test)
  false.positive.lasso[i] = sum(y.pred.lasso == 1 & test.response == 0) / nrow(dtm.madison.test)
}

plot(correct.gini, main = "Proportion of Classified Correctly", xlab = "n", ylab = "proportion", type = "b", col = 1)
lines(correct.info, type = "b", col = 2)
lines(correct.ridge, type = "b", col = 3)
lines(correct.lasso, type = "b", col = 4)
legend("bottomleft",legend = c("Gini", "Information", "Ridge", "Lasso"), col=1:4, pch = 1, cex = 0.75)

plot(false.negative.gini, main = "Proportion of False Negatives", xlab = "n", ylab = "proportion", type = "b", col = 1)
lines(false.negative.info, type = "b", col = 2)
lines(false.negative.ridge, type = "b", col = 3)
lines(false.negative.lasso, type = "b", col = 4)
legend("bottomleft",legend = c("Gini", "Information", "Ridge", "Lasso"), col=1:4, pch = 1, cex = 0.75)

plot(false.positive.gini, main = "Proportion of False Positives", xlab = "n", ylab = "proportion", type = "b", col = 1, ylim = c(-0.1,0.6))
lines(false.positive.info, type = "b", col = 2)
lines(false.positive.ridge, type = "b", col = 3)
lines(false.positive.lasso, type = "b", col = 4)
legend("topleft",legend = c("Gini", "Information", "Ridge", "Lasso"), col=1:4, pch = 1, cex = 0.75)

