#################
# Setup
#################

#setwd("")

# first include the relevant libraries
# note that a loading error might mean that you have to
# install the package into your R distribution.
# Use the package installer and be sure to install all dependencies
library(tm)

#################
# Preprocess the data to remove non-letter characters, remove stopwords, and stem words.
#################

##########################################
# This code uses tm to preprocess the papers into a format useful for NB
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
##########################################

preprocess.directory("fp_hamilton_train")
preprocess.directory("fp_hamilton_test")
preprocess.directory("fp_madison_train")
preprocess.directory("fp_madison_test")

#################
# Load each of the Federalist Papers from their corresponding directory into my workspace
#################

##########################################
# To read in data from the directories:
# Partially based on code from C. Shalizi
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
##########################################

hamilton.train = read.directory("fp_hamilton_train_clean")
hamilton.test = read.directory("fp_hamilton_test_clean")
madison.train = read.directory("fp_madison_train_clean")
madison.test = read.directory("fp_madison_test_clean")

#################
# Use all of the files (training and testing for both authors) to make a dictionary
#################

##########################################
# Make dictionary sorted by number of times a word appears in corpus 
# (useful for using commonly appearing words as factors)
# NOTE: Use the *entire* corpus: training, testing, spam and ham
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
##########################################

dictionary = make.sorted.dictionary.df(c(hamilton.train, hamilton.test, madison.train, madison.test))

#################
# Create counts for each dictionary word in all documents and place them in a document (rows) by word (columns) matrix
#################

##########################################
# Make a document-term matrix, which counts the number of times each 
# dictionary element is used in a document
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
    }
  }
  return(dtm);
}
##########################################

dtm.hamilton.train = make.document.term.matrix(hamilton.train, dictionary)
dtm.hamilton.test = make.document.term.matrix(hamilton.test, dictionary)
dtm.madison.train = make.document.term.matrix(madison.train, dictionary)
dtm.madison.test = make.document.term.matrix(madison.test, dictionary)

#################
# Compute naive Bayes probabilities with mu = smoothing parameter
#################

##########################################
make.log.pvec <- function(dtm,mu){
  # Sum up the number of instances per word
  pvec.no.mu <- colSums(dtm)
  # Sum up number of words
  n.words <- sum(pvec.no.mu)
  # Get dictionary size
  dic.len <- length(pvec.no.mu)
  # Incorporate mu and normalize
  log.pvec <- log(pvec.no.mu + mu) - log(mu*dic.len + n.words)
  return(log.pvec)
}
##########################################

mu = 1 / nrow(dictionary)
logp.hamilton.train = make.log.pvec(dtm.hamilton.train, mu)
logp.hamilton.test = make.log.pvec(dtm.hamilton.test, mu)
logp.madison.train = make.log.pvec(dtm.madison.train, mu)
logp.madison.test = make.log.pvec(dtm.madison.test, mu)

#################
# Naive Bayes classifier
#################

naive.bayes = function(logp.hamilton.train, logp.madison.train, log.prior.hamilton, log.prior.madison , dtm.test){
  likelihood.hamilton = log.prior.hamilton + dtm.test %*% matrix(logp.hamilton.train)
  likelihood.madison = log.prior.madison + dtm.test %*% matrix(logp.madison.train)
  authorship = as.integer(likelihood.hamilton >= likelihood.madison) 
  return(authorship)
}

#################
# Set mu=1/number of words, classify test papers
#################

log.prior.hamilton = log(nrow(dtm.hamilton.train) / (nrow(dtm.hamilton.train) + nrow(dtm.madison.train)))
log.prior.madison = log(nrow(dtm.madison.train) / (nrow(dtm.hamilton.train) + nrow(dtm.madison.train)))
authorship.hamilton = naive.bayes(logp.hamilton.train, logp.madison.train, log.prior.hamilton, log.prior.madison, dtm.hamilton.test)
authorship.madison = naive.bayes(logp.hamilton.train, logp.madison.train, log.prior.hamilton, log.prior.madison, dtm.madison.test)
ntest.hamilton = nrow(dtm.hamilton.test)
ntest.madison = nrow(dtm.madison.test)
correct = (sum(authorship.hamilton == 1) + sum(authorship.madison == 0)) / (ntest.hamilton + ntest.madison)
true.positive = sum(authorship.hamilton == 1) / ntest.hamilton
true.negative = sum(authorship.madison == 0) / ntest.madison
false.positive = sum(authorship.madison == 1) / ntest.madison
false.negative = sum(authorship.hamilton == 0) / ntest.hamilton
correct
true.positive
true.negative
false.positive
false.negative

#################
# use 5-fold cross-validation over the training set to search the parameter set mu
#################

dict2 = make.sorted.dictionary.df(c(hamilton.train, madison.train))
D = nrow(dict2)
mu = c(1/(10*D), 1/D, 10/D, 100/D, 1000/D)
dtm.train = rbind(dtm.hamilton.train, dtm.madison.train)
n.hamilton.train = nrow(dtm.hamilton.train)
n.madison.train = nrow(dtm.madison.train)
n.train = nrow(dtm.train)
ind.all = 1:n.train
k = 5
set.seed(1)
ind = matrix(sample(ind.all, n.train), nrow = k, byrow = T)    # k fold training set
correct.a = matrix(nrow = 5, ncol = 5)
false.negative.a = matrix(nrow = 5, ncol = 5)
false.positive.a = matrix(nrow = 5, ncol = 5)
for(i in 1:5){
  for(j in 1:5){
    hamilton.ind = ind[j,which(ind[j,]<=n.hamilton.train)]
    madison.ind = ind[j,which(ind[j,]>n.hamilton.train)]-n.hamilton.train
    logp.hamilton.train.a = make.log.pvec(dtm.hamilton.train[-hamilton.ind,], mu[i])
    logp.madison.train.a = make.log.pvec(dtm.madison.train[-madison.ind,], mu[i])
    ntest.hamilton.a = length(hamilton.ind)
    ntest.madison.a = length(madison.ind)
    log.prior.hamilton.a = log((n.hamilton.train-ntest.hamilton.a) / (n.train-ntest.hamilton.a-ntest.madison.a))
    log.prior.madison.a = log((n.madison.train-ntest.madison.a) / (n.train-ntest.hamilton.a-ntest.madison.a))
    dtm.hamilton.test.a = dtm.hamilton.train[hamilton.ind,]
    dtm.madison.test.a = dtm.madison.train[madison.ind,]
    authorship.hamilton.a = naive.bayes(logp.hamilton.train.a, logp.madison.train.a, log.prior.hamilton.a, log.prior.madison.a, dtm.hamilton.test.a)
    authorship.madison.a = naive.bayes(logp.hamilton.train.a, logp.madison.train.a, log.prior.hamilton.a, log.prior.madison.a, dtm.madison.test.a)
    correct.a[i,j] = (sum(authorship.hamilton.a == 1) + sum(authorship.madison.a == 0)) / (ntest.hamilton.a + ntest.madison.a)
    false.positive.a[i,j] = sum(authorship.madison.a == 1) / ntest.madison.a
    false.negative.a[i,j] = sum(authorship.hamilton.a == 0) / ntest.hamilton.a
  }
}
correct.mu.a = rowMeans(correct.a)
false.positive.mu.a = rowMeans(false.positive.a)
false.negative.mu.a = rowMeans(false.negative.a)
correct.mu.a
false.negative.mu.a
false.positive.mu.a
plot(correct.mu.a, xlab = "mu", ylab = "correct classification rate", main = "correct classification rate over different mu", type = "b")
plot(false.negative.mu.a, xlab = "mu", ylab = "false negative rate", main = "false negative rate over different mu", type = "b")
plot(false.positive.mu.a, xlab = "mu", ylab = "false positive rate", main = "false positive rate over different mu", type = "b")
