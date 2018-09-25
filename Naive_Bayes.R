#___________________________________________________________________________________________________________________________________
# 1.) This is an example workflow that predicts whether someone makes over 50k/year using the Naive Bayes algorithm.
# 
# 2.) This is an example workflow that predicts the author of text using the Naive Bayes algorithm along with 
#     other natural language processing (NLP) techniques.
#
# By: James Bowers
# 
# The data used in this code was taken from the UCI Machine Learning Repository: 
# https://archive.ics.uci.edu/ml/datasets/adult
# https://archive.ics.uci.edu/ml/datasets/Victorian+Era+Authorship+Attribution
# 
# Source Data Citation: 
# Dua, D. and Karra Taniskidou, E. (2017). UCI Machine Learning Repository 
# Irvine, CA: University of California, School of Information and Computer Science.
#
# Gungor, A. (2018). Fifty Victorian Era Novelists Authorship Attribution Data. 
# IUPUI University Library. http://dx.doi.org/10.7912/D2N65J
# 
#___________________________________________________________________________________________________________________________________

library(e1071)
library(caret)
library(dplyr)
library(readtext)
library(quanteda)
library(here)

options(scipen=999)

## 1.) NAIVE BAYES - CLASSIFICATION - BINARY DEPENDENT VARIABLE ####

### Get Data ####

#### get both URLs
trainUrl <- "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
testUrl <- "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"

#### store and inspect data
trainData <- read.csv(trainUrl, header=FALSE, stringsAsFactors=TRUE, sep=',') # naiveBayes needs factors for categorical variables
head(trainData)
testData <- read.csv(testUrl, header=FALSE, stringsAsFactors=TRUE, sep=',')
head(testData)

write.csv(trainData, file = "Census_Train.csv", row.names = FALSE)
write.csv(testData, file = "Census_Test.csv", row.names = FALSE)

#### combine datasets into a single data frame
allData <- rbind(trainData, testData[-1,]) # remove the 1st row in TestData due to bad data
head(allData,30)
str(allData)

### Cleanse ####

#### rename columns
colnames(allData) <- c("age","workclass","fnlwgt","education","education_num","marital_status","occupation",
                       "relationship","race","sex","capital_gain","capital_loss","hours_per_week","native_country","class")

#### convert data types
allData$age <- as.integer(allData$age)

#### clean up dirty data

##### convert ? to NA
x <- allData == " ?" 
sum(x)              # how many ? are there 
is.na(allData) <- x # convert ? to NA

##### remove rows with NA
allData <- allData[complete.cases(allData),]


##### clean up the "class" variable
allData$class <- 
  case_when(allData$class %in% c(" <=50K."," <=50K") ~ "<=50K",
            allData$class %in% c(" >50K.", " >50K" ) ~ ">50K")

allData$class <- factor(allData$class)

##### trim leading and trailing spaces
trim <- function (x) gsub("^\\s+|\\s+$", "", x)   
allData[,c(2,4,6:10,14)] <- apply(allData[,c(2,4,6:10,14)], 2, trim) 

allData
str(allData)


### Split Data ####
trainIndex <- createDataPartition(allData$class, p=0.8, list=FALSE) # pkg: caret
trainIndex
trainData <- allData[trainIndex,]
testData <- allData[-trainIndex,]

nrow(testData)/nrow(trainData) # ratio of test to train  

table(trainData$native_country)
#### Build & Evaluate Model 1 ####
nb_fit1 <- naiveBayes(class ~ . , data=trainData)

pred1 <- predict(nb_fit1, testData[,-15], type="raw")

testData1 <- testData
testData1$pred_conf <- round(pred1[,1], digits=10)

testData1$pred_class <-  ifelse(testData1$pred_conf >= .99, "<=50K", ">50K")
table(testData1$pred_class)

cm_nb1 <- confusionMatrix(data=testData1$pred_class, reference=testData1[,15], positive = ">50K")
cm_nb1


#### Build & Evaluate Model 2 ####
caret.control <- trainControl(method="repeatedcv", number=10, repeats=1)

nb_fit2 <- train(class ~ ., 
                 data = trainData,
                 method = "naive_bayes",
                 trControl = caret.control,
                 tuneLength=5)

pred2 <- predict(nb_fit2, testData[,-15], type = "prob") # or use class="prob" to get percentages so we can adjust threshold manually. 

testData2 <- testData
testData2$pred_conf <- round(pred2[,1], digits=10)


testData2$pred_class <-  ifelse(testData2$pred_conf >= .999999999, "<=50K", ">50K")
table(testData2$pred_class)

cm_nb2 <- confusionMatrix(data=testData2$pred_class, reference=testData2[,15], positive = ">50K")
cm_nb2


#### Predict New Data ####
newData <- data.frame(age=55, workclass="Private", fnlwgt=150000, education="Masters", education_num=14, marital_status="Never-married", occupation="Prof-specialty", relationship="Unmarried", 
                      race="White", sex="Male", capital_gain=2000, capital_loss=0, hours_per_week=42, native_country="United-States")

predict(nb_fit1, newData)
predict(nb_fit2, newData)








## 2.) NAIVE BAYES - CLASSIFICATION - MULTIPLE DEPENDENT VARIABLES ####

### Get Data ####

#### download .zip file and store in working directory
dataUrl <- "https://archive.ics.uci.edu/ml/machine-learning-databases/00217/C50.zip"
download.file(dataUrl, "C50.zip")

#### unzip the contents of .zip folder and store in working directory
zipFilePath <- here("C50.zip")
unzip(zipFilePath)

#### get list of authors (also the name of individual folders within train/test)
trainAuthorList <- list.files(path=here("C50train"))
testAuthorList <- list.files(path=here("C50test"))

#### initialize data frame for train and test data
rawTrainData <- data.frame(Author=NULL,doc_id=NULL, text=NULL)
rawTestData <- data.frame(Author=NULL,doc_id=NULL, text=NULL)

#### loop through all folders and retrieve the text for training data
for (i in 1:length(trainAuthorList)) {
  require(readtext)
  filePath <- sprintf(here("C50train","%s"), trainAuthorList[i])
  authorText <- readtext(filePath)
  rawTrainData <- rbind(rawTrainData, data.frame(Author=trainAuthorList[i], authorText))
}

#### loop through all folders and retrieve the text for test data
for (i in 1:length(testAuthorList)) {
  require(readtext)
  filePath <- sprintf(here("C50test","%s"), testAuthorList[i])
  authorText <- readtext(filePath)
  rawTestData <- rbind(rawTestData, data.frame(Author=testAuthorList[i], authorText))
}

rawAllData <- rbind(rawTrainData, rawTestData)

### Create a Clean Corpus ####

#### retrieve clean tokens (split text by word, only keep alpha characters)
allTokens <- tokens(rawAllData$text, what="word",
                    remove_numbers=TRUE, remove_punct = TRUE,
                    remove_symbols = TRUE, remove_hyphens = TRUE)

allTokens[[85]]

#### convert to lower case
allTokens <- tokens_tolower(allTokens)

allTokens[[85]]

#### remove stopwords
allTokens <- tokens_select(allTokens, stopwords(),
                           selection = "remove", verbose = TRUE) # edit the stopword list for your problem.
allTokens[[85]]

#### stem the tokens
allTokens <- tokens_wordstem(allTokens, language = "english")

allTokens[[85]]

### Create Document-Feature Matrix ####
allTokensDFM <- dfm(allTokens, tolower = FALSE, verbose = TRUE)

#### transform to a matrix to inspect
allTokensMatrix <- as.matrix(allTokensDFM)
View(allTokensMatrix[1:20, 1:100]) 
dim(allTokensMatrix)

### Optional: DFM Feature Reduction ####
allTokensFreqDFM <- dfm_trim(allTokensDFM, min_termfreq = 30, min_docfreq = 5)

paste0("# of Columns: ",ncol(allTokensFreqDFM),"  |  Features reduced by ",
       round( 100 * (1-ncol(allTokensFreqDFM) / ncol(allTokensDFM)),1),"%")

### Create Labeled Train Data Set #### 
allTokensDF <- cbind(Label = rawAllData$Author, data.frame(allTokensFreqDFM)[ ,-1]) # removed the unique document identifier

allTokensDF[1:3,1:5]


### Split Data ####
trainIndex <- createDataPartition(allTokensDF$Label, p=0.8, list=FALSE)
trainIndex

trainData <- allTokensDF[trainIndex,]
testData <- allTokensDF[-trainIndex,]

nrow(testData)/nrow(trainData) # ratio of test to train


#### Build & Evaluate Model 1 ####
nb_fit1 <- naiveBayes(Label ~ . , data=trainData)

pred1 <- predict(nb_fit1, testData[,-1], type="class")

testData1 <- testData
testData1$Label_pred <- pred1

table(testData1$Label)
table(testData1$Label_pred)

cm_nb1 <- confusionMatrix(data=testData1$Label_pred, reference=testData1$Label)

round(cm_nb1$overall,4)[1] # overall accuracy
cm_nb1


### Predict New Data ####

#### load new "article"
newText <- data.frame(text="This is an example of an article on the British War. The man responsible was 
                      quite the sleuth, his name was Sherlock Holmes. Sherlock had many vices, some of 
                      which include the violin, the pipe, and of course, solving the most mysterious of 
                      crimes to ever grace the presence of her majesty, the queen.  Ah Yes, the Queen was 
                      a fan as well. Her and sherlock traveled a great many places, including the vast 
                      plains of Idaho, where they ate not just one, but four bison.")
newText$text <- as.character(newText$text) 

#### convert article to clean DFM using the identical process as before
newText <- tokens(newText$text, what="word",
                  remove_numbers=TRUE, remove_punct = TRUE,
                  remove_symbols = TRUE, remove_hyphens = TRUE)
newText <- tokens_tolower(newText)
newText <- tokens_select(newText, stopwords(),
                         selection = "remove", verbose = TRUE)
newText <- tokens_wordstem(newText, language = "english")
newText <- dfm(newText, tolower = FALSE, verbose = TRUE)

#### predict the author
predict(nb_fit1, newText, type="class")
