# SMS Spam Detection

dataset: [UCC Irvine Machine Learning Repository](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
license: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)

## Goal: predict binary classes spam/ham for sms messages

## Steps
- data loading and preparing
- split data into training and validation data
- feature extraction
  CountVectorizer (skilearn) learns and count words as feature matrix
    (doc_id, word_id) -> count 
  alternative TF-IDF (skilearn)
    TF(t, d) = # term t in doc d / # total terms in doc d
    IDF(t, D) = log( (1 + # doc) / (1 + # doc containing t))
    TFIDF(t, d, D) = TF(t, d) * IDF(t, D)
- train the model linear neural network with single neuron
    BCE loss (binary loss entropy):
    L = y*log(sigmoid(yhat)) + (1-y)*log(1-sigmoid(yhat))  y:{0,1}
- apply sigmoid function to map the output to the range (0,1)
    p = sigmoid(x) = 1 / (1 + e^(-x))
- evaluate metrics depending on threshold after applying sigmoid
    accuracy, sensitivity, specificity, presision

