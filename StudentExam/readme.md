# Student Exam Performance Prediction

Dataset: [Kaggle](https://www.kaggle.com/datasets/mrsimple07/student-exam-performance-prediction)
License: [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)

## Columns
- Study Hours
- Previous Exam Score
- Pass/Fail 

## Goal
Create neural network classifie to predict if student fails or passes the exam

## Steps
- load data
- create a model
  BCE loss binary classifier
  optimizer Adam
  architecture: Linear(2, 10) -> Relu/Sigmoid -> Linear(10, 1)
- train the model with mini batches
- evaluate 