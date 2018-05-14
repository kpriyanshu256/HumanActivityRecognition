# HumanActivityRecognition

## Objective
To classify common human activities like walking,standing,laying on the basis of readings obtained from smartphone sensors

## Dataset
 Source: UCI ML Repository</br>
 Human Activity Recognition Using Smartphones Data Set</br>
 https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones#

## Model

* Dataset has 561 attributes so Principal Component Analysis(PCA) is used to reduce the dimension.
* Best results are obtained by taking about 200 principal components. 
* Linear SVM("one vs one") was used to classify the data

## About Repository

* ActivityRecognition.py --- script to pickle data
* ActivityRecognition2.py --- classification script
* HAR pca.png -- image showing 2 principal components of the data
* TDT.png -- shows a plot of training, development and testing accuracies over number of principal components

## Results

* Training accuracy ~ 99%
* Development or cross-validation accuracy ~ 98%
* Testing accuracy ~ 95-96%
* Most mis-classifications were obtained for standing and sitting classes as there is not quite of a difference between the 2 postures.

## Future Scope

* Neural networks can be tried for the dataset

