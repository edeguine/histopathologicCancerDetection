# Histopathologic cancer detection 

## The challenge
This code is my entry for the Kaggle competition here: https://www.kaggle.com/c/histopathologic-cancer-detection

## The code
### Code Features

The code has:
 - splitDataset.py: some code for organizing the dataset
 - loadData.py: loading the data to PyTorch
 - myCNN: a model definition
 - pipeline.py: the training code
 - generateTest.py: generating predictions to submit to Kaggle


### Usage

First download the data

Split the dataset into trainset and testset:
```
python splitDataset.py train_labels.csv
```

Train the model
```
python pipeline.py
```

Generate predictions on the test set
```
python generateTest.py
```

### License

The software is free but copyrighted. It is copyrighted under the [JRL license](https://en.wikipedia.org/wiki/Java_Research_License), commercial or proprietary use is forbidden but research and academic use are allowed.


