# Handwriting Recognition

Handwriting Recognition is a program that uses CNN (Convolutional Neural Network) to categorise handwritten letters.

## Getting Started

To get started with the program you need to clone or download zip file to your machine (unzip it), into the preferable directory.
For acquiring dataset used in this program go to url: https://mega.nz/file/bzoBiR5K#G_emMg6PIcJxMeBibkhnz3saZVH4Tm2QTLrK1Ks6jN0 download zip file and extract it to directory you previously cloned. 

### Prerequisites

  *Python 3.7 (https://www.python.org/downloads/release/python-370/)
  
  *Theano library
 ```
 pip install theano
 ```
 
  *Tensorflow library
 ```
 pip install tensorflow
 ```
 
  *Keras library
 ```
 pip install keras
 ```

  *Gzip library
 ```
 pip install gzip
 ```

  *Matplotlib library
 ```
 pip install matplotlib
 ```

  IDE of your choice
  
  *Alternative is using Anaconda

## Deployment

Directory Handwrithing-Recognition consist of 6 scripts:

*constants.py - parameters for building model and managing data loading(size of test and traing set, width and height of images)

*data_loading.py - reading data from files and converting them to format suitable for analysis

*model.py - contains model of neural network used for learning

*accuracy.py - AccuraciHistory class for tracking statistics

*handwriting_recognition_main.py - executes data loading and train model on that data (MAIN EXECUTION SCRIPT)

*analytics_support.py - if you previously executed _main script you do not need to execute first 3 imports in _support script. Execution gives visual and textual representation of results (SUPPORT EXECUTION SCRIPT)

To load data and train model execute script with suffix _main, due to amount of data and the type of tensorflow backend (CPU or GPU) this may take a while. When model is trained execute script with suffix _support to get evaluation of the model and statistic for training.

If kept with default parameters accuracy should be 86% percents over 20 (default) epochs. 

>DUE TO LARGE AMOUNT OF DATA PROCESS OF TRAINING MAY BE HARD FOR MACHINE (SUGESTION IS TO USE GPU FOR COMPUTING WITH TENSORFLOW-GPU VERSION IF SUPPORTED)!

## Built With

* [Python](http://www.dropwizard.io/1.0.2/docs/) - Programming language
* [EMNIST Dataset](https://www.nist.gov/itl/products-and-services/emnist-dataset) - Dataset used for training


## Authors

* **Vladimir Novakovic** - *Initial work* - [CupakabraNo1](https://github.com/CupakabraNo1)

## License

This project is not licensed.

## Acknowledgments

* University related project.

