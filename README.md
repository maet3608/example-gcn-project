# README

Example for a minimal Graph Convolutional Network project


## Install

Clone repo and install package:

```
git clone git@github.ibm.com:IBM-Research-AI/example-gcn-project.git
cd example-gcn-project
python setup.py develop
```

Create training data:

```
cd example-gcn-project/exgcn
python snowflakes.py
```


## Example run

```
cd example-gcn-project/exgcn
python train.py
```

Should produce output similar to this

```
creating network ...
loading samples ...
#samples [6, 24, 30]
training on  cuda ...
0..50  00:00 : 1.5745  (62.5% 0.0%)
1..50  00:00 : 0.8533  (50.0% 58.3%)
2..50  00:00 : 0.9490  (12.5% 58.3%)
3..50  00:00 : 0.6886  (62.5% 58.3%)
...
47..50  00:00 : 0.4005  (100.0% 100.0%)
48..50  00:00 : 0.4034  (87.5% 100.0%)
49..50  00:00 : 0.3989  (75.0% 100.0%)
```

The validation accuracy should be 100% (or very close to it.).
To compute the test accuracy run:

```
cd example-gcn-project/exgcn
python evaluate.py
```

And the output should be similar to this with a test accuracy of 100%.

```
creating network ...
loading weights ...
loading samples ...
evaluating ...
targets     [1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1]
predictions [1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1]
test accuracy: 100.0
```

## Hints

- training runs for 50 epochs, stores the weights for the best validation
  accuracy and plots loss, training and evaluation accuracy.
- `constants.py` contains paraemters to control learning rate, number of epochs,
   split ratios for the dataset, filepathes and others.
  