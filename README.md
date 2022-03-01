# Working-Memory-Learning-Curves

The project compares the improvement trajectory of a trained WM task with three transfer task in order to seperate tasl specific improvements (refered to as strategy) from transferable improvements (refered to as capacity). It consists of two parts:

### 1. Confirmatory Factor Analysis (CFA)
A common latent factor is defined using the transfer tasks (which are tested on six different occations) and strong measurement invariance. The trained task is then added to the model, first without fixing the trained task intercept (only week invariance). However, the intercepts could be fixated from the second intercept and onwards, suggesting that strategy development happens in the beginning of the training period while capacity improvement is a slower process which happens throughout the training period. 

The CFA was coded in R. 

### 2. Piecewise Linear Learning Curve fitted using Hidden Markov Models (HMM)
To give a more detailed description of when the strategy improvement occur, we fit a piecewise linear (PL) model, inspired by the results from the CFA, to individual training data. The PL model has two parts: In (1) the slope consists of a strategy coefficient and a capacity coefficent and in (2) the slope consists of only a capacity coefficent representing the period of learning where the strategy improvement has ended. The model suggested that strategy improvement ends after 2 days of training on average. 

We also fitted that data to a two stage model consisting of a tanh and a linear function. The tanh part represents strategy and converges after a few days, while the linear part respresents the constant capacity improvement. This model was very sensitive to noise (which was tested using stimulated training data) and thus the timing of the transition from strategy to no strategy improvement was not reliable. However, the model fitted the data well and showed similar results to the PL model further validating it.

The learning curves were coded in Python using Jupyter Notebooks.
