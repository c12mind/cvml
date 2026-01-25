# Deep learning analysis

Machine learning models can be thought of as universal function approximators (UFAs). That is, provided enough data to learn from, they are able to model any continuous, closed domain function to an arbitrary degree of accuracy[https://www.preprints.org/manuscript/202502.0272, https://link.springer.com/article/10.1007/s00365-025-09713-8]. Such models consist of interconnected "neurones" arranged in layers. As shown in Figure xxx,  these connections are dense, meaning each neurone is connected to all other neurones in the previous and next layer. 
[Figure xxx]


These dense connections gave rise to the common "fully connected" layer terminology. We will refer to a single layer as F_i^{n,m} where i is the position of the layer, n is the input size (the number of input neurones expected), and m is the output size. The layer computes:

y = F_i^{n, m}(x) = W_ix + b_i

where x is an n-dimensional vector, y is an m-dimensional vector, W_i is an n x m weight matrix and b_i is a constant term that is added to all outputs.  Layers may be chained sequentially, as long as the inner dimensions of adjascent layers are equal, i.e. W_i and W_{i+1} can be multiplied. Such a sequence of connected layers is commonly referred to as a Multilayer Perceptron (MLP). An important element of the layers of the MLP is the activation function, which is applied to the output of each neurone, and introduces non-linearity into the model. This is essential for an MLP to fit the definition of a universal function approximator as defined earlier, because any sequence of fully connected layers must necessarily collapse into a single linear transform, which is not powerful enough to represent more complicated functions. The Rectified Linear Unit function (ReLU) is a popular activation function designed for this purpose and is defined as follows:

ReLU(x) = {x if x > 0; 0 elsewhere}

Thus, the MLP must have a ReLU activation function between each hidden layer, giving the final model structure as laid out in Figure xxx. 

[Figure xxx]

The final output of the model is evaluated against the target value (also called the ground truth value) using a cost function L -- see section x.x for how this is defined for this application. The derivative of L is calculated with respect to each of the model's parameters:

\partL /  \partW = \part L / \partyhat * \partyhat / \partW

\partL / \partW tells the model how much, and the direction in which W should change in order to minimise L. Repeating these forward + backwards passes over the MLP over and over again brings the weights closer to their "ideal" values -- ones for which L is minimised. In other words, the optimsing algorithm uses the difference between the output of the model and its target to update W in the correct direction. This is what is meant by training a machine learning model.  

## Previous work
The application of machine learning systems to the prediction of specific capacitance values is varied -- some predict specific capacitance directly from experimental conditions [https://pubs.rsc.org/en/content/articlehtml/2025/ya/d4ya00577e], while others use these experimental conditions to predict hysteresis curves [https://pubs.acs.org/doi/10.1021/acsomega.3c10485]; the latter approach allows the indirect calculation of specific capacitance C_{sp}, which is proportional to the area between charge and discharge curves. It is this latter approach that is used by the present study, because predicting the full charge-discharge dynamics of cyclic voltammetry allows the model to generalise better to a wider range of conditions [https://www.nature.com/articles/s41467-024-45394-w] -- especially important given the limited data available for this study. Using this approach, C_{sp} is given by:

C_{sp} = \integral I_pred(E)dE / 2 * \nu * m * ΔV

where I_pred is the function that is approximated by the MLP, \nu is the scanrate, m is the total mass of the active material in the electrodes, and \delta V is the change in voltage over the whole charge/discharge cycle.

## Dataset
* the number of experiments used, and their properties: do this as a table. What are the inputs? what are the outputs? describe how the model is fed these features and how they are normalised/denormalised. Diagram?
The complete dataset is made up of current density samples (ask Alex if this is the correct term) from 7 cyclic voltammetry experiments -- see Table xxx. These were collected via potentiostat with a scanrate of 100mV^-s. Each experiment is made up of 264 rows, with columns for current (I / Amps), voltage step (E / Volts), electrode mass (M / mg), temperature (T / degC), electrolysis voltage (V_e / Volts), electrolysis duration (D / hours), and scan direction (S). The last feature tells the model if the particular point is on the charge or discharge curve and is represented by either +1 (charging)  or -1 (discharging).  

|Experiment | M / mg | T / degC | V_e / V | D / h |electrolye |
|-----------|--------|----------|---------|-------|-----------|
|BCMS2      | 1.9    |    800   |  2      |  2    |  Eth      |
|BCMS2.1    | 0.7    |    650   |  2      |  3    |  Eth      |
|BCMS3      | 2.2    |    750   |  2      |  2    |  Eth      |
|BCMS6      | 0.28   |    700   |  2.5    |  2    |  Eth      |
|BCMS7      | 1.7    |    750   |  2      |  2    |  Eth      |
|BCMS8      | 0.8    |    700   |  2.5    |  2    | KOH + Eth |
|BCMS9      | 5.7    |    700   |  2.8    |  2    |  Eth      |
<!-- 
\begin{table}
\centering
\begin{tblr}{
  hline{1,9} = {-}{0.08em},
  hline{2} = {-}{},
}
Experiment & M / mg & T / degC & V\_e / V & D / h & electrolye \\
BCMS2      & 1.9    & 800      & 2        & 2     & Eth        \\
BCMS2.1    & 0.7    & 650      & 2        & 3     & Eth        \\
BCMS3      & 2.2    & 750      & 2        & 2     & Eth        \\
BCMS6      & 0.28   & 700      & 2.5      & 2     & Eth        \\
BCMS7      & 1.7    & 750      & 2        & 2     & Eth        \\
BCMS8      & 0.8    & 700      & 2.5      & 2     & KOH+Eth    \\
BCMS9      & 5.7    & 700      & 2.8      & 2     & Eth        
\end{tblr}
\end{table}
 -->

Notice that the various inputs span many orders of magnitude -- this is a problem for machine learning models, which work best when their inputs are all near or around -1 to 1 (rephrase this). Using raw features such as those presented in Table xxx mean that ones with large absolute values dominate the weights of the model, even though they may not be as important, and vice versa, which can lead to lower performance [https://www.sciencedirect.com/science/article/pii/S2210670724003962]. For this reason, all features are feature scaled across all experiments such that they have zero mean and 1 s.d.:

scaled_feature = (feature - mean(feature)) / sd(feature)


## Experimental design and model architecture
* experimental design / leave one out stuff
We employ leave-one-out cross validation when training: for any one run, the model is trained only on 6 of the experiments and one is held out for validation. Here, validation means only evaluating the forward pass of the model and not updating the weights of the model. It is a test to determine if the model is able to generalise to "unseen" data, and each experiment is subjected to this. We also repeat each experiment 5 times [NOTE, I DID NOT ACTUALLY DO THIS YET, MUST DO IT BEFORE SUBMITTING FOR REVIEW] to build confidence that the model's output is reliable.

Our model is a multilayered perceptron with 4 hidden layers in between a 6-neuron input layer and a 1-neuron output layer. The model architecture is given in detail in Figure xxx.
[Figure xxx]

As mentioned in the Section xxx, the objective function scores the output of the MLP so it can be used in gradient calculations, which in turn are used to inform updates to the model's weights. Because the model simply predicts the total 
data_loss = mse = 1/n \sum_{i=1}^{n} (yhat_i - y_i)**2
smooth_loss = \Bigl\lVert\frac{\partial^2 I_{pred}}{\partial E^2}\Bigr\rVert_{2}^{2}
area_loss = mse = 1/n \sum_{i=1}^{n} (ahat_i - a_i)**2

data_loss + lambda_smooth * smooth loss + lambda_area * area_loss

## Results and discussion

|test_cycle |supycap / Fg^-1 | machine learning / Fg^-1 | excel / Fg^-1 | ml_err %     |
|-----------|----------------|--------------------------|---------------|--------------|
|BCMS2      | 4.118          |  4.156                   |   3.105       | 0.9227780476 |
|BCMS2.1    |15.336          |  15.694                  |  10.76        | 2.33437663   |
|BCMS3      |4.279           |   4.558                  |   4.23        | 6.520215004  |
|BCMS6      |170.262         |   154.215                |  130.31       | 9.424886352  |
|BCMS7      |6.19            |  6.381                   |   4.64        | 3.085621971  |
|BCMS8      |50.934          |   52.593                 |   39.64       | 3.25715632   |
|BCMS9      |1.476           |   1.345                  |    1.244      | 8.875338753  |

<!-- 
\begin{table}
\centering
\begin{tblr}{
  hline{1,10} = {-}{0.08em},
  hline{2} = {-}{},
}
Experiment      & MLP\_\{baseline\} & MLP\_\{baseline\} + drop & MLP\_\{baseline\}+drop+smooth & MLP\_\{baseline\}+drop+smooth+area \\
BCMS2Eth        & 11.491            & 6.304                    & 4.745                         & 0.733                              \\
BCMS2Eth\_again & 28.385            & 25.788                   & 17.547                        & 62.284                             \\
BCMS3Eth        & 0.552             & 10.432                   & 2.449                         & 19.004                             \\
BCMS6Eth        & 6.482             & 7.548                    & 8.757                         & 8.310                              \\
BCMS7Eth        & 8.795             & 2.582                    & 10.220                        & 9.344                              \\
BCMS8\_KOH\_Eth & 4.860             & 4.703                    & 3.490                         & 5.534                              \\
BCMS9Eth        & 48.238            & 24.038                   & 3.645                         & 24.173                             \\
Avg. error / \% & 15.543            & 11.628                   & 7.265                         & 18.483                             
\end{tblr}
\end{table} 
-->

* discussion - why larger errors in some but not others - use actual hysteresis area for this, not specific capacitance. speculate on how to fix it
* Limitations - Although each of the 7 experiments has 264 rows of data, each row has the same feature vector that is input into the model, the only thing that is different is the voltage step. But the voltage step/progression is common to all the experiments. so really, there are only 14 unique feature vectors in the dataset of ~2000  samples (counting forward and backward curves). I'm actually surprised this works at all. Very low variance in the features 