# Back Propagation
We are using ``Gradient Descent`` algorithm to update our weights.

## Model Architecture
Below is the neural network architecture which has one input layer, one hidden layer and one output layer. Every layer is a fully connected layer.
![Model Architecture](Model_Architecure.PNG)

## Gradient Descent
We will be using ``Gradient Descent`` algorithm to update our weights. Below is the equation used to update each weight.

<div align="center"><a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{120}&space;W_{i\_new}=W_{i}-\eta&space;*\frac{\partial&space;E_{total}}{\partial&space;W_{i}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;W_{i\_new}=W_{i}-\eta&space;*\frac{\partial&space;E_{total}}{\partial&space;W_{i}}" title="W_{i\_new}=W_{i}-\eta *\frac{\partial E_{total}}{\partial W_{i}}" /></a></div>

## Back Propagation using Gradient Descent
If we look at the architecture we can see that our total loss is

<div align="center"><a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{120}&space;E_{total}&space;=&space;E_1&space;&plus;&space;E_2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;E_{total}&space;=&space;E_1&space;&plus;&space;E_2" title="E_{total} = E_1 + E_2" /></a></div>

<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{120}&space;\begin{multline}&space;{\color{Black}&space;h_1=w_1*i_i&plus;w_2*i_2}&space;\\&space;h_2=w_3*i_1&plus;w_4*i_2&space;\end{multline}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;\begin{multline}&space;{\color{Black}&space;h_1=w_1*i_i&plus;w_2*i_2}&space;\\&space;h_2=w_3*i_1&plus;w_4*i_2&space;\end{multline}" title="\begin{multline} {\color{Black} h_1=w_1*i_i+w_2*i_2} \\ h_2=w_3*i_1+w_4*i_2 \end{multline}" /></a>