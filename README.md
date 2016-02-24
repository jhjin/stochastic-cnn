## Convolutional neural networks with stochastic input

Despite of the success of deep networks, they could be easily fooled by few pixels of noise so as to output incorrect answers.
Our feedforward model utilizes uncertainty information and achieves high robustness against strong noise on a large-scale dataset.
This package contains implementations of stochastic feedforward operators that were mostly modified and derived from `nn` and `cunn` packages.

The video summarizes our work http://youtube.com/watch?v=9cP06jFpxt0 .
More details are in the paper http://arxiv.org/abs/1511.06306 .


### Install

Choose both or either of `nn`/`cunn` backend packages depending on your computing environment.

```bash
luarocks install https://raw.githubusercontent.com/jhjin/stochastic-cnn/master/stnn-scm-1.rockspec    # cpu
luarocks install https://raw.githubusercontent.com/jhjin/stochastic-cnn/master/stcunn-scm-1.rockspec  # cuda
```


### Available modules

This is a list of available modules.

```lua
nn.StochasticCAddTable()
nn.StochasticConcatTable()
nn.StochasticDropout()
nn.StochasticIdentity()
nn.StochasticLinear()
nn.StochasticLogSoftMax()
nn.StochasticReLU()
nn.StochasticSoftMax()
nn.StochasticSpatialAveragePooling()
nn.StochasticSpatialBatchNormalization()
nn.StochasticSpatialConvolution()
nn.StochasticSpatialConvolutionMM()
nn.StochasticSpatialMaxPooling()
nn.StochasticSpatialSampling()
nn.StochasticSpatialSoftMax()
nn.StochasticThreshold()
nn.StochasticView()
```


### Example

Refer to the following code or check the `demo` directory.

```lua
require('stnn')

-- set dummy input and input variance
local x = torch.randn(1,1,4,4)
local x_var = x:clone():fill(0.1)

-- standard feedforward
local model = nn.Sequential()
model:add(nn.SpatialConvolution(1,8,3,3))
model:add(nn.SpatialMaxPooling(2,2,2,2))
local y_standard = model:forward(x)

-- stochastic feedforward
local model_st = nn.toStochasticModel(model)
local y_stochastic  = model_st:forward(x, x_var)

-- compare results
print(y_standard:view(-1))
print(y_stochastic:view(-1))
```
