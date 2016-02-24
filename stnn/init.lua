require('torch')
require('nn')
require('libstnn')

include('StochasticModule.lua')
include('StochasticContainer.lua')
include('StochasticSequential.lua')

include('StochasticSpatialConvolution.lua')
include('StochasticSpatialBatchNormalization.lua')
include('StochasticSpatialMaxPooling.lua')
include('StochasticSpatialAveragePooling.lua')
include('StochasticThreshold.lua')
include('StochasticReLU.lua')
include('StochasticLinear.lua')
include('StochasticDropout.lua')
include('StochasticView.lua')
include('StochasticSpatialSoftMax.lua')
include('StochasticSoftMax.lua')
include('StochasticLogSoftMax.lua')
include('StochasticSpatialSampling.lua')
include('StochasticIdentity.lua')
include('StochasticCAddTable.lua')
include('StochasticConcatTable.lua')


local function switchModule(m)
   local name = m.__typename

   if (name == 'nn.Sequential') then
      return nn.StochasticSequential()
      
   elseif (name == 'nn.SpatialConvolution') or
      (name == 'nn.SpatialConvolutionMM') or
      (name == 'cudnn.SpatialConvolution') then
      local m_new = nn.StochasticSpatialConvolution(m.nInputPlane, m.nOutputPlane, m.kW, m.kH, m.dW, m.dH, m.padW, m.padH)
      m_new.weight:copy(m.weight)
      m_new.bias:copy(m.bias)
      return m_new

   elseif (name == 'nn.SpatialBatchNormalization') then
      local m_new = nn.StochasticSpatialBatchNormalization(m.running_mean:nElement(), m.eps, m.momentum, m.affine)
      m_new.running_mean:copy(m.running_mean)
      m_new.running_var:copy(m.running_var)
      if m.affine then
         m_new.weight:copy(m.weight)
         m_new.bias:copy(m.bias)
      end
      return m_new

   elseif (name == 'nn.SpatialMaxPooling') or
          (name == 'cudnn.SpatialMaxPooling') then
      local sort = true
      local m_new = nn.StochasticSpatialMaxPooling(m.kW, m.kH, m.dW, m.dH, m.padW, m.padH, sort)
      if m.ceil_mode then
         return m_new:ceil()
      else
         return m_new:floor()
      end

   elseif (name == 'nn.SpatialAveragePooling') or
          (name == 'cudnn.SpatialAveragePooling') then
      local m_new = nn.StochasticSpatialAveragePooling(m.kW, m.kH, m.dW, m.dH, m.padW, m.padH)
      if m.ceil_mode then
         m_new:ceil()
      else
         m_new:floor()
      end
      if m.count_include_pad then
         return m_new:setCountIncludePad()
      else
         return m_new:setCountExcludePad()
      end

   elseif (name == 'nn.Threshold') then
      return nn.StochasticThreshold(m.threshold, m.val, m.inplace)

   elseif (name == 'nn.ReLU') or
          (name == 'cudnn.ReLU') then
      return nn.StochasticReLU(m.inplace)

   elseif (name == 'nn.Linear') then
      local m_new = nn.StochasticLinear(m.weight:size(2), m.weight:size(1))
      m_new.weight:copy(m.weight)
      m_new.bias:copy(m.bias)
      return m_new

   elseif (name == 'nn.BatchNormalization') then
      assert(false, name .. 'module not implemented yet')
      local m_new = nn.StochaasticBatchNormalization(m.running_mean:nElement(), m.eps, m.momentum, m.affine)
      m_new.running_mean:copy(m.running_mean)
      m_new.running_std:copy(m.running_std)
      if m.affine then
         m_new.weight:copy(m.weight)
         m_new.bias:copy(m.bias)
      end
      return m_new

   elseif (name == 'nn.Dropout') then
      return nn.StochasticDropout(m.p)

   elseif (name == 'nn.View') then
      return nn.StochasticView(m.size)

   elseif (name == 'nn.SpatialSoftMax') or
          (name == 'cudnn.SpatialSoftMax') then
      return nn.StochasticSpatialSoftMax()

   elseif (name == 'nn.SoftMax') or
          (name == 'cudnn.SoftMax') then
      return nn.StochasticSoftMax()

   elseif (name == 'nn.LogSoftMax') or
          (name == 'cudnn.LogSoftMax') then
      return nn.StochasticLogSoftMax()

   elseif (name == 'nn.Identity') then
      return nn.StochasticIdentity()

   elseif (name == 'nn.CAddTable') then
      return nn.StochasticCAddTable(m.inplace)

   elseif (name == 'nn.ConcatTable') then
      return nn.StochasticConcatTable()

   else
      assert(false, name .. ' module not supported')
   end
end

local function traverseSubmodules(x)
   -- if x is a module
   if (x.modules == nil) then
      return switchModule(x)

   -- if model is a container
   else
      local name = x.__typename
      if (name == 'nn.Sequential') or
         (name == 'nn.ConcatTable') then
         local x_new = switchModule(x)

         for i = 1, #x.modules do
            x_new:add(traverseSubmodules(x.modules[i]))
         end
         return x_new
      else
         assert(false, name .. ' container not supported')
      end
   end
end

function nn.toStochasticModel(x)
   local x_new = traverseSubmodules(x)
   x_new:evaluate()
   return x_new
end

return nn
