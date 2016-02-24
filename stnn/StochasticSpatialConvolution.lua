local StochasticSpatialConvolution, parent = torch.class('nn.StochasticSpatialConvolution', 'nn.StochasticModule')

function StochasticSpatialConvolution:__init(nInputPlane, nOutputPlane, kW, kH, dW, dH, padding)
   parent.__init(self)
   
   dW = dW or 1
   dH = dH or 1

   self.nInputPlane = nInputPlane
   self.nOutputPlane = nOutputPlane
   self.kW = kW
   self.kH = kH

   self.dW = dW
   self.dH = dH
   self.padding = padding or 0

   self.weight = torch.Tensor(nOutputPlane, nInputPlane*kH*kW)
   self.bias = torch.Tensor(nOutputPlane)

   self.finput = torch.Tensor()
   self.fgradInput = torch.Tensor()
   self.weight2 = torch.Tensor()
end

local function makeContiguous(self, input)
   if not input:isContiguous() then
      self._input = self._input or input.new()
      self._input:resizeAs(input):copy(input)
      input = self._input
   end
   return input
end

function StochasticSpatialConvolution:updateOutput(mu, var)
   assert(var ~= nil, 'two input arguments required')
   mu = makeContiguous(self, mu)
   var = makeContiguous(self, var)

   if not (self.weight2:nElement() == self.weight:nElement()) then
      self.weight2:resizeAs(self.weight):copy(self.weight):pow(2)
   end
   return mu.nn.StochasticSpatialConvolution_updateOutput(self, mu, var)
end

function StochasticSpatialConvolution:type(type)
   self.finput = torch.Tensor()
   return parent.type(self,type)
end

function StochasticSpatialConvolution:__tostring__()
   local s = string.format('%s(%d -> %d, %dx%d', torch.type(self),
   self.nInputPlane, self.nOutputPlane, self.kW, self.kH)
   if self.dW ~= 1 or self.dH ~= 1 or self.padW ~= 0 or self.padH ~= 0 then
      s = s .. string.format(', %d,%d', self.dW, self.dH)
   end
   if (self.padW or self.padH) and (self.padW ~= 0 or self.padH ~= 0) then
      s = s .. ', ' .. self.padW .. ',' .. self.padH
   end
   return s .. ')'
end
