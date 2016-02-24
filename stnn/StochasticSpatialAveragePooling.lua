local StochasticSpatialAveragePooling, parent = torch.class('nn.StochasticSpatialAveragePooling', 'nn.StochasticModule')

function StochasticSpatialAveragePooling:__init(kW, kH, dW, dH, padW, padH)
   parent.__init(self)

   self.kW = kW
   self.kH = kH
   self.dW = dW
   self.dH = dH
   self.padW = padW or 0
   self.padH = padH or 0
   self.ceil_mode = false
   self.count_include_pad = true
   self.divide = true
end

function StochasticSpatialAveragePooling:ceil()
   self.ceil_mode = true
   return self
end

function StochasticSpatialAveragePooling:floor()
   self.ceil_mode = false
   return self
end

function StochasticSpatialAveragePooling:setCountIncludePad()
   self.count_include_pad = true
   return self
end

function StochasticSpatialAveragePooling:setCountExcludePad()
   self.count_include_pad = false
   return self
end

function StochasticSpatialAveragePooling:updateOutput(mu, var)
   assert(var ~= nil, 'two input arguments required')
   return mu.nn.StochasticSpatialAveragePooling_updateOutput(self, mu, var)
end

function StochasticSpatialAveragePooling:__tostring__()
   local s = string.format('%s(%d,%d,%d,%d', torch.type(self),
   self.kW, self.kH, self.dW, self.dH)
   if (self.padW or self.padH) and (self.padW ~= 0 or self.padH ~= 0) then
      s = s .. ',' .. self.padW .. ','.. self.padH
   end
   s = s .. ')'
   return s 
end
