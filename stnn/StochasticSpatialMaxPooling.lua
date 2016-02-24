local StochasticSpatialMaxPooling, parent = torch.class('nn.StochasticSpatialMaxPooling', 'nn.StochasticModule')

function StochasticSpatialMaxPooling:__init(kW, kH, dW, dH, padW, padH, sort)
   parent.__init(self)

   dW = dW or kW
   dH = dH or kH
   
   self.kW = kW
   self.kH = kH
   self.dW = dW
   self.dH = dH

   self.padW = padW or 0
   self.padH = padH or 0

   self.ceil_mode = false
   self.sort = sort or true
end

function StochasticSpatialMaxPooling:ceil()
   self.ceil_mode = true
   return self
end

function StochasticSpatialMaxPooling:floor()
   self.ceil_mode = false
   return self
end

function StochasticSpatialMaxPooling:updateOutput(mu, var)
   assert(var ~= nil, 'two input arguments required')
   return mu.nn.StochasticSpatialMaxPooling_updateOutput(self, mu, var)
end

function StochasticSpatialMaxPooling:__tostring__()
   local s =  string.format('%s(%d,%d,%d,%d', torch.type(self),
   self.kW, self.kH, self.dW, self.dH)
   if (self.padW or self.padH) and (self.padW ~= 0 or self.padH ~= 0) then
      s = s .. ',' .. self.padW .. ','.. self.padH
   end
   s = s .. ')'

   return s
end
