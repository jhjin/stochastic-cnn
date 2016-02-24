local StochasticLinear, parent = torch.class('nn.StochasticLinear', 'nn.StochasticModule')

function StochasticLinear:__init(inputSize, outputSize)
   parent.__init(self)

   self.weight = torch.Tensor(outputSize, inputSize)
   self.bias = torch.Tensor(outputSize)

   self.weight2 = torch.Tensor()
end

function StochasticLinear:updateOutput(mu, var)
   assert(var ~= nil, 'two input arguments required')
   if not (self.weight2:nElement() == self.weight:nElement()) then
      self.weight2:resizeAs(self.weight):copy(self.weight):pow(2)
   end

   if mu:dim() == 1 then
      self.mu:resize(self.bias:size(1))
      self.mu:copy(self.bias)
      self.mu:addmv(1, self.weight, mu)

      self.var:resize(self.bias:size(1))
      self.var:zero()
      self.var:addmv(1, self.weight2, var)

   elseif mu:dim() == 2 then
      local nframe = mu:size(1)
      local nunit = self.bias:size(1)
      self.mu:resize(nframe, nunit)
      self.var:resize(nframe, nunit):zero()
      if not self.addBuffer or self.addBuffer:nElement() ~= nframe then
         self.addBuffer = mu.new(nframe):fill(1)
      end
      if nunit == 1 then
         -- Special case to fix output size of 1 bug:
         self.mu:copy(self.bias:view(1,nunit):expand(#self.mu))
         self.mu:select(2,1):addmv(1, mu, self.weight:select(1,1))
         self.var:select(2,1):addmv(1, var, self.weight2:select(1,1))
      else
         self.mu:zero():addr(1, self.addBuffer, self.bias)
         self.mu:addmm(1, mu, self.weight:t())
         self.var:addmm(1, var, self.weight2:t())
      end
   else
      error('input must be vector or matrix')
   end

   return self.mu, self.var
end

function StochasticLinear:__tostring__()
  return torch.type(self) ..
      string.format('(%d -> %d)', self.weight:size(2), self.weight:size(1))
end
