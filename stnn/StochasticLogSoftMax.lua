local StochasticLogSoftMax, parent = torch.class('nn.StochasticLogSoftMax', 'nn.StochasticModule')

function StochasticLogSoftMax:__init()
   parent:__init(self)
   self.lsm = nn.LogSoftMax()
end

function StochasticLogSoftMax:updateOutput(mu, var)
   assert(var ~= nil, 'two input arguments required')
   self.lsm:updateOutput(mu)
   self.mu = self.lsm.output
   self.var = var  -- no processing for var
   return self.mu, self.var
end
