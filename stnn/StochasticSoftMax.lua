local StochasticSoftMax, parent = torch.class('nn.StochasticSoftMax', 'nn.StochasticModule')

function StochasticSoftMax:__init()
   parent:__init(self)
   self.sm = nn.SoftMax()
end

function StochasticSoftMax:updateOutput(mu, var)
   assert(var ~= nil, 'two input arguments required')
   self.sm:updateOutput(mu)
   self.mu = self.sm.output
   self.var = var  -- no processing for var
   return self.mu, self.var
end
