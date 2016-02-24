local StochasticSpatialSoftMax, parent = torch.class('nn.StochasticSpatialSoftMax', 'nn.StochasticModule')

function StochasticSpatialSoftMax:__init()
   parent:__init(self)
   self.ssm = nn.SpatialSoftMax()
end

function StochasticSpatialSoftMax:updateOutput(mu, var)
   assert(var ~= nil, 'two input arguments required')
   self.ssm:updateOutput(mu)
   self.mu = self.ssm.output
   self.var = var  -- no processing for var
   return self.mu, self.var
end
