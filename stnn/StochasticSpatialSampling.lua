local StochasticSpatialSampling, parent = torch.class('nn.StochasticSpatialSampling','nn.StochasticModule')

function StochasticSpatialSampling:__init(r)
   parent.__init(self)
   self.radius = r or 1
   self.randi = torch.Tensor()
end

function StochasticSpatialSampling:updateOutput(mu, var)
   assert(var ~= nil, 'two input arguments required')
   local px = (2*self.radius+1)*(2*self.radius+1)
   self.randi:rand(mu:size()):mul(px-1e-9):floor()

   return mu.nn.StochasticSpatialSampling_updateOutput(self, mu, var)
end
