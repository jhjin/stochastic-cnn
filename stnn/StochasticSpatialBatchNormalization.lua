local StochasticBN,parent = torch.class('nn.StochasticSpatialBatchNormalization', 'nn.StochasticModule')

function StochasticBN:__init(nFeature, eps, momentum, affine)
   parent.__init(self)
   assert(nFeature and type(nFeature) == 'number',
          'Missing argument #1: Number of feature planes. ')
   assert(nFeature ~= 0, 'To set affine=false call SpatialBatchNormalization'
     .. '(nFeature,  eps, momentum, false) ')
   if affine ~= nil then
      assert(type(affine) == 'boolean', 'affine has to be true/false')
      self.affine = affine
   else
      self.affine = true
   end
   self.eps = eps or 1e-5
   self.train = true

   self.running_mean = torch.zeros(nFeature)
   self.running_var = torch.ones(nFeature)
   if self.affine then
      self.weight = torch.Tensor(nFeature)
      self.bias = torch.Tensor(nFeature)
   end
end

function StochasticBN:updateOutput(mu, var)
   assert(var ~= nil, 'two input arguments required')
   assert(self.train == false, 'training mode is not allowed')
   self.mu:resizeAs(mu)
   self.var:resizeAs(var)

   mu.nn.StochasticSpatialBatchNormalization_updateOutput(
      mu,
      var,
      self.mu,
      self.var,
      self.weight,
      self.bias,
      self.eps,
      self.running_mean,
      self.running_var)

   return self.mu, self.var
end
