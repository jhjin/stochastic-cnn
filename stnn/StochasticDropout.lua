local StochasticDropout, Parent = torch.class('nn.StochasticDropout', 'nn.StochasticModule')

function StochasticDropout:__init(p,v1)
   Parent.__init(self)
   self.p = p or 0.5
   self.train = true
   -- version 2 scales output during training instead of evaluation
   self.v2 = not v1
   if self.p >= 1 or self.p < 0 then
      error('<StochasticDropout> illegal percentage, must be 0 <= p < 1')
   end
end

function StochasticDropout:updateOutput(mu, var)
   assert(var ~= nil, 'two input arguments required')
   self.mu:resizeAs(mu):copy(mu)
   self.var:resizeAs(var):copy(var)

   if self.train then
      assert(false, 'not considered in training mode')
   elseif not self.v2 then
      self.mu:mul(1-self.p)
      self.var:mul((1-self.p)^2)
   end

   return self.mu, self.var
end

function StochasticDropout:__tostring__()
   return string.format('%s(%f)', torch.type(self), self.p)
end
