local StochasticCAddTable, parent = torch.class('nn.StochasticCAddTable', 'nn.StochasticModule')

function StochasticCAddTable:__init(ip)
   parent.__init(self)
   self.inplace = ip
end

function StochasticCAddTable:updateOutput(mu, var)
   if self.inplace then
      self.mu:set(mu[1])
      self.var:set(var[1])
   else
      self.mu:resizeAs(mu[1]):copy(mu[1])
      self.var:resizeAs(var[1]):copy(var[1])
   end
   for i=2,#mu do
      self.mu:add(mu[i])
      self.var:add(var[i])
   end
   return self.mu, self.var
end
