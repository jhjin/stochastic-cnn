local StochasticIdentity, _ = torch.class('nn.StochasticIdentity', 'nn.StochasticModule')

function StochasticIdentity:updateOutput(mu, var)
   self.mu = mu
   self.var = var
   return self.mu, self.var
end
