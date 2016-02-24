local StochasticModule, parent = torch.class('nn.StochasticModule', 'nn.Module')

function StochasticModule:__init()
   parent.__init(self)

   self.mu = torch.Tensor()
   self.var = torch.Tensor()
end

function StochasticModule:updateOutput(mu, var)
   return self.mu, self.var
end

function StochasticModule:forward(mu, var)
   return self:updateOutput(mu, var)
end

function StochasticModule:backward()
   assert(false, 'not supported')
end

function StochasticModule:backwardUpdate()
   assert(false, 'not supported')
end

function StochasticModule:updateGradInput()
   assert(false, 'not supported')
end

function StochasticModule:accGradParameters()
   assert(false, 'not supported')
end

function StochasticModule:accUpdateGradParameters()
   assert(false, 'not supported')
end

function StochasticModule:sharedAccUpdateGradParameters()
   assert(false, 'not supported')
end

function StochasticModule:zeroGradParameters()
   assert(false, 'not supported')
end

function StochasticModule:updateParameters()
   assert(false, 'not supported')
end

function StochasticModule:clearState()
   assert(false, 'not supported')
end
