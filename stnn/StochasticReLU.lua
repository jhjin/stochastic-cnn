local StochasticReLU, Parent = torch.class('nn.StochasticReLU', 'nn.StochasticThreshold')

function StochasticReLU:__init(p)
   Parent.__init(self,0,0,p)
end
