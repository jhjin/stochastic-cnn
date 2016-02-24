local StochasticThreshold, parent = torch.class('nn.StochasticThreshold','nn.StochasticModule')

function StochasticThreshold:__init(th,v,ip)
   parent.__init(self)
   self.threshold = th or 1e-6
   self.val = v or 0
   if (th and type(th) ~= 'number') or (v and type(v) ~= 'number') then
      error('nn.StochasticThreshold(threshold, value)')
   end
   -- default for inplace is false
   self.inplace = ip or false
   if (ip and type(ip) ~= 'boolean') then
      error('in-place flag must be boolean')
   end
   self:validateParameters()
end


function StochasticThreshold:updateOutput(mu, var)
   assert(var ~= nil, 'two input arguments required')
   self:validateParameters()
   return mu.nn.StochasticThreshold_updateOutput(self, mu, var)
end


function StochasticThreshold:validateParameters()
   self.inplace = self.inplace or false -- backwards compatibility pre inplace
   if self.inplace then
      if self.val > self.threshold then
         error('in-place processing requires value (' .. self.val ..
                  ') not exceed threshold (' .. self.threshold .. ')')
      end
   end
end
