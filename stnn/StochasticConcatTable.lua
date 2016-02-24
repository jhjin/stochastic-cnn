local StochasticConcatTable, parent = torch.class('nn.StochasticConcatTable', 'nn.StochasticContainer')

function StochasticConcatTable:__init()
   parent.__init(self)
   self.modules = {}
   self.mu = {}
   self.var = {}
end

function StochasticConcatTable:updateOutput(mu, var)
   for i=1,#self.modules do
      self.mu[i], self.var[i] = self.modules[i]:updateOutput(mu, var)
   end
   return self.mu, self.var
end

function StochasticConcatTable:__tostring__()
   local tab = '  '
   local line = '\n'
   local next = '  |`-> '
   local ext = '  |    '
   local extlast = '       '
   local last = '   ... -> '
   local str = torch.type(self)
   str = str .. ' {' .. line .. tab .. 'input'
   for i=1,#self.modules do
      if i == self.modules then
         str = str .. line .. tab .. next .. '(' .. i .. '): ' .. tostring(self.modules[i]):gsub(line, line .. tab .. extlast)
      else
         str = str .. line .. tab .. next .. '(' .. i .. '): ' .. tostring(self.modules[i]):gsub(line, line .. tab .. ext)
      end
   end
   str = str .. line .. tab .. last .. 'output'
   str = str .. line .. '}'
   return str
end
