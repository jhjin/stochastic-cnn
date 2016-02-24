local StochasticSequential, _ = torch.class('nn.StochasticSequential', 'nn.StochasticContainer')

function StochasticSequential:__len()
   return #self.modules
end

function StochasticSequential:add(module)
   table.insert(self.modules, module)
   self.mu = module.mu
   self.var = module.var
   return self
end

function StochasticSequential:insert(module, index)
   index = index or (#self.modules + 1)
   if index > (#self.modules + 1) or index < 1 then
      error"index should be contiguous to existing modules"
   end
   table.insert(self.modules, index, module)
   self.mu = self.modules[#self.modules].mu
   self.var = self.modules[#self.modules].var
end

function StochasticSequential:remove(index)
   index = index or #self.modules
   if index > #self.modules or index < 1 then
      error"index out of range"
   end
   table.remove(self.modules, index)
   if #self.modules > 0 then
       self.mu = self.modules[#self.modules].mu
       self.var = self.modules[#self.modules].var
   else
       self.mu = torch.Tensor()
       self.var = torch.Tensor()
   end
end

function StochasticSequential:updateOutput(mu, var)
   assert(var ~= nil, 'two input arguments required')
   local currentMu = mu
   local currentVar = var
   for i=1,#self.modules do
      currentMu, currentVar = self.modules[i]:updateOutput(currentMu, currentVar)
   end
   self.mu = currentMu
   self.var = currentVar
   return currentMu, currentVar
end

function StochasticSequential:__tostring__()
   local tab = '  '
   local line = '\n'
   local next = ' -> '
   local str = 'nn.StochasticSequential'
   str = str .. ' {' .. line .. tab .. '[input'
   for i=1,#self.modules do
      str = str .. next .. '(' .. i .. ')'
   end
   str = str .. next .. 'output]'
   for i=1,#self.modules do
      str = str .. line .. tab .. '(' .. i .. '): ' .. tostring(self.modules[i]):gsub(line, line .. tab)
   end
   str = str .. line .. '}'
   return str
end
