local StochasticView, parent = torch.class('nn.StochasticView', 'nn.StochasticModule')

function StochasticView:__init(...)
   parent.__init(self)
   if select('#', ...) == 1 and torch.typename(select(1, ...)) == 'torch.LongStorage' then
      self.size = select(1, ...)
   else
      self.size = torch.LongStorage({...})
   end

   self.numElements = 1
   local inferdim = false
   for i = 1,#self.size do
      local szi = self.size[i]
      if szi >= 0 then
         self.numElements = self.numElements * self.size[i]
      else
         assert(szi == -1, 'size should be positive or -1')
         assert(not inferdim, 'only one dimension can be at -1')
         inferdim = true
      end
   end

   self.output = nil
   self.gradInput = nil
   self.numInputDims = nil
   self.mu = nil
   self.var = nil
end

function StochasticView:setNumInputDims(numInputDims)
   self.numInputDims = numInputDims
   return self
end

local function batchsize(input, size, numInputDims, numElements)
   local ind = input:nDimension()
   local isz = input:size()
   local maxdim = numInputDims and numInputDims or ind
   local ine = 1
   for i=ind,ind-maxdim+1,-1 do
      ine = ine * isz[i]
   end

   if ine % numElements ~= 0 then
      error(string.format(
               'input view (%s) and desired view (%s) do not match',
               table.concat(input:size():totable(), 'x'),
               table.concat(size:totable(), 'x')))
   end

   -- the remainder is either the batch...
   local bsz = ine / numElements

   -- ... or the missing size dim
   for i=1,size:size() do
      if size[i] == -1 then
         bsz = 1
         break
      end
   end

   -- for dim over maxdim, it is definitively the batch
   for i=ind-maxdim,1,-1 do
      bsz = bsz * isz[i]
   end

   -- special card
   if bsz == 1 and (not numInputDims or input:nDimension() <= numInputDims) then
      return
   end

   return bsz
end

function StochasticView:updateOutput(mu, var)
   assert(var ~= nil, 'two input arguments required')
   local bsz = batchsize(mu, self.size, self.numInputDims, self.numElements)
   if bsz then
      self.mu = mu:view(bsz, unpack(self.size:totable()))
      self.var = var:view(bsz, unpack(self.size:totable()))
   else
      self.mu = mu:view(self.size)
      self.var = var:view(self.size)
   end
   return self.mu, self.var
end

function StochasticView:__tostring__()
   return torch.type(self)..'('..table.concat(self.size:totable(),',')..')'
end
