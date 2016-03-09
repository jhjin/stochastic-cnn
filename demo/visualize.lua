require('torch')
require('qtwidget')
require('paths')
require('image')
require('nn')
require('cunn')
require('cudnn')
require('stnn')
require('stcunn')
local THNN = require('nn.THNN')
local DataLoader = require('dataloader')
torch.setdefaulttensortype('torch.FloatTensor')


local function backward(self, input, gradOutput, scale, gradInput, gradWeight, gradBias)
   self:checkInputDim(input)
   self:checkInputDim(gradOutput)
   assert(self.train == true, 'should be in training mode when self.train is true')
   assert(self.save_mean and self.save_std, 'must call :updateOutput() first')

   input, gradOutput = makeContiguous(self, input, gradOutput)

   scale = scale or 1
   if gradInput then
      gradInput:resizeAs(gradOutput)
   end

   input.THNN.BatchNormalization_backward(
      input:cdata(),
      gradOutput:cdata(),
      THNN.optionalTensor(gradInput),
      THNN.optionalTensor(gradWeight),
      THNN.optionalTensor(gradBias),
      THNN.optionalTensor(self.weight),
      self.save_mean:cdata(),
      self.save_std:cdata(),
      scale)

   return self.gradInput
end


function nn.BatchNormalization:updateGradInput(input, gradOutput)
   if self.train then
      return backward(self, input, gradOutput, 1, self.gradInput)
   else
      self.zeros = self.zeros or torch.Tensor():typeAs(self.running_mean)
      self.zeros:resizeAs(self.running_mean):zero()
      self.gradInput:resizeAs(gradOutput)

      input.THNN.BatchNormalization_updateOutput(
         gradOutput:cdata(),
         self.gradInput:cdata(),
         THNN.optionalTensor(self.weight),
         THNN.NULL,
         self.zeros:cdata(),
         self.running_var:cdata(),
         THNN.NULL,
         THNN.NULL,
         false,
         self.momentum,
         self.eps)

      return self.gradInput
   end
end

local cmd = torch.CmdLine()
cmd:option('-data',       'data/',         'Path to dataset')
cmd:option('-dataset',    'imagenet', 'Options: imagenet | cifar10')
cmd:option('-manualSeed', 2,          'Manually set RNG seed')
cmd:option('-nGPU',       1,          'Number of GPUs to use by default')
cmd:option('-backend',    'cudnn',    'Options: cudnn | cunn')
cmd:option('-cudnn',      'fastest',  'Options: fastest | default | deterministic')
cmd:option('-gen',        'temp',     'Path to save generated files')
cmd:option('-nThreads',   2,          'number of data loading threads')
cmd:option('-batchSize',  1,          'mini-batch size (1 = pure stochastic)')
cmd:option('-tenCrop',    'false',    'Ten-crop testing')
cmd:option('-depth',      101,        'ResNet depth: 18 | 34 | 50 | 101 | ...', 'number')
cmd:option('-noise',      1,          'adversarial noise intensity [px]')
cmd:option('-var',        0.17,       'stochastic input variance')
cmd:text()
local opt = cmd:parse(arg or {})
opt.tenCrop = opt.tenCrop ~= 'false'
if not paths.dirp(opt.data) then
   assert(false, 'Error: dataset does not exist')
end
if not (opt.depth == 18 or opt.depth == 34 or opt.depth == 50 or opt.depth == 101) then
   assert(false, 'Error: unknown residual network')
end
torch.manualSeed(opt.manualSeed)
cutorch.manualSeedAll(opt.manualSeed)
cutorch.setDevice(1)


-- acquire model
paths.mkdir(opt.gen)
local path_to_model = paths.concat(opt.gen, 'resnet-'..opt.depth..'.t7')
if not paths.filep(path_to_model) then
   local aws_http = 'http://torch7.s3-website-us-east-1.amazonaws.com/data/'
   os.execute('wget -O '..path_to_model..' '..aws_http..'resnet-'..opt.depth..'.t7')
end
local model = torch.load(path_to_model)
local modelv = nn.toStochasticModel(model)
local loss = nn.CrossEntropyCriterion()


-- Set the CUDNN flags
if opt.cudnn == 'fastest' then
   cudnn.fastest = true
   cudnn.benchmark = true
elseif opt.cudnn == 'deterministic' then
   -- Use a deterministic convolution implementation
   model:apply(function(m)
      if m.setMode then m:setMode(1, 1, 1) end
   end)
end


model:cuda()
modelv:cuda()
loss:cuda()


local x, xv, yt
local _, validationLoader = DataLoader.create(opt)
local size = validationLoader:size()
local std = { 0.229, 0.224, 0.225 }


-- display helper
local display = require('display')
display:init()
local name = require('imagenet-cat')
local img = {
   raw = torch.FloatTensor(),
   adv = torch.FloatTensor(),
   noise = torch.FloatTensor(),
}
local pred = {gt = 0, br = {}, pr = {}}


-- Computes test loop
model:evaluate()
for n, sample in validationLoader:run() do
   -- Copies the input to a CUDA tensor, if using 1 GPU, or to pinned memory,
   -- if using DataParallelTable. The target is always copied to a CUDA tensor
   x = x or (opt.nGPU == 1 and torch.CudaTensor() or cutorch.createCudaHostTensor())
   xv = xv or (opt.nGPU == 1 and torch.CudaTensor() or cutorch.createCudaHostTensor())
   yt = yt or torch.CudaTensor()
   x:resize(sample.x:size()):copy(sample.x)
   xv:resize(sample.x:size()):fill(opt.var)
   yt:resize(sample.yt:size()):copy(sample.yt)


   -- standard-feedforward(x)
   local y = model:forward(x)

   -- generate adv noise
   loss:forward(y, yt)
   local cost = loss:backward(y, yt)
   local x_grad = model:updateGradInput(x, cost)
   local noise = x_grad:sign()
   for c = 1, 3 do
      noise[{{},{c},{},{}}]:mul(opt.noise/std[c]/255)
   end
   img.raw:resize(x[1]:size()):copy(x[1])
   img.noise:resize(x[1]:size()):copy(noise[1])
   x:add(noise)
   img.adv:resize(x[1]:size()):copy(x[1])

   -- standard-feedforward(x + noise)
   local y = model:forward(x)
   local _, idx = y:float():view(-1):sort(1, true)
   for i = 1, 5 do
      table.insert(pred.br, idx[i])
   end

   -- stochastic-feedforward(x + noise)
   local y = modelv:forward(x, xv)
   local _, idx = y:float():view(-1):sort(1, true)
   for i = 1, 5 do
      table.insert(pred.pr, idx[i])
   end

   -- display
   pred.gt = yt[1]
   display:loop(img, pred, name)
   pred.br = {}
   pred.pr = {}
   os.execute("sleep " .. tonumber(1.5))
end
