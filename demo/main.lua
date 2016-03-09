require('torch')
require('paths')
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
cmd:option('-data',       'data/',    'Path to dataset')
cmd:option('-dataset',    'imagenet', 'Options: imagenet | cifar10')
cmd:option('-manualSeed', 2,          'Manually set RNG seed')
cmd:option('-nGPU',       1,          'Number of GPUs to use by default')
cmd:option('-backend',    'cudnn',    'Options: cudnn | cunn')
cmd:option('-cudnn',      'fastest',  'Options: fastest | default | deterministic')
cmd:option('-gen',        'temp',     'Path to save generated files')
cmd:option('-nThreads',   2,          'number of data loading threads')
cmd:option('-batchSize',  16,         'mini-batch size (1 = pure stochastic)')
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

--[[ Create model
if opt.nGPU > 1 then
   local gpus = torch.range(1, opt.nGPU):totable()
   local fastest, benchmark = cudnn.fastest, cudnn.benchmark

   local dpt = nn.DataParallelTable(1, true, true)
   :add(model, gpus)
   :threads(function()
      local cudnn = require 'cudnn'
      cudnn.fastest, cudnn.benchmark = fastest, benchmark
   end)
   dpt.gradInput = nil
   model = dpt
end]]

model:cuda()
modelv:cuda()
loss:cuda()


local function computeScore(output, target, nCrops)
   if nCrops > 1 then
      output = output:view(output:size(1) / nCrops, nCrops, output:size(2)):sum(2):squeeze(2)
   end

   -- Coputes the top1 and top5 error rate
   local batchSize = output:size(1)
   local _ , predictions = output:float():sort(2, true) -- descending

   -- Find which predictions match the target
   local correct = predictions:eq(target:long():view(batchSize, 1):expandAs(output))

   local top1 = 1.0 - correct:narrow(2, 1, 1):sum() / batchSize
   local top5 = 1.0 - correct:narrow(2, 1, 5):sum() / batchSize

   return top1 * 100, top5 * 100
end


local timer = torch.Timer()
local dataTimer = torch.Timer()

local x, xv, yt
local _, validationLoader = DataLoader.create(opt)
local size = validationLoader:size()
local nCrops = opt.tenCrop and 10 or 1
local top1Sum, top5Sum = 0.0, 0.0
local N = 0
local std = { 0.229, 0.224, 0.225 }
local E = {
   std = {x = {[1] = 0, [5] = 0}, adv = {[1] = 0, [5] = 0}},
   sto = {x = {[1] = 0, [5] = 0}, adv = {[1] = 0, [5] = 0}},
}


-- Computes test loop
model:evaluate()
for n, sample in validationLoader:run() do
   local dataTime = dataTimer:time().real

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
   local top1, top5 = computeScore(y:float(), yt, nCrops)
   E.std.x[1] = E.std.x[1] + top1
   E.std.x[5] = E.std.x[5] + top5

   -- generate adv noise
   loss:forward(y, yt)
   local cost = loss:backward(y, yt)
   local x_grad = model:updateGradInput(x, cost)
   local noise = x_grad:sign()
   for c = 1, 3 do
      noise[{{},{c},{},{}}]:mul(opt.noise/std[c]/255)
   end
   x:add(noise)

   -- standard-feedforward(x + noise)
   local y = model:forward(x)
   local top1, top5 = computeScore(y:float(), yt, nCrops)
   E.std.adv[1] = E.std.adv[1] + top1
   E.std.adv[5] = E.std.adv[5] + top5

   -- stochastic-feedforward(x + noise)
   local y = modelv:forward(x, xv)
   local top1, top5 = computeScore(y:float(), yt, nCrops)
   E.sto.adv[1] = E.sto.adv[1] + top1
   E.sto.adv[5] = E.sto.adv[5] + top5
   N = N + 1

   print(('[%d/%d] Time %.3f  Data %.3f  [1]x/x+/p(x+) %7.3f/%7.3f/%7.3f'..
                                      '  [5]x/x+/p(x+) %7.3f/%7.3f/%7.3f')
         :format(n, size, timer:time().real, dataTime,
                 E.std.x[1]/N, E.std.adv[1]/N, E.sto.adv[1]/N,
                 E.std.x[5]/N, E.std.adv[5]/N, E.sto.adv[5]/N))

   timer:reset()
   dataTimer:reset()
end

print(string.format('==> Error\n'..
                    '  standard(x)     - top1: %6.3f  top5: %6.3f\n'..
                    '  standard(adv)   - top1: %6.3f  top5: %6.3f\n'..
                    '  stochastic(adv) - top1: %6.3f  top5: %6.3f\n',
                    E.std.x[1]/N, E.std.x[5]/N,
                    E.std.adv[1]/N, E.std.adv[5]/N,
                    E.sto.adv[1]/N, E.sto.adv[5]/N))
