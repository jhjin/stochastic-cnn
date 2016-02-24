require('torch')
require('qtwidget')
require('image')

local display = {}
local width = 800
local height = 600
local fontSize = {large = 20, medium = 15, small = 12}

function display:init()
   self.win = qtwidget.newwindow(width, height, 'Robust CNNs under Adversarial Noise')
   local win = self.win

   -- bgcolor
   win:setcolor("black")
   win:rectangle(0, 0, width, height)
   win:fill()

   -- title
   win:setcolor(1, 1, 1, 0.6)
   win:setfont(qt.QFont{size=fontSize.large})
   win:moveto(90,50); win:show("Robust Convolutional Neural Networks under Adversarial Noise")

   -- image index
   win:setcolor(1, 1, 1, 0.6)
   win:setfont(qt.QFont{size=fontSize.medium})
   win:moveto(44+6,           110); win:show("Original")
   win:moveto(44+6+20*1+224*1,110); win:show("Adversarial")
   win:moveto(44+6+20*2+224*2,110); win:show("Noise (1px)")

   -- credit
   win:setcolor(1, 1, 1, 0.6)
   win:setfont(qt.QFont{size=fontSize.small})
   win:moveto(6,590); win:show("Tested on ResNet-101 (https://github.com/facebook/fb.resnet.torch)")

   -- label index
   win:setcolor(1, 1, 1, 0.4)
   win:setfont(qt.QFont{size=fontSize.medium})
   win:moveto(160,390); win:show("Top")
   win:moveto(205,390); win:show("Ground Truth")
   win:moveto(335,390); win:show("Baseline (18% acc)")
   win:moveto(500,390); win:show("Proposed (60% acc)")
end

function display:loop(img, pred, name)
   local win = self.win

   -- display images
   image.display{image=img.raw,   win=win, x=44,            y=120}
   image.display{image=img.adv,   win=win, x=44+20*1+224*1, y=120}
   image.display{image=img.noise, win=win, x=44+20*2+224*2, y=120}

   -- bgcolor
   win:setcolor("black")
   win:rectangle(0, 395, width, 180)
   win:fill()

   -- paint shade for table rows
   local row = 30
   win:setfont(qt.QFont{size=fontSize.medium})
   win:setcolor(1, 1, 1, 0.05); win:rectangle(150, 400,       500, row); win:fill()
   win:setcolor(1, 1, 1, 0.01); win:rectangle(150, 400+row*1, 500, row); win:fill()
   win:setcolor(1, 1, 1, 0.05); win:rectangle(150, 400+row*2, 500, row); win:fill()
   win:setcolor(1, 1, 1, 0.01); win:rectangle(150, 400+row*3, 500, row); win:fill()
   win:setcolor(1, 1, 1, 0.05); win:rectangle(150, 400+row*4, 500, row); win:fill()

   -- write table row numbering
   win:setcolor(1, 1, 1, 0.4)
   win:setfont(qt.QFont{size=fontSize.medium})
   win:moveto(170,390+row*1); win:show("1")
   win:moveto(170,390+row*2); win:show("2")
   win:moveto(170,390+row*3); win:show("3")
   win:moveto(170,390+row*4); win:show("4")
   win:moveto(170,390+row*5); win:show("5")

   local chk = function (a, b)
      if (a == b) then
         win:setcolor(1, 0, 0, 0.6)
      else
         win:setcolor(1, 1, 1, 0.6)
      end
   end

   -- show ground truth class name
   local limit = 14
   win:setcolor(1, 0, 0, 0.6)
   win:setfont(qt.QFont{size=fontSize.medium})
   win:moveto(205,390+row*1); win:show(string.sub(name[pred.gt],1,limit))

   -- show probable class names from baseline
   win:setcolor(1, 1, 1, 0.6)
   win:setfont(qt.QFont{size=fontSize.medium})
   win:moveto(335,390+row*1); chk(pred.gt, pred.br[1]); win:show(string.sub(name[pred.br[1]],1,limit))
   win:moveto(335,390+row*2); chk(pred.gt, pred.br[2]); win:show(string.sub(name[pred.br[2]],1,limit))
   win:moveto(335,390+row*3); chk(pred.gt, pred.br[3]); win:show(string.sub(name[pred.br[3]],1,limit))
   win:moveto(335,390+row*4); chk(pred.gt, pred.br[4]); win:show(string.sub(name[pred.br[4]],1,limit))
   win:moveto(335,390+row*5); chk(pred.gt, pred.br[5]); win:show(string.sub(name[pred.br[5]],1,limit))

   -- show probable class names from proposed
   win:setcolor(1, 1, 1, 0.6)
   win:setfont(qt.QFont{size=fontSize.medium})
   win:moveto(500,390+row*1); chk(pred.gt, pred.pr[1]); win:show(string.sub(name[pred.pr[1]],1,limit))
   win:moveto(500,390+row*2); chk(pred.gt, pred.pr[2]); win:show(string.sub(name[pred.pr[2]],1,limit))
   win:moveto(500,390+row*3); chk(pred.gt, pred.pr[3]); win:show(string.sub(name[pred.pr[3]],1,limit))
   win:moveto(500,390+row*4); chk(pred.gt, pred.pr[4]); win:show(string.sub(name[pred.pr[4]],1,limit))
   win:moveto(500,390+row*5); chk(pred.gt, pred.pr[5]); win:show(string.sub(name[pred.pr[5]],1,limit))
end

return display
