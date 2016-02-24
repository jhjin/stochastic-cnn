-- This is code common to container modules, which are collections of
-- smaller constituent modules like Parallel, Sequential, etc.
local StochasticContainer, parent = torch.class('nn.StochasticContainer', 'nn.StochasticModule')

function StochasticContainer:__init(...)
    parent.__init(self, ...)
    self.modules = {}
end

function StochasticContainer:add(module)
    table.insert(self.modules, module)
    return self
end

function StochasticContainer:get(index)
    return self.modules[index]
end

function StochasticContainer:size()
    return #self.modules
end

function StochasticContainer:applyToModules(func)
    for _, module in ipairs(self.modules) do
        func(module)
    end
end

function StochasticContainer:training()
    self:applyToModules(function(module) module:training() end)
    parent.training(self)
end

function StochasticContainer:evaluate()
    self:applyToModules(function(module) module:evaluate() end)
    parent.evaluate(self)
end

function StochasticContainer:share(mlp, ...)
    for i=1,#self.modules do
        self.modules[i]:share(mlp.modules[i], ...);
    end
end

function StochasticContainer:reset(stdv)
    self:applyToModules(function(module) module:reset(stdv) end)
end

function StochasticContainer:parameters()
    local function tinsert(to, from)
        if type(from) == 'table' then
            for i=1,#from do
                tinsert(to,from[i])
            end
        else
            table.insert(to,from)
        end
    end
    local w = {}
    local gw = {}
    for i=1,#self.modules do
        local mw,mgw = self.modules[i]:parameters()
        if mw then
            tinsert(w,mw)
            tinsert(gw,mgw)
        end
    end
    return w,gw
end
