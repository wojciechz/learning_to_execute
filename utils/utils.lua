--[[
  Copyright 2014 Google Inc. All Rights Reserved.

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
]]--

require "env"
require "torch"
pcall(require, 'cunn')
require "nn"
require "nngraph"
require "optim"

--[[ Creates clones of the given network.
The clones share all weights and gradWeights with the original network.
Accumulating of the gradients sums the gradients properly.
The clone also allows parameters for which gradients are never computed
to be shared. Such parameters must be returns by the parametersNoGrad
method, which can be null.
--]]
function cloneManyTimes(net, T)
  local clones = {}
  local params, gradParams = net:parameters()
  if params == nil then
    params = {}
  end
  local paramsNoGrad
  if net.parametersNoGrad then
    paramsNoGrad = net:parametersNoGrad()
  end
  local mem = torch.MemoryFile("w"):binary()
  mem:writeObject(net)
  for t = 1, T do
    -- We need to use a new reader for each clone.
    -- We don't want to use the pointers to already read objects.
    local reader = torch.MemoryFile(mem:storage(), "r"):binary()
    local clone = reader:readObject()
    reader:close()
    local cloneParams, cloneGradParams = clone:parameters()
    local cloneParamsNoGrad
    for i = 1, #params do
      cloneParams[i]:set(params[i])
      cloneGradParams[i]:set(gradParams[i])
    end
    if paramsNoGrad then
      cloneParamsNoGrad = clone:parametersNoGrad()
      for i =1,#paramsNoGrad do
        cloneParamsNoGrad[i]:set(paramsNoGrad[i])
      end
    end
    clones[t] = clone
    collectgarbage()
  end
  mem:close()
  return clones
end

function str_hash(str)
  local hash = 1
  for i = 1, #str, 2 do
    hash = math.fmod(hash * 12345, 452930459) +
    ((string.byte(str, i) or (len - i)) * 67890) +
    ((string.byte(str, i + 1) or i) * 13579)
  end
  return hash
end

function init_gpu(gpuidx)
  if params.gpuidx > 0  then
    cutorch.setDevice(gpuidx)
  end
  make_deterministic(1)
end

function make_deterministic(seed)
  torch.manualSeed(seed)
  if params.gpuidx > 0 then
    cutorch.manualSeed(seed)
    torch.zeros(1, 1):cuda():uniform()
  end
end

function copy_table(to, from)
  assert(#to == #from)
  for i = 1, #to do
    to[i]:copy(from[i])
  end
end

function os.capture(cmd)
  local f = assert(io.popen(cmd, 'r'))
  local s = assert(f:read('*a'))
  f:close()
  s = string.gsub(s, '[\n\r]+', ' ')
  return s
end

function script_path()
  return debug.getinfo(2, "S").source:sub(2)
end

function argmax(vector)
  if vector:dim() == 1 then
    for i = 1, vector:size(1) do
      if vector[i] == vector:max() then
        return i
      end
    end
  else
    error("Argmax only supports vectors")
  end
end

floor = torch.floor
ceil = torch.ceil
random = torch.random
