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

local VariablesManager = torch.class('VariablesManager')

function VariablesManager:__init()
  self.vars = {}
  self.last_var_idx = 0
end

function VariablesManager:get_unused_variables(number)
  local ret = {}
  if #self.vars == 0 then
    self.vars = torch.randperm(10):add(string.byte("a") - 1)
  end
  for i = 1, number do
    ret[#ret + 1] = string.char(self.vars[i + self.last_var_idx])
  end
  self.last_var_idx = self.last_var_idx + number
  return unpack(ret)
end

function VariablesManager:clean()
  self.vars = {}
  self.last_var_idx = 0
end
