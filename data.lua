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
include "utils/operations.lua"
include "utils/stack.lua"
include "utils/symbolsManager.lua"
include "utils/variablesManager.lua"
include "utils/utils.lua"

local stack = Stack()
variablesManager = VariablesManager()
symbolsManager = SymbolsManager()

function to_data(code, var, output)
  local x = {}
  local y = {}
  local output = string.format("%d.", output)
  local input = ""
  for i = 1, #code do
    input = string.format("%s%s#", input, code[i])
  end
  input = string.format("%sprint(%s)@", input, var)
  for j = 1, #input do
    table.insert(y, 0)
    table.insert(x, symbolsManager:get_symbol_idx(input:byte(j)))
  end
  for j = 1, #output do
    table.insert(x, symbolsManager:get_symbol_idx(output:byte(j)))
    table.insert(y, symbolsManager:get_symbol_idx(output:byte(j)))
  end
  local orig = string.format("%s%s", input, output)
  return {x, y, orig}
end

function compose(hardness)
  stack:clean()
  variablesManager:clean()
  local funcs = {}
  local names = {}
  for i, v in pairs(_G) do
    if (string.find(i, "_opr") ~= nil) then
      funcs[#funcs + 1] = v
      names[#names + 1] = i
    end
  end
  local code = {}
  local hard, nest = hardness()
  for h = 1, nest do
    local f_idx = random(#funcs)
    local f = funcs[f_idx]
    local code_tmp, var_tmp, output_tmp = f(hardness)
    for i = 1, #code_tmp do
      code[#code + 1] = code_tmp[i]
    end
    stack:push({var_tmp, output_tmp})
  end
  local var, output = unpack(stack:pop())
  return code, var, output
end

function get_operand(hardness)
  if stack:is_empty() then
    local eval = random(math.pow(10, hardness()))
    local expr = string.format("%d", eval)
    return expr, eval
  else
    return unpack(stack:pop())
  end
end

function get_operands(hardness, nr)
  local ret = {}
  local perm = torch.randperm(nr)
  for i = 1, nr do
    local expr, eval = get_operand(hardness)
    ret[perm[i]] = {expr=expr, eval=eval}
  end
  return unpack(ret)
end

function get_data(state)
  make_deterministic(state.seed)
  local len = state.len
  local batch_size = state.batch_size
  if state.data == nil then
    state.data = {}
    state.data.x = torch.ones(len, batch_size)
    state.data.y = torch.zeros(len, batch_size)
  end
  local x = state.data.x
  local y = state.data.y
  local count = 0
  io.write(string.format("Few exemplary newly generated " ..
                         "samples from the %s dataset:\n", state.name))
  local idx = 1
  local batch_idx = 1
  local i = 0
  while true do
    data = to_data(compose(state.hardness))
    input, target, orig = unpack(data)
    if str_hash(orig) % 3 == state.kind then
      count = count + #orig
      if idx + #input > x:size(1) then
        idx = 1
        batch_idx = batch_idx + 1
        if batch_idx > batch_size then
          break;
        end
      end
     
      for j = 1, #input do
        x[idx][batch_idx] = input[j]
        y[idx][batch_idx] = target[j]
        idx = idx + 1
        -- Added to take care of smaller state lens
        if idx > x:size(1) then 
          idx = 1 
          batch_idx = batch_idx + 1
          if batch_idx > batch_size then 
            break
          end
        end
      end
      
      if batch_idx > batch_size then 
            break
      end 
      
      if (i <= 2) then
        io.write("\tInput:\t")
        local orig = string.format("     %s", orig)
        orig = orig:gsub("#", "\n\t\t     ")
        orig = orig:gsub("@", "\n\tTarget:      ")
        io.write(orig)
        io.write("\n")
        io.write("\t-----------------------------\n")
        i = i + 1
      end
    end
  end
  io.write("\n")
end

function load_data(state)
  if state.currently_loaded_seed == state.seed then
    return
  else
    state.currently_loaded_seed = state.seed
    get_data(state)
  end
end

function hardness_fun()
  return 8, 4
end

if script_path() == "data.lua" then
  make_deterministic(1)
  print("Data verification")
  for k = 1, 1000 do
    code, var, output = compose(hardness_fun)
    output = string.format("%d", output)
    print("\n__________________\n")
    local input = ""
    for i = 1, #code do
      input = string.format("%s%s\n", input, code[i])
    end
    input = string.format("%sprint(%s)", input, var)
    print(string.format("Input: \n%s\n", input))
    print(string.format("Target: %s", output))
    lines = os.capture(string.format("python2.7 -c '%s'", input))
    print(lines)
    lines = string.sub(lines, 1, string.len(lines) - 1)
    if lines ~= output then
      print(string.format("\nERROR!\noutput from python: '%s', " ..
                          "doesn't match target output: '%s'", lines, output))
      exit(-1)
    end
  end
  print("\n__________________\n")
  print("Successfully verified coherence of generated a " .. 
        "targets with python interpreter.")
end
