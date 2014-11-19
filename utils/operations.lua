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

function pair_opr(hardness)
  local a, b = get_operands(hardness, 2)
  if random(2) == 1 then
    eval = a.eval + b.eval
    expr = string.format("(%s+%s)", a.expr, b.expr)
  else
    eval = a.eval - b.eval
    expr = string.format("(%s-%s)", a.expr, b.expr)
  end
  return {}, expr, eval
end

function smallmul_opr(hardness)
  local expr, eval = get_operand(hardness)
  local b = random(4 * hardness())
  local eval = eval * b
  if random(2) == 1 then
    expr = string.format("(%s*%d)", expr, b)
  else
    expr = string.format("(%d*%s)", b, expr)
  end
  return {}, expr, eval
end

function equality_opr(hardness)
  local expr, eval = get_operand(hardness)
  return {}, expr, eval
end

function vars_opr(hardness)
  local var = variablesManager:get_unused_variables(1)
  local a, b = get_operands(hardness, 2)
  if random(2) == 1 then
    eval = a.eval + b.eval
    code = {string.format("%s=%s;", var, a.expr)}
    expr = string.format("(%s+%s)", var, b.expr)
  else
    eval = a.eval - b.eval
    code = {string.format("%s=%s;", var, a.expr)}
    expr = string.format("(%s-%s)", var, b.expr)
  end
  return code, expr, eval
end

function small_loop_opr(hardness)
  local r_small = hardness()
  local var = variablesManager:get_unused_variables(1)
  local a, b = get_operands(hardness, 2)
  local loop = random(4 * hardness())
  local op = ""
  local val = 0
  if random(2) == 2 then
    op = "+"
    eval = a.eval + loop * b.eval
  else
    op = "-"
    eval = a.eval - loop * b.eval
  end
  local code = {string.format("%s=%s", var, a.expr),
                string.format("for x in range(%d):%s%s=%s", loop, var,
                                                            op, b.expr, var)}
  local expr = var
  return code, expr, eval
end

function ifstat_opr(hardness)
  local r_small = hardness()
  local a, b, c, d = get_operands(hardness, 4)
  if random(2) == 1 then
    name = ">"
    if a.eval > b.eval then
      output = c.eval
    else
      output = d.eval
    end
  else
    name = "<"
    if a.eval < b.eval then
      output = c.eval
    else
      output = d.eval
    end
  end
  local expr = string.format("(%s if %s%s%s else %s)",
                             c.expr, a.expr, name, b.expr, d.expr)
  return {}, expr, output
end
