# Copyright 2017 Mamy André-Ratsimbazafy
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

proc pack_panel[T](k: int,
                      pM: ptr T, # pointer to first element of Tensor data
                      lsm, ssm: int, # Leading and secondary (dimension) stride of M, Leading: incColA/incRowB.
                      LR: static[int], # Leading block dimension, MR for A (MxK), NR for B (KxN)
                      buffer: var ptr T # N = MCKC for A, KCNC for B
                      ) = #{.noSideEffect.} =
  ## Pack blocks of size LR of the matrices in the corresponding buffer
  var M = pM
  for s in 0 ..< k: # Loop along the leaing dimension
    for lead in 0 ..< LR:
      buffer[lead] = M[lead*lsm]
    buffer += LR
    M += ssm

proc pack_dim[T](lc, kc: int, # lc = mc for A (MxK matrix) and lc = nc for B (KxN matrix)
                    pM: ptr T, # pointer to first element of Tensor data
                    lsm, ssm: int, # Leading and secondary (dimension) stride of M, Leading: incColA/incRowB.
                    LR: static[int], # Leading block dimension, MR for A (MxK), NR for B (KxN)
                    pBuffer: ptr T # N = MCKC for A, KCNC for B
                    ) = #{.noSideEffect.} =

  let lp = lc div LR # Number of whole blocks along leading dim
  let lr = lc mod MR # Reminder of leading dim

  var M = pM
  var buffer = pBuffer

  for lead in 0..<lp:
    pack_panel(kc, M, lsm, ssm, LR, buffer)
    M += LR * lsm

  if lr > 0:
    for s in 0 ..< kc:
      for lead in 0 ..< lr:
        buffer[lead] = M[lead * lsm]
      for lead in lr ..< LR:
        buffer[lead] = 0.T
      buffer += LR
      M      += ssm