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

proc gemm_macro_kernel[T](mc, nc, kc: int,
                          alpha: T,
                          beta: T,
                          C: ptr T,
                          incRowC, incColC: int,
                          buffer_A: ptr T,
                          buffer_B: ptr T,
                          buffer_C: ptr T
                          ) = # {.noSideEffect.} =
  let mp = (mc+MR-1) div MR
  let np = (nc+NR-1) div NR

  let mod_mr = mc mod MR
  let mod_nr = nc mod NR

  var mr: int
  var nr: int

  for j in 0 ..< np:
    nr = if (j != np-1 or mod_nr == 0): NR
         else: mod_nr
    for i in 0 ..< mp:
      mr = if (i != mp-1 or mod_mr == 0): MR
           else: mod_mr

      if (mr==MR and nr==NR):
        gemm_micro_kernel(kc, alpha,
                          addr buffer_A[i*kc*MR],
                          addr buffer_B[j*kc*NR],
                          beta,
                          addr C[i*MR*incRowC + j*NR*incColC],
                          incRowC, incColC)
      else:
        gemm_micro_kernel(kc, alpha,
                          addr buffer_A[i*kc*MR],
                          addr buffer_B[j*kc*NR],
                          0.T,
                          buffer_C,
                          1, MR)
        gescal( mr, nr, beta,
                addr C[i*MR*incRowC + j*NR*incColC],
                incRowC, incColC)
        geaxpy( mr, nr,
                1.T,
                buffer_C,
                1, MR,
                addr C[i*MR*incRowC+j*NR*incColC],
                incRowC, incColC)
