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

template gemm_micro_kernelT[T](
            kc: int,
            alpha: T,
            pA: typed,
            pB: typed,
            beta: T,
            pC: typed,
            incRowC, incColC: int): untyped =
  var AB: array[MR*NR, T]
  
  var A = pA
  var B = pB
  var C = pC

  ## Compute A*B
  for _ in 0 ..< kc:
    for j in 0 ..< NR:
      for i in 0 .. <MR:
        AB[i + j*MR] += A[i] * B[j]
    A += MR
    B += NR

  ## C <- beta * C
  if beta == 0.T:
    for j in 0 ..< NR:
      for i in 0 ..< MR:
        C[i*incRowC + j*incColC] = 0.T ## Todo use a pointer for the sequences as well
  elif beta != 1.T:
    for j in 0 ..< NR:
      for i in 0 ..< MR:
        C[i*incRowC + j*incColC] *= beta

  ## C <- C + alpha*AB, alpha !=0
  if alpha == 1.T:
    for j in 0 ..< NR:
      for i in 0 ..< MR:
        C[i*incRowC + j*incColC] += AB[i + j*MR]
  else:
    for j in 0 ..< NR:
      for i in 0 ..< MR:
        C[i*incRowC + j*incColC] += alpha*AB[i + j*MR]

proc gemm_micro_kernel[T](kc: int,
                          alpha: T,
                          A: BufferPtr[MCKC, T],
                          B: BufferPtr[KCNC, T],
                          beta: T,
                          C: BufferPtr[MRNR, T],
                          incRowC, incColC: int
                          ) = # {.noSideEffect.} =
  gemm_micro_kernelT(kc, alpha, A, B, beta, C, incRowC, incColc)

proc gemm_micro_kernel[T](kc: int,
                          alpha: T,
                          A: BufferPtr[MCKC, T],
                          B: BufferPtr[KCNC, T],
                          beta: T,
                          C: SeqPtr[T],
                          incRowC, incColC: int
                          ) = # {.noSideEffect.} =
  gemm_micro_kernelT(kc, alpha, A, B, beta, C, incRowC, incColc)