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

proc astype*[T, U](t: Tensor[T], typ: typedesc[U]): Tensor[U] {.noSideEffect.}=
  ## Apply type conversion on the whole tensor
  result.shape = t.shape
  result.strides = t.strides
  result.offset = t.offset
  result.data = t.data.map(x => x.U)

proc fmap*[T, U](t: Tensor[T], g: T -> U): Tensor[U] {.noSideEffect.}=
  ## Map a unary function T -> U on Tensor[T]

  # We use this opportunity to reshape the data internally
  # Iteration should be almost as fast for contiguous non-sliced Tensors
  # But may avoid a lot of unnecessary computations on slices
  result.shape = t.shape
  result.strides = shape_to_strides(result.shape)
  result.offset = 0

  result.data = newSeq[U](result.shape.product)
  var i = 0 # TODO: use pairs/enumerate instead - pending https://forum.nim-lang.org/t/2972
  for val in t:
    result.data[i] = g(val)
    inc i

proc fmap2*[T, U, V](t1: Tensor[T], t2: Tensor[U], g: (T,U) -> V): Tensor[V] {.noSideEffect.}=
  ## Map a binary function (T,U) -> V on Tensor[T]
  ## It applies the function to each matching elements
  ## Tensors must have the same shape

  assert t1.shape == t2.shape

  result.shape = t1.shape
  result.strides = shape_to_strides(result.shape)
  result.offset = 0

  result.data = newSeq[U](result.shape.product)
  var i = 0 # TODO: use pairs/enumerate instead - pending https://forum.nim-lang.org/t/2972
  for ai, bi in zip(t1.values, t2.values):
    result.data[i] = g(ai, bi)
    inc i

template makeUniversal*(func_name: untyped) =
  # Lift an unary function into an exported universal function.
  #
  # Universal functions apply element-wise
  #
  # ``makeUniversal`` does not work when internal type is changing,
  # use fmap instead
  proc func_name*(t: Tensor): Tensor {.noSideEffect.}=
    ## Universal version of the function.
    ##
    ## The function can be used directly on tensors and will work element-wise.
    t.fmap(func_name)
  export func_name

template makeUniversalLocal*(func_name: untyped) =
  # Lift an unary function into a non-exported universal function
  #
  # Universal functions apply element-wise
  #
  # ``makeUniversalLocal`` does not work when internal type is changing
  # use fmap instead
  proc func_name(t: Tensor): Tensor {.noSideEffect.}=
    t.fmap(func_name)

# Unary functions from Nim math library

makeUniversal(fac)
#makeUniversal(classify)
#makeUniversal(isPowerOfTwo)
#makeUniversal(nextPowerOfTwo)
#makeUniversal(countBits32)
#makeUniversal(sum)
makeUniversal(sqrt)
makeUniversal(cbrt)
makeUniversal(ln)
makeUniversal(log10)
makeUniversal(log2)
makeUniversal(exp)
makeUniversal(arccos)
makeUniversal(arcsin)
makeUniversal(arctan)
makeUniversal(cos)
makeUniversal(cosh)
makeUniversal(sinh)
makeUniversal(sin)
makeUniversal(tan)
makeUniversal(tanh)
makeUniversal(erf)
makeUniversal(erfc)
makeUniversal(lgamma)
makeUniversal(tgamma)
makeUniversal(floor)
makeUniversal(ceil)
makeUniversal(trunc)
makeUniversal(round)
#makeUniversal(splitDecimal)
makeUniversal(degToRad)
makeUniversal(radToDeg)