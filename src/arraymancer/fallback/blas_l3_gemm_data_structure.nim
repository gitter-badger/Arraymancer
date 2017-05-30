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

# Unsafe pointer arithmetics 
template `+`[T](p: ptr T, off: int): ptr T =
  cast[ptr type(p[])](cast[ByteAddress](p) +% off * sizeof(p[]))

template `+=`[T](p: ptr T, off: int)  =
  # We don't use var ptr, that would be a ptr to a ptr
  p = p + off

template `[]`[T](p: ptr T, off: int): T =
  # var T for p[index] *= support
  (p + off)[]

template `[]=`[T](p: ptr T, off: int, val: T) =
  (p + off)[] = val


template get_data_ptr[N,T](ra: ref array[N, T]): ptr T = addr ra[0]

proc newRefArray[T: SomeNumber](N: static[int], typ: typedesc[T]): ref array[N,T] {.noSideEffect.} =
  new result
  for i in 0 ..< N:
    result[i] = 0.T


###################################################
# Defining Buffer array and a safe pointer to it
type
  BufferArray[N: static[int], T] = ref array[N, T]

  BufferPtr[N: static[int], T] = object
    when compileOption("boundChecks"):
      len: int
    at: ptr T

# Bound checking is broken here
proc `+`[N,T](p: BufferPtr[N,T], off: int): BufferPtr[N,T] {.noSideEffect.} =
  when compileOption("boundChecks"):
    # if off < p.len - N or off >= p.len:
    #   raise newException(IndexError,"Index out of bounds. You are at index " & $(N - p.len) &
    #                                 ". The length of the data is " & $N & "." &
    #                                 " You are trying to access offset " & $off & ".")
    result.len = p.len - off
  result.at = cast[ptr type(p.at[])](cast[ByteAddress](p.at) +% off * sizeof(p.at[]))

proc `+=`[N,T](p: var BufferPtr[N,T], off: int)  {.noSideEffect.} =
  p = p + off

proc `[]`[N,T](p: BufferPtr[N,T], i: int): T {.noSideEffect.}=
  result = (p + i).at[]

proc `[]`[N,T](p: var BufferPtr[N,T], i: int): var T {.noSideEffect.}=
  result = (p + i).at[]

proc `[]=`*[N,T](p: var BufferPtr[N,T], i: int, val: T) {.noSideEffect.}=
  (p + i).at[] = val

proc newBufferArrayPtr[T: SomeNumber](N: static[int], typ: typedesc[T]): (BufferArray[N, T], BufferPtr[N,T]) {.noSideEffect.} =
  var bufarr: BufferArray[N, T]
  new bufarr
  for i in 0 ..< N:
    bufarr[i] = 0.T
  
  when compileOption("boundChecks"):
    return (bufarr, BufferPtr[N,T](len: N, at: addr bufarr[0]))
  else:
    return (bufarr, BufferPtr[N,T](at: addr bufarr[0]))

# Defining a safe pointer to seq
type
  SeqPtr[T] = object
    when compileOption("boundChecks"):
      N, len: int
    at: ptr T

proc to_ptr[T](s: seq[T]): SeqPtr[T] =
  when compileOption("boundChecks"):
    return SeqPtr[T](N: s.len,
                    len: s.len,
                    at: unsafeAddr(s[0]))
  else:
    return SeqPtr[T](at: unsafeAddr(s[0]))

# Bounds checking is broken here
proc `+`[T](p: SeqPtr[T], off: int): SeqPtr[T] {.noSideEffect.} =
  when compileOption("boundChecks"):
    # if off < p.len - p.N or off > p.len:
    #   raise newException(IndexError,"Index out of bounds. You are at index " & $(p.N - p.len) &
    #                                 ". The length of the data is " & $p.N & "." &
    #                                 " You are trying to access offset " & $off & ".")
    result.N = p.N
    result.len = p.len - off
  result.at = cast[ptr type(p.at[])](cast[ByteAddress](p.at) +% off * sizeof(p.at[]))

proc `+=`[T](p: var SeqPtr[T], off: int)  {.noSideEffect.}=
  p = p + off

proc `[]`[T](p: SeqPtr[T], i: int): T {.noSideEffect.}=
  result = (p + i).at[]

proc `[]`[T](p: var SeqPtr[T], i: int): var T {.noSideEffect.}=
  result = (p + i).at[]

proc `[]=`*[T](p: var SeqPtr[T], i: int, val: T) {.noSideEffect.}=
  (p + i).at[] = val