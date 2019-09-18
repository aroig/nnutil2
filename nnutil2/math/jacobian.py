#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# nnutil2 - Tensorflow utilities for training neural networks
# Copyright (c) 2019, Abdó Roig-Maranges <abdo.roig@gmail.com>
#
# This file is part of 'nnutil2'.
#
# This file may be modified and distributed under the terms of the 3-clause BSD
# license. See the LICENSE file for details.

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import gradients as gradient_ops
from tensorflow.python.ops.parallel_for import control_flow_ops
from tensorflow.python.util import nest


def jacobian(output, inputs, use_pfor=False, parallel_iterations=None):
  """Computes jacobian of `output` w.r.t. `inputs`.
  Args:
    output: A tensor.
    inputs: A tensor or a nested structure of tensor objects.
    use_pfor: If true, uses pfor for computing the jacobian. Else uses
      tf.while_loop.
    parallel_iterations: A knob to control how many iterations and dispatched in
      parallel. This knob can be used to control the total memory usage.
  Returns:
    A tensor or a nested strucutre of tensors with the same structure as
    `inputs`. Each entry is the jacobian of `output` w.r.t. to the corresponding
    value in `inputs`. If output has shape [y_1, ..., y_n] and inputs_i has
    shape [x_1, ..., x_m], the corresponding jacobian has shape
    [y_1, ..., y_n, x_1, ..., x_m]. Note that in cases where the gradient is
    sparse (IndexedSlices), jacobian function currently makes it dense and
    returns a Tensor instead. This may change in the future.
  """
  flat_inputs = nest.flatten(inputs)
  output_tensor_shape = output.shape
  output_shape = array_ops.shape(output)
  output = array_ops.reshape(output, [-1])

  def loop_fn(i):
    y = array_ops.gather(output, i)
    return gradient_ops.gradients(y, flat_inputs)

  try:
    output_size = int(output.shape[0])
  except TypeError:
    output_size = array_ops.shape(output)[0]

  if use_pfor:
    pfor_outputs = control_flow_ops.pfor(
        loop_fn, output_size, parallel_iterations=parallel_iterations)
  else:
    pfor_outputs = control_flow_ops.for_loop(
        loop_fn,
        [output.dtype] * len(flat_inputs),
        output_size,
        parallel_iterations=parallel_iterations)

  for i, out in enumerate(pfor_outputs):
    if isinstance(out, ops.Tensor):
      new_shape = array_ops.concat(
          [output_shape, array_ops.shape(out)[1:]], axis=0)
      out = array_ops.reshape(out, new_shape)
      out.set_shape(output_tensor_shape.concatenate(flat_inputs[i].shape))
      pfor_outputs[i] = out

  return nest.pack_sequence_as(inputs, pfor_outputs)


def batch_jacobian(output, inp, use_pfor=False, parallel_iterations=None):
  """Computes and stacks jacobians of `output[i,...]` w.r.t. `input[i,...]`.
  e.g.
  x = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
  y = x * x
  jacobian = batch_jacobian(y, x)
  # => [[[2,  0], [0,  4]], [[6,  0], [0,  8]]]
  Args:
    output: A tensor with shape [b, y1, ..., y_n]. `output[i,...]` should
      only depend on `inp[i,...]`.
    inp: A tensor with shape [b, x1, ..., x_m]
    use_pfor: If true, uses pfor for computing the Jacobian. Else uses a
      tf.while_loop.
    parallel_iterations: A knob to control how many iterations and dispatched in
      parallel. This knob can be used to control the total memory usage.
  Returns:
    A tensor `t` with shape [b, y_1, ..., y_n, x1, ..., x_m] where `t[i, ...]`
    is the jacobian of `output[i, ...]` w.r.t. `inp[i, ...]`, i.e. stacked
    per-example jacobians.
  Raises:
    ValueError: if first dimension of `output` and `inp` do not match.
  """
  output_shape = output.shape
  if output_shape[0] != inp.shape[0]:
    raise ValueError("Need first dimension of output shape (%s) and inp shape "
                     "(%s) to match." % (output.shape, inp.shape))
  if output_shape.is_fully_defined():
    batch_size = int(output_shape[0])
    output_row_size = output_shape.num_elements() // batch_size
  else:
    output_shape = array_ops.shape(output)
    batch_size = output_shape[0]
    output_row_size = array_ops.size(output) // batch_size
  inp_shape = array_ops.shape(inp)
  # Flatten output to 2-D.
  with ops.control_dependencies(
      [check_ops.assert_equal(batch_size, inp_shape[0])]):
    output = array_ops.reshape(output, [batch_size, output_row_size])

  def loop_fn(i):
    y = array_ops.gather(output, i, axis=1)
    return gradient_ops.gradients(y, inp)[0]

  if use_pfor:
    pfor_output = control_flow_ops.pfor(loop_fn, output_row_size,
                                        parallel_iterations=parallel_iterations)
  else:
    pfor_output = control_flow_ops.for_loop(
        loop_fn, output.dtype,
        output_row_size,
        parallel_iterations=parallel_iterations)
  if pfor_output is None:
    return None
  pfor_output = array_ops.reshape(pfor_output,
                                  [output_row_size, batch_size, -1])
  output = array_ops.transpose(pfor_output, [1, 0, 2])
  new_shape = array_ops.concat([output_shape, inp_shape[1:]], axis=0)
  return array_ops.reshape(output, new_shape)
