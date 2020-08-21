# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
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

import tensorflow as tf
import optimization
import time

# report latency and throughput during eval
class LogEvalRunHook(tf.estimator.SessionRunHook):
  def __init__(self, global_batch_size, hvd_rank=-1):
    self.global_batch_size = global_batch_size
    self.hvd_rank = hvd_rank
    self.count = 0
    self.time_list = []

  def before_run(self, run_context):
    self.t0 = time.time()

  def after_run(self, run_context, run_values):
    elapsed_secs = time.time() - self.t0
    self.count += 1
    self.time_list.append(elapsed_secs)

# report throughput during training
class LogTrainRunHook(tf.estimator.SessionRunHook):
  def __init__(self, global_batch_size, hvd_rank=-1, save_checkpoints_steps=1000, num_steps_ignore_xla=100):
    self.global_batch_size = global_batch_size
    self.hvd_rank = hvd_rank
    self.save_checkpoints_steps = save_checkpoints_steps

    self.total_time = 0.0
    self.count = 0 # Holds number of iterations, including skipped iterations for fp16 loss scaling
    self.skipped = 0
    self.num_steps_ignore_xla = num_steps_ignore_xla 
    #initial steps while xla is still compilingneed to be ignored from throughput computation 
  def after_create_session(self, session, coord):
    self.init_global_step = session.run(tf.train.get_global_step())

  def before_run(self, run_context):
    self.t0 = time.time()
    return tf.estimator.SessionRunArgs(
        fetches=['step_update:0'])

  def after_run(self, run_context, run_values):
    elapsed_secs = time.time() - self.t0
    self.global_step = run_values.results[0]
    self.count += 1

    # Removing first step + first two steps after every checkpoint save
    if (self.global_step - self.init_global_step) <= self.num_steps_ignore_xla or (self.global_step - self.init_global_step) % self.save_checkpoints_steps < 5:
      print("Skipping time record for ", self.global_step, " due to checkpoint-saving/warmup overhead")
      self.skipped += 1
    else:
      self.total_time += elapsed_secs

  # def end(self, session):
  #   num_global_steps = self.global_step - self.init_global_step

  #   self.skipped = (num_global_steps // self.save_checkpoints_steps) * 2 + \
  #                  min(2, num_global_steps % self.save_checkpoints_steps) - 1

class SWAUpdateHook(tf.estimator.SessionRunHook):
  def __init__(self, num_steps_per_swa_update, hvd_rank=-1):
    self.num_steps_per_swa_update = num_steps_per_swa_update
    self.hvd_rank = hvd_rank
    self._step_timer = 0
    self._do_swa = False
    self.swa_times = 0
    self._original_loss = None

  # def after_create_session(self, session, coord):

  def before_run(self, run_context):
    # def make_inference(logist, label_ids):
    #   return tf.reduce_mean(tf.cast(tf.equal(label_ids,
    #       tf.argmax(logits, axis=-1, output_type=tf.int32)), dtype=tf.float32))
    self._step_timer += 1
    if self._step_timer % self.num_steps_per_swa_update == 0:
      self._step_timer = 0
      self.swa_times += 1
      self._do_swa = True
      return tf.estimator.SessionRunArgs(fetches=['cls_loss'])
  
  def after_run(self, run_context, run_values):
    if self._do_swa:
      self._original_loss = run_values.results
      self._trainable_var = {
          v.name: v for v in tf.compat.v1.get_collection(
              tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
      }
      (swa_op, swa_to_weights, save_weight_backups, 
        restore_weight_backups) = optimization.SWAops(_trainable_var)
      run_context.session.run(save_weight_backups)
      run_context.session.run(swa_op)
      run_context.session.run(swa_to_weights)
      
      run_context.session.run(restore_weight_backups)
