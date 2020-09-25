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
    # self._original_loss = None

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
      # return tf.estimator.SessionRunArgs(fetches=['cls_loss'])
  
  def after_run(self, run_context, run_values):
    if self._do_swa:
      # self._original_loss = run_values.results
      self._trainable_var = {
          v.name: v for v in tf.compat.v1.get_collection(
              tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
      }
      (swa_op, swa_to_weights, save_weight_backups, 
        restore_weight_backups) = optimization.SWAops(self._trainable_var)
      run_context.session.run(save_weight_backups)
      run_context.session.run(swa_op)
      run_context.session.run(swa_to_weights)
      
      run_context.session.run(restore_weight_backups)

class AveragingWeightLoggingHook(tf.estimator.SessionRunHook):
  def __init__(self, wa_opt, logging_period=100):
    super(AveragingWeightLoggingHook, self).__init__()
    self.wa_opt = wa_opt
    # self._name_scope = name_scope
    self._logging_period = logging_period
    self._timer = tf.estimator.SecondOrStepTimer(every_steps=logging_period)
    self._steps = 0
  
  def begin(self):
    # self._vars = tf.get_collection(self._name_scope)
    # self._wa_vars_map = self.wa_opt.averaging_var_map()
    # print(self._wa_vars_map)
    pass

  def before_run(self, run_context):
    self._should_trigger = self._timer.should_trigger_for_step(self._steps)
    if self._should_trigger:
      self._wa_vars_map = self.wa_opt.averaging_var_map()
      # print(f'map between vars and wa_vars: {self._wa_vars_map}')
      return tf.estimator.SessionRunArgs(fetches=self._wa_vars_map)
    else:
      return None

  def after_run(self, run_context, run_values):
    _ = run_context
    if self._should_trigger:
      for name, value in run_values.results.items():
        tf.compat.v1.logging.info(f'{name}: {value}')
    self._steps += 1

class AveragingWeightSavingHook(tf.estimator.SessionRunHook):
  def __init__(self, wa_opt, name_scope=None, saving_period=100):
    super(AveragingWeightSavingHook, self).__init__()
    self.wa_opt = wa_opt
    self._name_scope = name_scope
    self._logging_period = saving_period
    self._timer = tf.estimator.SecondOrStepTimer(every_steps=saving_period)
    self._steps = 0
  
  def begin(self):
    # self._vars = tf.get_collection(self._name_scope)
    # self._wa_vars_map = self.wa_opt.averaging_var_map()
    # print(self._wa_vars_map)
    pass

  def before_run(self, run_context):
    self._should_trigger = self._timer.should_trigger_for_step(self._steps)
    if self._should_trigger:
      self._vars = tf.get_collection(self._name_scope)
      self._wa_vars_map = self.wa_opt.averaging_var_map(self._vars)
      self._wa_vars = self._wa_vars_map.keys()
      return tf.estimator.SessionRunArgs(fetches=self._wa_vars)
    else:
      return None

  def after_run(self, run_context, run_values):
    _ = run_context
    if self._should_trigger:
      saving_dict = dict(zip(self._wa_vars, run_values.results))
      self.ckpt = tf.train.Checkpoint(saving_dict)
      self.ckpt.save(f'./wa-ckpt-{self._steps}')
    self._steps += 1

class RestoreAveragingWeight(tf.estimator.SessionRunHook):
  # def __init__(self, wa_opt, scope=None):
  def __init__(self, scope=None):
    super(RestoreAveragingWeight, self).__init__()
    # assert hasattr(wa_opt, _avg_vars)
    # self._opt = wa_opt
    self._scope = scope
    # self._restore_op = None
  
  def begin(self):
    # tvars = tf.trainable_variables()
    wa_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self._scope)
    
    print("***Printing the averaged variables from checkpoint***")
    for wa_var in wa_vars:
      tf.compat.v1.print({wa_var.name: wa_var})
    # for var in tavrs:
    #   var_name = self._opt._get_variable_name(var.name)
    #   with tf.compat.v1.variable_scope(self._scope):
    #     average_var = tf.compat.v1.get_variable(name=var_name)
    # self._restore_op = tf.assign(v, self._opt.)

class LoggingHook(tf.train.SessionRunHook):
  def __init__(self, query_dict={}, log_steps=100):
    self._log_steps = log_steps
    self._count = 0
    self._log_names = []
    self._query_names = []
    for key, value in query_dict.items():
      if key is not None and value is not None:
        self._log_names.append(key)
        self._query_names.append(value)
      else:
        continue

  def begin(self):
    # self._query_tensors = [tf.get_default_graph().get_tensor_by_name(name) for name in self._query_names]
    pass

  def before_run(self, run_context):
    if self._count % self._log_steps == 0:
      return tf.train.SessionRunArgs(self._query_names)
    else:
      pass

  def after_run(self, run_context, run_values):
    if self._count % self._log_steps == 0:
      for key, value in zip(self._log_names, run_values.results):
        print(f"{key}: {value}\n")
    self._count += 1