#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#-*- coding: utf-8 -*-

import numpy as np
import paddle.fluid as fluid
import parl
from parl import layers
import random
from env import ContainerNumber
from env import NodeNumber
flag=[]
flag_temp=[]
for o in range( ContainerNumber * NodeNumber):
    flag.append(0)
    flag_temp.append(0)

class Agent(parl.Agent):
    def __init__(self,
                 algorithm,
                 obs_dim,
                 act_dim,
                 e_greed,   # random exploration probability
                 e_greed_decrement=0):  
        assert isinstance(obs_dim, int) # State dimension, int
        assert isinstance(act_dim, int) # Action dimension, int
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        super(Agent, self).__init__(algorithm)

        self.global_step = 0
        self.update_target_steps = 200  
        self.e_greed = e_greed  
        self.e_greed_decrement = e_greed_decrement  


    def build_program(self):
        self.pred_program = fluid.Program()
        self.learn_program = fluid.Program()

        with fluid.program_guard(self.pred_program):  # to predict actions
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            # set obs as data variable
            self.value = self.alg.predict(obs)

        with fluid.program_guard(self.learn_program):  # to update Q network
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            action = layers.data(name='act', shape=[2], dtype='int32')
            reward = layers.data(name='reward', shape=[], dtype='float32')
            next_obs = layers.data(
                name='next_obs', shape=[self.obs_dim], dtype='float32')
            terminal = layers.data(name='terminal', shape=[], dtype='bool')
            self.cost = self.alg.learn(obs, action, reward, next_obs, terminal)

    def sample(self, obs):
        sample = random.random()  # 0~1
        limit = ContainerNumber * NodeNumber - 1
        if sample < self.e_greed:
            temp = random.randint(0,limit)
            while flag[temp] == 1 or flag_temp[temp % ContainerNumber] == 1:
                temp = random.randint(0,limit)
            act = temp
            flag[act] = 1
            flag_temp[act] = 1
            flag_temp[temp % ContainerNumber] = 1
        else:
            act = self.predict(obs)  # the optimal action
        self.e_greed = max(
            0.01, self.e_greed - self.e_greed_decrement)  
        return act

    def predict(self, obs):  # choose the optimal action
        # convert numbers to 1D vectors
        obs = np.expand_dims(obs, axis=0)
        obs = [obs]  
        obs = np.array(obs)

        pred_Q = self.fluid_executor.run(
            self.pred_program,     
            feed={'obs': obs.astype('float32')},
            fetch_list=[self.value])[0]

        # pred_Q_sorted is the sorted index value
        preq_Q_sorted = np.argsort(pred_Q)
        preq_Q_sorted = preq_Q_sorted[0]
        act = preq_Q_sorted[self.act_dim - 1]  
        i = 1
        while flag[act] == 1 or flag_temp[act % ContainerNumber] == 1:
            i += 1
            act = preq_Q_sorted[self.act_dim - i]
        flag[act] = 1
        flag_temp[act] = 1
        flag_temp[act % ContainerNumber ] = 1
        return act


    def learn(self, obs, act, reward, next_obs, terminal):
        #  Synchronize the parameters of model and target_model every 200 training steps
        if self.global_step % self.update_target_steps == 0:
            self.alg.sync_target()
        self.global_step += 1
        act = np.expand_dims(act, -1)
        feed = {
            'obs': obs.astype('float32'),
            'act': act.astype('int32'),
            'reward': reward,
            'next_obs': next_obs.astype('float32'),
            'terminal': terminal
        }
        cost = self.fluid_executor.run(
            self.learn_program, feed=feed, fetch_list=[self.cost])[0] 
        return cost
