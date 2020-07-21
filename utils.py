import requests      
import json          
import pandas as pd                            
import time   
import numpy as np
import random
import os
from sklearn.metrics import mean_squared_error as mse
import tensorflow as tf 



# exchange data
def state_environment_3D():
    
    root_url_orderbook = 'https://api.binance.com/api/v1/depth'

    symbol = 'BTCUSDT'

    limit = '500' 

    url_orderbook = root_url_orderbook + '?symbol=' + symbol + '&limit=' + limit

    data_orderbook = json.loads(requests.get(url_orderbook).text)

    ##########################################
    
    root_url_trades_list = 'https://api.binance.com/api/v1/trades'

    #symbol = 'BTCUSDT'

    limit_trades_list = '500' 

    url_trades_list = root_url_trades_list + '?symbol=' + symbol + '&limit=' + limit_trades_list

    data_trades_list = json.loads(requests.get(url_trades_list).text)
    ##########################################
    ############## For 3D data ###############
    eps = 1e-8
    
    data_bids_price_array = np.array(data_orderbook['bids'])[:, 0].astype(float) 
    
    data_bids_price_array_mean = data_bids_price_array.mean(axis=0)
    data_bids_price_array_var = data_bids_price_array.var(axis=0)
    data_bids_price_array_std = np.sqrt(data_bids_price_array_var + eps)
    
    data_bids_price_array_centered = data_bids_price_array - data_bids_price_array_mean
    data_bids_price_array_norm = data_bids_price_array_centered / data_bids_price_array_std

    data_bids_volume_array = np.array(data_orderbook['bids'])[:, 1].astype(float) 
    
    data_bids_volume_array_mean = data_bids_volume_array.mean(axis=0)
    data_bids_volume_array_var = data_bids_volume_array.var(axis=0)
    data_bids_volume_array_std = np.sqrt(data_bids_volume_array_var + eps)
    
    data_bids_volume_array_centered = data_bids_volume_array - data_bids_volume_array_mean
    data_bids_volume_array_norm = data_bids_volume_array_centered / data_bids_volume_array_std

    data_asks_price_array = np.array(data_orderbook['asks'])[:, 0].astype(float)
    
    data_asks_price_array_mean = data_asks_price_array.mean(axis=0)
    data_asks_price_array_var = data_asks_price_array.var(axis=0)
    data_asks_price_array_std = np.sqrt(data_asks_price_array_var + eps)
    
    data_asks_price_array_centered = data_asks_price_array - data_asks_price_array_mean
    data_asks_price_array_norm = data_asks_price_array_centered / data_asks_price_array_std

    data_asks_volume_array = np.array(data_orderbook['asks'])[:, 1].astype(float) 
    
    data_asks_volume_array_mean = data_asks_volume_array.mean(axis=0)
    data_asks_volume_array_var = data_asks_volume_array.var(axis=0)
    data_asks_volume_array_std = np.sqrt(data_asks_volume_array_var + eps)
    
    data_asks_volume_array_centered = data_asks_volume_array - data_asks_volume_array_mean
    data_asks_volume_array_norm = data_asks_volume_array_centered / data_asks_volume_array_std

    data_trades_list_isBuyerMaker = []
    for i in range(len(data_trades_list)):

        data_trades_list_isBuyerMaker.append(data_trades_list[i]['isBuyerMaker'])
    data_trades_array_isBuyerMaker_eps = np.array(data_trades_list_isBuyerMaker).astype(int) + eps
 
    data_bp_ap_bv_av_th_array_3D = np.concatenate((data_bids_price_array_norm, data_asks_price_array_norm, 
                                                         data_bids_volume_array_norm, data_asks_volume_array_norm, 
                                                         data_trades_array_isBuyerMaker_eps))

    for i in range(2):
        
        time.sleep(np.random.randint(3, 6)) # 3 ~ 5 seconds
        
        data_orderbook = json.loads(requests.get(url_orderbook).text)
        
        # data_bids_price_array
        data_bids_price_array = np.array(data_orderbook['bids'])[:, 0].astype(float) 

        data_bids_price_array_centered = data_bids_price_array - data_bids_price_array_mean
        data_bids_price_array_norm = data_bids_price_array_centered / data_bids_price_array_std

        # data_bids_volume_array
        data_bids_volume_array = np.array(data_orderbook['bids'])[:, 1].astype(float) 

        data_bids_volume_array_centered = data_bids_volume_array - data_bids_volume_array_mean
        data_bids_volume_array_norm = data_bids_volume_array_centered / data_bids_volume_array_std

        # data_asks_price_array
        data_asks_price_array = np.array(data_orderbook['asks'])[:, 0].astype(float)

        data_asks_price_array_centered = data_asks_price_array - data_asks_price_array_mean
        data_asks_price_array_norm = data_asks_price_array_centered / data_asks_price_array_std

        # data_asks_volume_array
        data_asks_volume_array = np.array(data_orderbook['asks'])[:, 1].astype(float) 

        data_asks_volume_array_centered = data_asks_volume_array - data_asks_volume_array_mean
        data_asks_volume_array_norm = data_asks_volume_array_centered / data_asks_volume_array_std

        # data_trades_list 
        data_trades_list_isBuyerMaker = []
        data_trades_list = json.loads(requests.get(url_trades_list).text)
        
        for i in range(len(data_trades_list)):
            
            data_trades_list_isBuyerMaker.append(data_trades_list[i]['isBuyerMaker'])
        data_trades_array_isBuyerMaker_eps = np.array(data_trades_list_isBuyerMaker).astype(int) + eps

        data_bp_ap_bv_av_th_array_3D = np.concatenate((data_bp_ap_bv_av_th_array_3D, data_bids_price_array_norm, 
                                                       data_asks_price_array_norm, data_bids_volume_array_norm, 
                                                       data_asks_volume_array_norm, data_trades_array_isBuyerMaker_eps))

    ############## End off 3D data ###############

    df_orderbook_bids = pd.DataFrame(data_orderbook['bids'], columns = ['price', 'volume']).astype(float)

    df_orderbook_asks = pd.DataFrame(data_orderbook['asks'], columns = ['price', 'volume']).astype(float)

    best_ask_bid_vol_price = {'volume':{'buy':df_orderbook_asks['volume'][:10], 'sell':df_orderbook_bids['volume'][:10]}, 
                          'price':{'buy':df_orderbook_asks['price'][:10], 'sell':df_orderbook_bids['price'][:10]}}

    return data_bp_ap_bv_av_th_array_3D, best_ask_bid_vol_price

# last asset price
Previous_price_global = None

def last_price():
    counter = 1
    data_last_price = False

    root_url_last_price = 'https://api.binance.com/api/v3/ticker/price'

    symbol = 'BTCUSDT'

    url_last_price = root_url_last_price + '?symbol=' + symbol
    
    while counter < 5 and not data_last_price:
        counter += 1
        try:
            data_last_price = float(json.loads(requests.get(url_last_price, timeout=3).text)['price'])

        except Exception: # Exception Вторая попытка
            print('Попытка № {}'.format(counter))
            if counter == 5:
                print('Будет присвоенна цена предыдущего состояния')

    return data_last_price if data_last_price else Previous_price_global

# initialization of weights and biases
def weight_variable(lock, shape, name, first_generation = True):
    
    if first_generation == True:
        # shape = [filter_size, filter_size, num_in_channel, num_filters]
        initializer_w = tf.contrib.layers.xavier_initializer()
        return tf.get_variable(name,
                               dtype=tf.float32,
                               shape=shape,
                               initializer=initializer_w)
    else:
        
        PATH_final_weights_biases = 'weights_biases_numpy_arrays/final_weights_biases/'
        with lock:
            weights_np = np.load(PATH_final_weights_biases+name+'.npy', allow_pickle=True)
        
        weight = tf.Variable(tf.convert_to_tensor(weights_np, np.float32), name=name)
        assert shape == weight.shape
        return weight
 
def bias_variable(lock, shape, name, first_generation = True):
    
    if first_generation == True:
        # shape = [num_filters]
        initializer_b = tf.contrib.layers.xavier_initializer()
        return tf.get_variable(name,
                               dtype=tf.float32,
                               shape=shape,
                               initializer=initializer_b)
    else:
        PATH_final_weights_biases = 'weights_biases_numpy_arrays/final_weights_biases/'
        with lock:
            biases_np = np.load(PATH_final_weights_biases+name+'.npy', allow_pickle=True)
        
        bias = tf.Variable(tf.convert_to_tensor(biases_np, np.float32), name=name)
        assert shape == bias.shape
        return bias

# neural networks price and time
### main ### 
class mainNetwork():
    def __init__(self, h_size, rnn_cell, weights, biases):

        self.rawInput = tf.placeholder(shape=[None,7500],dtype=tf.float32,name='main_rawInput')

        self.input_3d = tf.reshape(self.rawInput,shape=[-1,50,50,3], name='input_3d_reshape')

        '''
        # Можно использовать tf.nn.relu(self.conv1) tf.nn.conv3d
        self.conv1 = tf.nn.relu(tf.nn.bias_add(
            tf.nn.conv2d(self.input_3d, weights['w_conv1'], 
                         strides=[1, 3, 3, 1], padding='VALID', 
                         name='conv1_layer_main'), biases['b_conv1'], name='bias_add_1'),name='relu_add_1')  # 'SAME', name=

        self.conv2 = tf.nn.relu(tf.nn.bias_add(
            tf.nn.conv2d(self.conv1, weights['w_conv2'],
                         strides=[1, 2, 2, 1], padding='VALID', 
                         name='conv2_layer_main'), biases['b_conv2'], name='bias_add_2'),name='relu_add_2')

        self.conv3 = tf.nn.relu(tf.nn.bias_add(
            tf.nn.conv2d(self.conv2, weights['w_conv3'],
                         strides=[1, 1, 1, 1], padding='VALID', 
                         name='conv3_layer_main'), biases['b_conv3'], name='bias_add_3'),name='relu_add_3')

        self.conv4 = tf.nn.relu(tf.nn.bias_add(
            tf.nn.conv2d(self.conv3, weights['w_conv4'], 
                         strides=[1, 1, 1, 1], padding='VALID',
                         name='conv4_layer_main'), biases['b_conv4'], name='bias_add_4'),name='relu_add_4')

        '''
        self.conv1 = tf.nn.bias_add(
            tf.nn.conv2d(self.input_3d, weights['w_conv1'], 
                         strides=[1, 3, 3, 1], padding='VALID', 
                         name='conv1_layer_main'), biases['b_conv1'], name='bias_add_1')  # 'SAME', name=

        self.conv2 = tf.nn.bias_add(
            tf.nn.conv2d(self.conv1, weights['w_conv2'],
                         strides=[1, 2, 2, 1], padding='VALID', 
                         name='conv2_layer_main'), biases['b_conv2'], name='bias_add_2')

        self.conv3 = tf.nn.bias_add(
            tf.nn.conv2d(self.conv2, weights['w_conv3'],
                         strides=[1, 1, 1, 1], padding='VALID', 
                         name='conv3_layer_main'), biases['b_conv3'], name='bias_add_3')

        self.conv4 = tf.nn.bias_add(
            tf.nn.conv2d(self.conv3, weights['w_conv4'], 
                         strides=[1, 1, 1, 1], padding='VALID',
                         name='conv4_layer_main'), biases['b_conv4'], name='bias_add_4')
        
        #'''
        self.trainLength = tf.placeholder(shape=[],dtype=tf.int32,name='main_trainLength')

        self.batch_size = tf.placeholder(shape=[],dtype=tf.int32,name='main_batch_size') 

        self.convFlat = tf.reshape(tf.contrib.layers.flatten(self.conv4),
                                   [self.batch_size,self.trainLength,h_size], name='convFlat_reshape')
        
        self.state_in = rnn_cell.zero_state(self.batch_size, tf.float32)
        
        self.rnn,self.rnn_state = tf.nn.dynamic_rnn(\
                inputs=self.convFlat,cell=rnn_cell,dtype=tf.float32,initial_state=self.state_in,scope='main_rnn')
        
        self.rnn = tf.reshape(self.rnn,shape=[-1,h_size], name='rnn_reshape')
 
        self.predict = tf.matmul(self.rnn, weights['w_predict'], name='w_predict_layer_main')

        self.target = tf.placeholder(shape=[None],dtype=tf.float32,name='main_target')
        
        self.stability_constant_against_nan = tf.constant(1e-10, dtype=tf.float32, name='st_const') # == 0.0000000001

        self.error = tf.sqrt(tf.square(self.target - self.predict, name='error_square') 
                                + self.stability_constant_against_nan, name='error')

        self.maskA = tf.zeros([self.batch_size,self.trainLength//2], name='maskA')
        self.maskB = tf.ones([self.batch_size,self.trainLength//2], name='maskB')
        self.mask = tf.concat([self.maskA,self.maskB],1, name='mask')
        self.mask = tf.reshape(self.mask,[-1], name='mask_finaly')
        self.loss = tf.reduce_mean(self.error * self.mask, name='loss')
         
        self.global_step = tf.Variable(0, trainable=False, name='global_step') # с 0 более мягкий распад

        self.learning_rate_main = tf.train.exponential_decay(
                          learning_rate=0.0001,            
                          global_step=self.global_step,   
                          decay_steps=10,                 
                          decay_rate=0.99,                
                          staircase=True,                 
                          name='learning_rate_main')      
    
        #self.trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_main, name='trainer')
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.000001, name='trainer')
   
        self.updateModel = self.trainer.minimize(self.loss, global_step=self.global_step,name='updateModel')
        
### time ###
class timeNetwork():
    def __init__(self, h_size, rnn_cell_time, weights, biases):
        
        self.rawInput_time = tf.placeholder(shape=[None,7500],dtype=tf.float32,name='time_rawInput')
        self.input_3d_time = tf.reshape(self.rawInput_time,shape=[-1,50,50,3], name='time_input_3d_time_reshape')
        
        '''
        # Можно использовать tf.nn.leaky_relu(self.conv1)  # 'SAME', name=
        self.conv1_time = tf.nn.relu(tf.nn.bias_add(
            tf.nn.conv2d(self.input_3d_time, weights['w_conv1_time'], 
                         strides=[1, 3, 3, 1], padding='VALID', 
                         name='conv1_layer_time'), biases['b_conv1_time'], name='bias_add_1_time'),name='relu_add_1_time') 

        self.conv2_time = tf.nn.leaky_relu(tf.nn.bias_add(
            tf.nn.conv2d(self.conv1_time, weights['w_conv2_time'],
                         strides=[1, 2, 2, 1], padding='VALID', 
                         name='conv2_layer_time'), biases['b_conv2_time'], name='bias_add_2_time'),name='relu_add_2_time')

        self.conv3_time = tf.nn.leaky_relu(tf.nn.bias_add(
            tf.nn.conv2d(self.conv2_time, weights['w_conv3_time'],
                         strides=[1, 1, 1, 1], padding='VALID', 
                         name='conv3_layer_time'), biases['b_conv3_time'], name='bias_add_3_time'),name='relu_add_3_time')

        self.conv4_time = tf.nn.leaky_relu(tf.nn.bias_add(
            tf.nn.conv2d(self.conv3_time, weights['w_conv4_time'], 
                         strides=[1, 1, 1, 1], padding='VALID',
                         name='conv4_layer_time'), biases['b_conv4_time'], name='bias_add_4_time'),name='relu_add_4_time')
        
        '''       
        self.conv1_time = tf.nn.bias_add(
            tf.nn.conv2d(self.input_3d_time, weights['w_conv1_time'], 
                         strides=[1, 3, 3, 1], padding='VALID', 
                         name='conv1_layer_time'), biases['b_conv1_time'], name='bias_add_1_time')

        self.conv2_time = tf.nn.bias_add(
            tf.nn.conv2d(self.conv1_time, weights['w_conv2_time'],
                         strides=[1, 2, 2, 1], padding='VALID', 
                         name='conv2_layer_time'), biases['b_conv2_time'], name='bias_add_2_time')

        self.conv3_time = tf.nn.bias_add(
            tf.nn.conv2d(self.conv2_time, weights['w_conv3_time'],
                         strides=[1, 1, 1, 1], padding='VALID', 
                         name='conv3_layer_time'), biases['b_conv3_time'], name='bias_add_3_time')

        self.conv4_time = tf.nn.bias_add(
            tf.nn.conv2d(self.conv3_time, weights['w_conv4_time'], 
                         strides=[1, 1, 1, 1], padding='VALID',
                         name='conv4_layer_time'), biases['b_conv4_time'], name='bias_add_4_time')
        
        #'''
        self.trainLength_time = tf.placeholder(shape=[],dtype=tf.int32,name='time_trainLength')

        self.batch_size_time = tf.placeholder(shape=[],dtype=tf.int32,name='time_batch_size') 

        self.convFlat_time = tf.reshape(tf.contrib.layers.flatten(self.conv4_time),
                                        [self.batch_size_time,self.trainLength_time,h_size], name='time_convFlat_reshape')
 
        self.state_in_time = rnn_cell_time.zero_state(self.batch_size_time, tf.float32)
        
        self.rnn_time,self.rnn_state_time = tf.nn.dynamic_rnn(
            inputs=self.convFlat_time,cell=rnn_cell_time,dtype=tf.float32,
            initial_state=self.state_in_time,scope='time_rnn')
        
        self.rnn_time = tf.reshape(self.rnn_time,shape=[-1,h_size], name='time_rnn_reshape')
        
        self.predict_time = tf.matmul(self.rnn_time, weights['w_predict_time'], name='w_predict_layer_time')
        
        self.target_time = tf.placeholder(dtype=tf.float32,name='time_target')
        
        self.stab_const_time_1 = tf.constant(1e-10, dtype=tf.float32, name='time_st_const_1') 
   
        self.error_time = tf.sqrt(tf.square(self.target_time - self.predict_time, name='error_square_time')
                                     + self.stab_const_time_1, name='error_time')
        
        self.maskA_time = tf.zeros([self.batch_size_time,self.trainLength_time//2], name='maskA_time')
        self.maskB_time = tf.ones([self.batch_size_time,self.trainLength_time//2], name='maskB_time')
        self.mask_time = tf.concat([self.maskA_time,self.maskB_time],1, name='mask_time')
        self.mask_time = tf.reshape(self.mask_time,[-1], name='mask_time_final')
        self.loss_time = tf.reduce_mean(self.error_time * self.mask_time, name='loss_time')
         
        self.global_step_time = tf.Variable(0, trainable=False, name='global_step_time') 
        
        self.learning_rate_time = tf.train.exponential_decay(
                          learning_rate=0.0001,                 
                          global_step=self.global_step_time,   
                          decay_steps=10,                      
                          decay_rate=0.9,                     
                          staircase=True,                      
                          name='learning_rate_time')           

        #self.trainer_time = tf.train.AdamOptimizer(learning_rate=self.learning_rate_time, name='trainer_time')
        self.trainer_time = tf.train.AdamOptimizer(learning_rate=0.0000017, name='trainer_time') # epsilon=1e-08,
        
        self.updateModel_time = self.trainer_time.minimize(
            self.loss_time, global_step=self.global_step_time,name='updateModel_time')
        
#      
class experience_buffer():
    def __init__(self, lock, buffer_size = 100, save_counter = 0):
        self.lock = lock
        self.buffer = []
        self.buffer_size = buffer_size
        self.save_counter = save_counter
        self.load_Buffer_data = None
    
    def add(self,experience):
        self.save_counter += 1
         
        if len(self.buffer) + 1 >= self.buffer_size:
            self.buffer[0:(1+len(self.buffer))-self.buffer_size] = []
        self.buffer.append(experience)
        
        if self.save_counter % self.buffer_size == 0 and self.save_counter != 0:
            
            self.save_counter = 80
            with self.lock:
                np.save('last_buffer/last_buffer', self.buffer) 

    def load_Buffer(self):
        with self.lock:
            self.load_Buffer_data = np.load('last_buffer/last_buffer_140_10_180.npy', allow_pickle=True)
        
        print('Data uploaded successfully')
        
    def sample_offline(self,batch_size,trace_length):
        
        sampled_episodes = random.sample(list(self.load_Buffer_data),batch_size)

        sampledTraces = []
        for episode in sampled_episodes:
            point = np.random.randint(0,len(episode)+1-trace_length)
            sampledTraces.append(episode[point:point+trace_length])
        sampledTraces = np.array(sampledTraces)
        
        
        return np.reshape(sampledTraces,[batch_size*trace_length,3]) # 2
 
    def sample(self,batch_size,trace_length, online_data = True):
        if online_data == True:
            sampled_episodes = random.sample(list(self.buffer),batch_size)
        else:
            with self.lock:
                sampled_episodes = random.sample(list(np.load('last_buffer/last_buffer.npy', 
                allow_pickle=True)),batch_size)
  
        sampledTraces = []
        for episode in sampled_episodes:
            point = np.random.randint(0,len(episode)+1-trace_length)
            sampledTraces.append(episode[point:point+trace_length])
        sampledTraces = np.array(sampledTraces)
        return np.reshape(sampledTraces,[batch_size*trace_length,3])  # 2



# conversion to the required format
def price_format_conversion(digital_value, rounding_threshold):
    """
    example:
    >>> price_format_conversion(0.000000082725, 9)
    '0.000000082'
    """
    if rounding_threshold <= 0:
        return str(int(digital_value))
    string_value = str(float(digital_value))
    if 'e' in string_value:
        string_value_1 = string_value.split('e-')
        string_value_2 = ''
        for i in range(int(string_value_1[1]) - 1):
            if i == '.':
                continue
            string_value_2 = string_value_2 + '0'
        string_value_3 = ''
        for i in  string_value_1[0]:
            if i == '.':
                continue
            string_value_3 = string_value_3 + i
        string_value = '0.' + string_value_2 + string_value_3
        return string_value[:rounding_threshold+2]
    else:
        string_value = string_value.split('.')
        return string_value[0] + '.' + string_value[1][:rounding_threshold]



# The function of calculating the coefficients for taking a specific proportion of the weights
def coeff_for_share_weights(rmse_agents):
    # More social option, tries to take into account the experience of all agents, protection from zero
    if len(rmse_agents) > 1 and sum([0 if val < 1 else 1 for val in rmse_agents]) == 0:
        reverse_rmse = [sum(rmse_agents) - rmse for rmse in rmse_agents]
        koef_agents = [reverse / sum(reverse_rmse) for reverse in reverse_rmse]
    
    # A rougher option that strongly believes in the best and strongly discards the weak, 
    # but only if the agents' error is greater than> 1
    else:
        new_rmse_exp = [np.exp(-power) for power in rmse_agents]
        koef_agents = [new_rmse / sum(new_rmse_exp) for new_rmse in new_rmse_exp]

    if sum(koef_agents) != 1:
        diff = 1 - sum(koef_agents)
        rand_index = np.random.choice(len(koef_agents))
        koef_agents[rand_index] = koef_agents[rand_index] + diff

    return koef_agents


def genetic_modification(lock):
    
    with lock:
        count_arrays =  sum(['array_weights_biases' in file for file in os.listdir('weights_biases_numpy_arrays')]) 

    rmse_end_episode_list = []
    for i in range(count_arrays):
        with lock:
            rmse_end_episode_array = np.load('weights_biases_numpy_arrays/rmse_end_episode_agent_'+str(i)+'.npy',
                                            allow_pickle=True)
        rmse_end_episode_list.append(rmse_end_episode_array)

    discount_coeff = coeff_for_share_weights(rmse_end_episode_list)

    ### main ###
    w_conv1_np = 0
    w_conv2_np = 0
    w_conv3_np = 0
    w_conv4_np = 0
    w_predict_np = 0
    LSTM_weights_np = 0

    b_conv1_np = 0
    b_conv2_np = 0
    b_conv3_np = 0
    b_conv4_np = 0
    LSTM_biases_np = 0

    ### time ###
    w_conv1_np_time = 0
    w_conv2_np_time = 0
    w_conv3_np_time = 0
    w_conv4_np_time = 0
    w_predict_np_time = 0
    LSTM_weights_np_time = 0

    b_conv1_np_time = 0
    b_conv2_np_time = 0
    b_conv3_np_time = 0
    b_conv4_np_time = 0
    LSTM_biases_np_time = 0

    with lock:
        max_time_wind_sec = np.load('weights_biases_numpy_arrays/max_time_window_sec.npy', allow_pickle=True)

    for i in range(count_arrays):
        
        with lock:
            weights_biases_array = np.load('weights_biases_numpy_arrays/array_weights_biases_'+str(i)
                                        +'_agent_max_time_'+str(max_time_wind_sec)+'.npy', 
                                        allow_pickle=True)
        
        ### main ###
        w_conv1_np += weights_biases_array[0][0] * discount_coeff[i]
        w_conv2_np += weights_biases_array[0][1] * discount_coeff[i]
        w_conv3_np += weights_biases_array[0][2] * discount_coeff[i]
        w_conv4_np += weights_biases_array[0][3] * discount_coeff[i]
        w_predict_np += weights_biases_array[0][4] * discount_coeff[i]
        LSTM_weights_np += weights_biases_array[1][0] * discount_coeff[i]
        
        b_conv1_np += weights_biases_array[0][5] * discount_coeff[i]
        b_conv2_np += weights_biases_array[0][6] * discount_coeff[i]
        b_conv3_np += weights_biases_array[0][7] * discount_coeff[i]
        b_conv4_np += weights_biases_array[0][8] * discount_coeff[i]
        LSTM_biases_np += weights_biases_array[1][1] * discount_coeff[i]
        
        ### time ###
        
        w_conv1_np_time += weights_biases_array[0][9] * discount_coeff[i]
        w_conv2_np_time += weights_biases_array[0][10] * discount_coeff[i]
        w_conv3_np_time += weights_biases_array[0][11] * discount_coeff[i]
        w_conv4_np_time += weights_biases_array[0][12] * discount_coeff[i]
        w_predict_np_time += weights_biases_array[0][13] * discount_coeff[i]
        LSTM_weights_np_time += weights_biases_array[2][0] * discount_coeff[i]

        b_conv1_np_time += weights_biases_array[0][14] * discount_coeff[i]
        b_conv2_np_time += weights_biases_array[0][15] * discount_coeff[i]
        b_conv3_np_time += weights_biases_array[0][16] * discount_coeff[i]
        b_conv4_np_time += weights_biases_array[0][17] * discount_coeff[i]
        LSTM_biases_np_time += weights_biases_array[2][1] * discount_coeff[i]
        
    final_weights_biases_arrays = [w_conv1_np, w_conv2_np, w_conv3_np, w_conv4_np, w_predict_np, b_conv1_np, 
                                b_conv2_np, b_conv3_np, b_conv4_np, w_conv1_np_time, w_conv2_np_time, 
                                w_conv3_np_time, w_conv4_np_time, w_predict_np_time, b_conv1_np_time, 
                                b_conv2_np_time, b_conv3_np_time, b_conv4_np_time, 
                                [LSTM_weights_np, LSTM_biases_np], [LSTM_weights_np_time, LSTM_biases_np_time]]

    list_names_params_save = ['w_conv1', 'w_conv2', 'w_conv3', 'w_conv4', 'w_predict', 'b_conv1', 'b_conv2', 
                            'b_conv3', 'b_conv4', 'w_conv1_time', 'w_conv2_time', 'w_conv3_time', 
                            'w_conv4_time', 'w_predict_time','b_conv1_time', 'b_conv2_time', 'b_conv3_time', 
                            'b_conv4_time', 'LSTM_weights_biases', 'LSTM_weights_biases_time']

    for i, name_param in enumerate(list_names_params_save):
        with lock:
            np.save('weights_biases_numpy_arrays/final_weights_biases/'+name_param, 
                    final_weights_biases_arrays[i])

# FINAL VERSION OF THE METRIC
def final_metric(list_price_in_state, list_predicts, risk, commission):

    risk_level = 1 - risk

    true = [list_price_in_state[index+1] / price for index, price in enumerate(\
                    list_price_in_state) if index < len(list_price_in_state)-1]
    
    pred = list_predicts[:len(list_price_in_state[1:])]
    
    error = np.sqrt(mse(true, pred))
    true_trend_with_error = []
    
    for index in range(len(true)):
        if true[index] > (1 + commission):
            true_trend_with_error.append((pred[index] - error) > 1)
        elif true[index] < (1 - commission):
            true_trend_with_error.append((pred[index] + error) < 1)

        elif round(abs(true[index] - 1), 10) <= commission:
            true_trend_with_error.append(True)
        
    model_quality_with_error = sum(true_trend_with_error) / len(true)

    true_trend_not_error = [int(true[index]) == int(pred[index]) for index in range(len(true))]
    model_quality_not_error = sum(true_trend_not_error) / len(true)

    average_model_quality = (model_quality_with_error + model_quality_not_error) / 2
    
    #print(f'true: {true}')
    #print(f'model_quality_with_error = {model_quality_with_error}')
    #print(f'model_quality_not_error = {model_quality_not_error}')
    #print(f'RMSE = {error}')
    #print(f'Trend with error: {true_trend_with_error}')
    #print(f'Trend not error: {true_trend_not_error}')
    #print(f'Average model quality = {average_model_quality}')

    if average_model_quality >= risk_level and error <= (np.std(true) * 2):
        signal = True
        return signal, error
    else:
        signal = False
        return signal, error


def primary_initialization_window_time_rmse(seconds, number_workers):

    np.save('weights_biases_numpy_arrays/max_time_window_sec', np.array(seconds))

    error = 10000
    for number in range(number_workers):
        np.save('weights_biases_numpy_arrays/rmse_end_episode_agent_'+str(number), np.array(error))


# Window time update function (FULL AUTONOMY)
class Window_time_update():
    def __init__(self, lock, number_workers, length_signals_errors = 5): 
        self.lock = lock
        self.length_signals_errors = length_signals_errors
        self.count_not_decrease = 0
        self.first_error = np.inf 
        self.signals = []
        self.condition_error = 'error decreases'
        self.condition_trend = 'trend well'
        self.one_time_pass = True
        self.stop_train_trade = False
        self.number_workers = number_workers
        with self.lock:
            np.save('weights_biases_numpy_arrays/stop_train_trade', np.array(self.stop_train_trade))
    
    def time_update(self, signal):
        
        with self.lock:
            
            rmse_list = []
            for i in range(self.number_workers):
                rmse_array = np.load('weights_biases_numpy_arrays/rmse_end_episode_agent_'+str(i)+'.npy',
                                                    allow_pickle=True)
                rmse_list.append(rmse_array)

        mean_error = sum(rmse_list) / len(rmse_list)
        
        ############# ERRORS ##############
        if mean_error < self.first_error:
            self.count_not_decrease = 0
            self.first_error = mean_error 

        elif mean_error >= self.first_error:
            self.count_not_decrease += 1
            if self.count_not_decrease >= self.length_signals_errors:
                self.condition_error = 'error not decrease'
                
        ############# SIGNALS ##############
        if len(self.signals) >= self.length_signals_errors:
            self.signals[0:1] = []
            self.signals.append(signal)
        
        else:
            self.signals.append(signal)

        ###########################    
        if True in self.signals:
            self.condition_trend = 'trend well'
            
        elif True not in self.signals:
            self.condition_trend = 'trend badly'
        
        if self.condition_trend == 'trend badly' and self.condition_error == 'error not decrease':
            
            if self.one_time_pass:
                
                self.one_time_pass = False

                self.stop_train_trade = True
                with self.lock:
                    max_time_wind_sec = np.load('weights_biases_numpy_arrays/max_time_window_sec.npy', allow_pickle=True)
                   
                    max_time_wind_sec += 60
                    
                    np.save('weights_biases_numpy_arrays/max_time_window_sec', np.array(max_time_wind_sec))
                    
                    np.save('weights_biases_numpy_arrays/stop_train_trade', np.array(self.stop_train_trade))
