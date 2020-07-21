from utils import mainNetwork, timeNetwork, weight_variable, bias_variable
from utils import experience_buffer, state_environment_3D, last_price
from sklearn.metrics import mean_squared_error as mse
import numpy as np
import tensorflow as tf 
import warnings
warnings.filterwarnings("ignore")
import time 
import random

def train_agent(lock, individual_agent_number):

    # ENDLESS CYCLE for continuous exploration of window time, learning and trading
    while True:
    #for _ in range(2):

        ######### PARAMS ########
        # global_episodes THIS IS AN EVOLUTIONARY STRATEGY
        global_episodes = 10000 
        individual_agent_number = individual_agent_number  
        number_training_iterations = 25 
        h_size = 512 
        size_batch = 4 
        trace_length = 4 
        num_episodes = 10 
        max_epLength = 10 
        min_number_batches = 50 
        with lock:
            max_time_window_sec = np.load('weights_biases_numpy_arrays/max_time_window_sec.npy', allow_pickle=True) 
        sleep_time_start_interval = 3 
        sleep_time_end_interval = 6 
        # False == EVOLUTIONARY STRATEGY
        # True == INITIALIZATION FROM ZERO
        first_generation = False  
        list_names_params = ['w_conv1:0', 'w_conv2:0', 'w_conv3:0', 'w_conv4:0', 'w_predict:0', 'b_conv1:0', 'b_conv2:0',
                            'b_conv3:0', 'b_conv4:0', 'w_conv1_time:0', 'w_conv2_time:0', 'w_conv3_time:0',
                            'w_conv4_time:0', 'w_predict_time:0', 'b_conv1_time:0', 'b_conv2_time:0', 'b_conv3_time:0', 
                            'b_conv4_time:0']
        PATH_final_weights_LSTM = 'weights_biases_numpy_arrays/final_weights_biases/'

        rmse = lambda y_true, y_pred: np.sqrt(mse(y_true, y_pred))

        Previous_price_global = last_price()

        ############## START ############

        list_states_exchange = []
        list_steps_time_accumulator = []
        list_absolute_difference = []
        list_next_prices = []
        list_targets = []
        steps_time_accumulator = 0 

        myBuffer = experience_buffer(lock)

        buffer_total_steps = 0

        state_exchange, _ = state_environment_3D() # Execution time: 7 sec ~ 11 sec best_ask_bid_volume_price

        price_in_state = last_price() # Execution time: 0.7 sec

        Previous_price_global = price_in_state
        
        for num_batch in range(min_number_batches):  
            episodeBuffer = []

            j = 0
            while j < max_epLength: 
                
                buffer_total_steps += 1

                if j > 0 or num_batch > 0:
                    
                    steps_time_accumulator = list_steps_time_accumulator[-1] - list_steps_time_accumulator[index_max_val]

                    list_steps_time_accumulator = [t_step - list_steps_time_accumulator[
                        index_max_val] for t_step in list_steps_time_accumulator[index_max_val+1:]]
                    
                    if steps_time_accumulator > max_time_window_sec:
                        steps_time_accumulator = max_time_window_sec - 1

                while steps_time_accumulator < max_time_window_sec:

                    sleep_time_step = np.random.randint(sleep_time_start_interval, sleep_time_end_interval) # (3, 6)

                    time.sleep(sleep_time_step)

                    local_start_time = time.time()

                    local_state_exchange, _ = state_environment_3D() # Execution time: 9 sec ~ 13 sec

                    next_price = last_price()
                    Previous_price_global = next_price

                    local_work_time = time.time() - local_start_time
   
                    list_states_exchange.append(local_state_exchange)
                    list_next_prices.append(next_price)

                    target_change_price = next_price / price_in_state
                    list_targets.append(target_change_price)

                    steps_time_accumulator += (local_work_time + sleep_time_step) # от 12 до 18 секунд
                    list_steps_time_accumulator.append(steps_time_accumulator)

                    absolute_difference = abs(target_change_price - 1)
                    list_absolute_difference.append(absolute_difference)

                    if steps_time_accumulator > max_time_window_sec:

                        index_max_val = list_absolute_difference.index(max(list_absolute_difference))

                        best_target_change_price = list_targets[index_max_val]

                        best_step_time_accumulator = list_steps_time_accumulator[index_max_val] / max_time_window_sec

                        episodeBuffer.append(np.reshape(np.array([state_exchange, 
                                                        best_target_change_price, best_step_time_accumulator]),[1,3]))

                        price_in_state = list_next_prices[index_max_val]

                        list_targets = [n_price / price_in_state for n_price in list_next_prices[index_max_val+1:]]
                        list_absolute_difference = [abs(change_price - 1) for change_price in list_targets]
                        
                        list_next_prices = list_next_prices[index_max_val+1:]
                        
                        state_exchange = list_states_exchange[index_max_val]
                        list_states_exchange = list_states_exchange[index_max_val+1:]

                j += 1        

            bufferArray = np.array(episodeBuffer)
            episodeBuffer = list(zip(bufferArray))
            myBuffer.add(episodeBuffer) 
        

        for _ in range(global_episodes):

            tf.reset_default_graph()

            if first_generation == True:
             
                weights = {                             ### mine ###
                    # 5x5 filter_size, 3 channel, 32 num_filters
                    'w_conv1': weight_variable(lock, shape=[5, 5, 3, 32], name='w_conv1', first_generation = True),
                    # 4x4 filter_size, 32 channel, 64 num_filters
                    'w_conv2': weight_variable(lock, shape=[4, 4, 32, 64], name='w_conv2', first_generation = True),
                    # 3x3 filter_size, 32 channel, 64 num_filters
                    'w_conv3': weight_variable(lock, shape=[3, 3, 64, 64], name='w_conv3', first_generation = True),
                    # 3x3 filter_size, 32 channel, 64 num_filters
                    'w_conv4': weight_variable(lock, shape=[5, 5, 64, 512], name='w_conv4', first_generation = True),
                    # fully connected, 512 inputs, 1 outputs
                    'w_predict': weight_variable(lock, shape=[512, 1], name='w_predict', first_generation = True),
                                                            
                                                        ### time ### 
                    
                    # 5x5 filter_size, 3 channel, 32 num_filters
                    'w_conv1_time': weight_variable(lock, shape=[5, 5, 3, 32], name='w_conv1_time', first_generation = True),
                    # 4x4 filter_size, 32 channel, 64 num_filters
                    'w_conv2_time': weight_variable(lock, shape=[4, 4, 32, 64], name='w_conv2_time', first_generation = True),
                    # 3x3 filter_size, 32 channel, 64 num_filters
                    'w_conv3_time': weight_variable(lock, shape=[3, 3, 64, 64], name='w_conv3_time', first_generation = True),
                    # 3x3 filter_size, 32 channel, 64 num_filters
                    'w_conv4_time': weight_variable(lock, shape=[5, 5, 64, 512], name='w_conv4_time', first_generation = True),
                    # fully connected, 512 inputs, 1 outputs
                    'w_predict_time': weight_variable(lock, shape=[512, 1], name='w_predict_time', first_generation = True)
                }

                biases = {                              ### mine ###

                    'b_conv1': bias_variable(lock, shape=[32], name='b_conv1', first_generation = True),
                    'b_conv2': bias_variable(lock, shape=[64], name='b_conv2', first_generation = True),
                    'b_conv3': bias_variable(lock, shape=[64], name='b_conv3', first_generation = True),
                    'b_conv4': bias_variable(lock, shape=[512], name='b_conv4', first_generation = True),
                    
                                                        ### time ###
                    'b_conv1_time': bias_variable(lock, shape=[32], name='b_conv1_time', first_generation = True),
                    'b_conv2_time': bias_variable(lock, shape=[64], name='b_conv2_time', first_generation = True),
                    'b_conv3_time': bias_variable(lock, shape=[64], name='b_conv3_time', first_generation = True),
                    'b_conv4_time': bias_variable(lock, shape=[512], name='b_conv4_time', first_generation = True)
                }
                
            if first_generation == False:

                weights = {                              ### mine ###
                    # 5x5 filter_size, 3 channel, 32 num_filters
                    'w_conv1': weight_variable(lock, shape=[5, 5, 3, 32], name='w_conv1', first_generation = False),
                    # 4x4 filter_size, 32 channel, 64 num_filters
                    'w_conv2': weight_variable(lock, shape=[4, 4, 32, 64], name='w_conv2', first_generation = False),
                    # 3x3 filter_size, 32 channel, 64 num_filters
                    'w_conv3': weight_variable(lock, shape=[3, 3, 64, 64], name='w_conv3', first_generation = False),
                    # 3x3 filter_size, 32 channel, 64 num_filters
                    'w_conv4': weight_variable(lock, shape=[5, 5, 64, 512], name='w_conv4', first_generation = False),
                    # fully connected, 512 inputs, 1 outputs
                    'w_predict': weight_variable(lock, shape=[512, 1], name='w_predict', first_generation = False),
                    
                                                        ### time ###
                    
                    # 5x5 filter_size, 3 channel, 32 num_filters
                    'w_conv1_time': weight_variable(lock, shape=[5, 5, 3, 32], name='w_conv1_time', first_generation = False),
                    # 4x4 filter_size, 32 channel, 64 num_filters
                    'w_conv2_time': weight_variable(lock, shape=[4, 4, 32, 64], name='w_conv2_time', first_generation = False),
                    # 3x3 filter_size, 32 channel, 64 num_filters
                    'w_conv3_time': weight_variable(lock, shape=[3, 3, 64, 64], name='w_conv3_time', first_generation = False),
                    # 3x3 filter_size, 32 channel, 64 num_filters
                    'w_conv4_time': weight_variable(lock, shape=[5, 5, 64, 512], name='w_conv4_time', first_generation = False),
                    # fully connected, 512 inputs, 1 outputs
                    'w_predict_time': weight_variable(lock, shape=[512, 1], name='w_predict_time', first_generation = False)
                }

                biases = {                              ### mine ###

                    'b_conv1': bias_variable(lock, shape=[32], name='b_conv1', first_generation = False),
                    'b_conv2': bias_variable(lock, shape=[64], name='b_conv2', first_generation = False),
                    'b_conv3': bias_variable(lock, shape=[64], name='b_conv3', first_generation = False),
                    'b_conv4': bias_variable(lock, shape=[512], name='b_conv4', first_generation = False),
                    
                                                        ### time ###
                    
                    'b_conv1_time': bias_variable(lock, shape=[32], name='b_conv1_time', first_generation = False),
                    'b_conv2_time': bias_variable(lock, shape=[64], name='b_conv2_time', first_generation = False),
                    'b_conv3_time': bias_variable(lock, shape=[64], name='b_conv3_time', first_generation = False),
                    'b_conv4_time': bias_variable(lock, shape=[512], name='b_conv4_time', first_generation = False)
                }

            cell_mainN = tf.contrib.rnn.BasicLSTMCell(num_units=h_size, state_is_tuple=True)

            cell_timeN = tf.contrib.rnn.BasicLSTMCell(num_units=h_size, state_is_tuple=True)

            mainN = mainNetwork(h_size, cell_mainN, weights, biases)

            timeN = timeNetwork(h_size, cell_timeN, weights, biases)

            init = tf.global_variables_initializer()

            with tf.Session() as sess:
                
                sess.run(init)

                if first_generation == False:
                    
                    ### mainN ###
                    with lock:
                        weights_np_LSTM_mainQN = np.load(PATH_final_weights_LSTM+'LSTM_weights_biases.npy', 
                        allow_pickle=True)
                    
                    cell_mainN.set_weights(weights_np_LSTM_mainQN)
                    
                    ### timeN ###
                    with lock:
                        weights_np_LSTM_timeQN = np.load(PATH_final_weights_LSTM+'LSTM_weights_biases_time.npy', 
                        allow_pickle=True)
                    
                    cell_timeN.set_weights(weights_np_LSTM_timeQN)

                for _ in range(num_episodes): 

                    episodeBuffer = []
                   
                    j = 0
                    while j < max_epLength: 
                      
                        buffer_total_steps += 1
                        
                        steps_time_accumulator = list_steps_time_accumulator[-1] - list_steps_time_accumulator[index_max_val]
                       
                        list_steps_time_accumulator = [t_step - list_steps_time_accumulator[
                            index_max_val] for t_step in list_steps_time_accumulator[index_max_val+1:]]
                        
                        if steps_time_accumulator > max_time_window_sec:
                            steps_time_accumulator = max_time_window_sec - 1
                        
                        while steps_time_accumulator < max_time_window_sec:

                            sleep_time_step = np.random.randint(sleep_time_start_interval, sleep_time_end_interval) # (3, 6)

                            time.sleep(sleep_time_step)

                            local_start_time = time.time()

                            local_state_exchange, _ = state_environment_3D() # Execution time: 9 sec ~ 13 sec
         
                            next_price = last_price()
                            Previous_price_global = next_price
                            
                            local_work_time = time.time() - local_start_time
                            
                            list_states_exchange.append(local_state_exchange)
                            list_next_prices.append(next_price)
                            
                            target_change_price = next_price / price_in_state
                            list_targets.append(target_change_price)
                            
                            steps_time_accumulator += (local_work_time + sleep_time_step) 
                            list_steps_time_accumulator.append(steps_time_accumulator)

                            absolute_difference = abs(target_change_price - 1)
                            list_absolute_difference.append(absolute_difference)
                            
                            if steps_time_accumulator > max_time_window_sec:
                               
                                index_max_val = list_absolute_difference.index(max(list_absolute_difference))

                                best_target_change_price = list_targets[index_max_val]
                       
                                best_step_time_accumulator = list_steps_time_accumulator[index_max_val] / max_time_window_sec
                                
                                episodeBuffer.append(np.reshape(np.array([state_exchange, 
                                                                best_target_change_price, best_step_time_accumulator]),[1,3]))
                                
                                price_in_state = list_next_prices[index_max_val]
             
                                list_targets = [n_price / price_in_state for n_price in list_next_prices[index_max_val+1:]]
                                list_absolute_difference = [abs(change_price - 1) for change_price in list_targets]
                                
                                list_next_prices = list_next_prices[index_max_val+1:]
                                
                                state_exchange = list_states_exchange[index_max_val]
                                list_states_exchange = list_states_exchange[index_max_val+1:]

                        j += 1

                        for _ in range(number_training_iterations):
                          
                            state_train = (np.zeros([size_batch,h_size]),np.zeros([size_batch,h_size])) 
                            
                            state_train_time = (np.zeros([size_batch,h_size]),np.zeros([size_batch,h_size]))

                            trainBatch = myBuffer.sample(size_batch,trace_length) 
                            
                            target = trainBatch[:,1]
                            target_time = trainBatch[:,2]

                            _0, _1 = sess.run([mainN.updateModel, timeN.updateModel_time],
                                                                        feed_dict={
                                                                            mainN.rawInput:np.vstack(trainBatch[:,0]),
                                                                            timeN.rawInput_time:np.vstack(trainBatch[:,0]),
                                                                            mainN.target:target,
                                                                            timeN.target_time:target_time,
                                                                            mainN.trainLength:trace_length,
                                                                            mainN.state_in:state_train,
                                                                            mainN.batch_size:size_batch,
                                                                            timeN.trainLength_time:trace_length,
                                                                            timeN.state_in_time:state_train_time,
                                                                            timeN.batch_size_time:size_batch})
                            
                    prediction_batch = sess.run(mainN.predict,
                                            feed_dict={
                                                mainN.rawInput:np.vstack(trainBatch[:,0]), 
                                                mainN.trainLength:trace_length,
                                                mainN.state_in:state_train,
                                                mainN.batch_size:size_batch})
                    
                    rmse_end_episode = rmse(target, np.ravel(prediction_batch))

                    ##### mainN ####
                    list_weights_biases = [sess.run(list_names_params)]
                    
                    list_weights_biases.append(cell_mainN.get_weights())

                    ##### timeN ####
                    list_weights_biases.append(cell_timeN.get_weights())

                    with lock:
                        np.save('weights_biases_numpy_arrays/array_weights_biases_'+str(individual_agent_number)
                                +'_agent_max_time_'+str(max_time_window_sec), np.array(list_weights_biases))

                        np.save('weights_biases_numpy_arrays/rmse_end_episode_agent_'+str(individual_agent_number),
                                np.array(rmse_end_episode))
                        
                        if individual_agent_number == 0:
                        
                            np.save('weights_biases_numpy_arrays/max_time_window_sec', np.array(max_time_window_sec))
                
                    bufferArray = np.array(episodeBuffer)
                    episodeBuffer = list(zip(bufferArray))
                    myBuffer.add(episodeBuffer)        
            
            with lock:
                stop_train_trade_signal = np.load('weights_biases_numpy_arrays/stop_train_trade.npy', allow_pickle=True)
            if stop_train_trade_signal:
                break
                
        print('buffer_total_steps =', buffer_total_steps)

