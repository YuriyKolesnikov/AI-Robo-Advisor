from utils import mainNetwork, timeNetwork, weight_variable, bias_variable
from utils import state_environment_3D, last_price, price_format_conversion, Window_time_update
from utils import genetic_modification, final_metric 
from sklearn.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt
import matplotlib.dates as md
import datetime
import numpy as np
import tensorflow as tf 
import warnings
warnings.filterwarnings("ignore")
import time 
from binance.client import Client

def trade_agent(lock, number_workers):

    # ENDLESS CYCLE for continuous exploration of window time, learning and trading
    while True:
    # for _ in range(2):
        
        class_wtupdate = Window_time_update(lock, number_workers)
        rmse = lambda y_true, y_pred: np.sqrt(mse(y_true, y_pred))
        api_key = '......................................................'
        api_secret = '......................................................'
        client = Client(api_key, api_secret)
        risk = 0.2 
        global_episodes = 10000 
        num_episodes = 10 
        h_size = 512

        with lock:
            max_time_window_sec = np.load('weights_biases_numpy_arrays/max_time_window_sec.npy', allow_pickle=True)                        
            
        PATH_final_weights_LSTM = 'weights_biases_numpy_arrays/final_weights_biases/'

        state = (np.zeros([1,h_size]),np.zeros([1,h_size])) 
        state_time = (np.zeros([1,h_size]),np.zeros([1,h_size]))

        commission = 0.001 #  # 0.00075

        list_predicts = []
        list_predicts_time = []
        list_date_time = []

        list_price_in_state = []

        total_steps = 0    
        global_steps = 0

        for _ in range(global_episodes):
            
            genetic_modification(lock)

            tf.reset_default_graph()

            weights = {                              ### mine ###
                # 5x5 filter_size, 3 channel, 32 num_filters
                'w_conv1': weight_variable(lock, shape=[5, 5, 3, 32], name='w_conv1', first_generation = False),
                # 4x4 filter_size, 32 channel, 64 num_filters
                'w_conv2': weight_variable(lock, shape=[4, 4, 32, 64], name='w_conv2', first_generation = False),
                # 3x3 filter_size, 64 channel, 64 num_filters
                'w_conv3': weight_variable(lock, shape=[3, 3, 64, 64], name='w_conv3', first_generation = False),
                # 3x3 filter_size, 64 channel, 512 num_filters
                'w_conv4': weight_variable(lock, shape=[5, 5, 64, 512], name='w_conv4', first_generation = False),
                # fully connected, 512 inputs, 1 outputs
                'w_predict': weight_variable(lock, shape=[512, 1], name='w_predict', first_generation = False),

                                                    ### time ###

                # 5x5 filter_size, 3 channel, 32 num_filters
                'w_conv1_time': weight_variable(lock, shape=[5, 5, 3, 32], name='w_conv1_time', first_generation = False),
                # 4x4 filter_size, 32 channel, 64 num_filters
                'w_conv2_time': weight_variable(lock, shape=[4, 4, 32, 64], name='w_conv2_time', first_generation = False),
                # 3x3 filter_size, 64 channel, 64 num_filters
                'w_conv3_time': weight_variable(lock, shape=[3, 3, 64, 64], name='w_conv3_time', first_generation = False),
                # 3x3 filter_size, 64 channel, 512 num_filters
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

                ### mainN ###
                ### timeN ###
                with lock:
                    weights_np_LSTM_mainQN = np.load(PATH_final_weights_LSTM+'LSTM_weights_biases.npy', allow_pickle=True)
                    weights_np_LSTM_timeQN = np.load(PATH_final_weights_LSTM+'LSTM_weights_biases_time.npy', allow_pickle=True)

                ### mainN ###
                ### timeN ###
                cell_mainN.set_weights(weights_np_LSTM_mainQN)
                cell_timeN.set_weights(weights_np_LSTM_timeQN)

                for _ in range(num_episodes):  

                    total_steps += 1 

                    state_exchange, best_ask_bid_volume_price = state_environment_3D() 
                    price_in_state = last_price() 
                 
                    Previous_price_global = price_in_state
                    
                    if len(list_price_in_state) == num_episodes:
                        list_price_in_state[0:1] = []
                        list_price_in_state.append(price_in_state)
               
                    else:
                        list_price_in_state.append(price_in_state)

                    prediction, state1 = sess.run([mainN.predict,mainN.rnn_state],
                                                feed_dict={
                                                    mainN.rawInput:[state_exchange],
                                                    mainN.trainLength:1,
                                                    mainN.state_in:state,
                                                    mainN.batch_size:1})

                    state = state1

                    predict_change_price = float(prediction)
                    
                    if len(list_predicts) == num_episodes:
                        list_predicts[0:1] = []
                        list_predicts.append(predict_change_price)
                    
                    else:
                        list_predicts.append(predict_change_price)
 
                    prediction_time, state1_time = sess.run([timeN.predict_time ,timeN.rnn_state_time],
                                                feed_dict={
                                                    timeN.rawInput_time:[state_exchange],
                                                    timeN.trainLength_time:1,
                                                    timeN.state_in_time:state_time,
                                                    timeN.batch_size_time:1})

                    state_time = state1_time

                    pred_time = float(prediction_time)
                    
                    if len(list_predicts_time) == num_episodes:
                        list_predicts_time[0:1] = []
                        list_date_time[0:1] = []
                        list_predicts_time.append(pred_time)
                        list_date_time.append(datetime.datetime.now())
                    
                    else:
                        list_predicts_time.append(pred_time)
                        list_date_time.append(datetime.datetime.now())
                    
                    if global_steps > 0:

                        signal_for_trade, error_online = final_metric(list_price_in_state, list_predicts, risk, commission)
                        
                        class_wtupdate.time_update(signal=signal_for_trade)

                        adjusted_predict_change_price = predict_change_price - error_online if \
                            predict_change_price >= 1 else predict_change_price + error_online
                       
                        trend_matching = int(predict_change_price) == int(adjusted_predict_change_price)

                                        ### SIGNAL and TREND ###
                        if signal_for_trade and trend_matching:

                            predict_change_price = adjusted_predict_change_price
                           
                            action_predict = ['sell', 'buy'][int(predict_change_price)]

                            money_predict = float(client.get_asset_balance(asset='USDT').get('free'))

                            bitcoin_predict = float(client.get_asset_balance(asset='BTC').get('free')) * price_in_state
                            
                            if money_predict >= 10 and action_predict == 'buy' and predict_change_price - 1 > commission:
   
                                if best_ask_bid_volume_price.get('volume').get('buy')[0] * price_in_state > money_predict:

                                    quantity_BTC = price_format_conversion(money_predict / price_in_state, 6)

                                    order = client.order_market_buy(symbol = 'BTCUSDT', quantity = quantity_BTC)

                                    quantity_true_action_predict += 1

                                else:
                                  
                                    best_quantity_BTC = str(best_ask_bid_volume_price.get('volume').get('buy')[0]) 

                                    order = client.order_market_buy(symbol = 'BTCUSDT', quantity = best_quantity_BTC)

                                    quantity_true_action_predict += 1

                                    for i in range(1,10):

                                        money_predict = float(client.get_asset_balance(asset='USDT').get('free'))
                                        
                                        if money_predict >= 10: 
                                           
                                            predict_change_price_next = ((price_in_state * predict_change_price
                                                                        ) / best_ask_bid_volume_price.get('price').get('buy')[i])

                                            if predict_change_price_next - 1 > commission:

                                                volume_action_money_predict = (best_ask_bid_volume_price.get('volume').get('buy')[i]
                                                                        * best_ask_bid_volume_price.get('price').get('buy')[i])

                                                if volume_action_money_predict >= money_predict:

                                                    best_quantity_BTC = price_format_conversion(money_predict / price_in_state, 6)

                                                    order = client.order_market_buy(symbol = 'BTCUSDT', quantity = best_quantity_BTC)

                                                    quantity_true_action_predict += 1

                                                else:

                                                    best_quantity_BTC = str(best_ask_bid_volume_price.get('volume').get('buy')[i])

                                                    order = client.order_market_buy(symbol = 'BTCUSDT', quantity = best_quantity_BTC)

                                                    quantity_true_action_predict += 1

                                            else:
                                                
                                                break

                            ############# bitcoin_predict ################

                            if bitcoin_predict >= 10 and action_predict == 'sell' and 1 - predict_change_price > commission:

                                if best_ask_bid_volume_price.get('volume').get('sell')[0] * price_in_state > bitcoin_predict:

                                    quantity_BTC = price_format_conversion(bitcoin_predict / price_in_state, 6)

                                    order = client.order_market_sell(symbol = 'BTCUSDT', quantity = quantity_BTC)

                                    quantity_true_action_predict += 1

                                else:
                                    
                                    best_quantity_BTC = str(best_ask_bid_volume_price.get('volume').get('sell')[0]) 

                                    order = client.order_market_sell(symbol = 'BTCUSDT', quantity = best_quantity_BTC)

                                    quantity_true_action_predict += 1

                                    for i in range(1,10):

                                        bitcoin_predict = float(client.get_asset_balance(asset='BTC').get('free')) * price_in_state
                                      
                                        if bitcoin_predict >= 10:
                               
                                            predict_change_price_next = ((price_in_state * predict_change_price 
                                                                        ) / best_ask_bid_volume_price.get('price').get('sell')[i])

                                            if 1 - predict_change_price_next > commission:

                                                volume_action_bitcoin_predict = (best_ask_bid_volume_price.get('volume').get('sell')[i]
                                                                        * best_ask_bid_volume_price.get('price').get('sell')[i])

                                                if volume_action_bitcoin_predict >= bitcoin_predict:

                                                    best_quantity_BTC = price_format_conversion(bitcoin_predict / price_in_state, 6)

                                                    order = client.order_market_sell(symbol = 'BTCUSDT', quantity = best_quantity_BTC)

                                                    quantity_true_action_predict += 1

                                                else:

                                                    best_quantity_BTC = str(best_ask_bid_volume_price.get('volume').get('sell')[i])

                                                    order = client.order_market_sell(symbol = 'BTCUSDT', quantity = best_quantity_BTC)

                                                    quantity_true_action_predict += 1

                                            else:

                                                break

                        list_pred_abs_price = [list_predicts[index] * price for index, price in enumerate(list_price_in_state)]
                        
                        list_pred_abs_time = [max_time_window_sec * t for t in list_predicts_time]

                        list_date_time_for_pred = list_date_time[:]

                        list_date_time_for_pred.append(list_date_time[-1] + datetime.timedelta(seconds=(list_pred_abs_time[-1])))

                        rmse_online = rmse(list_price_in_state[1:], list_pred_abs_price[:len(list_price_in_state[1:])])

                        plt.ion() 
                        plt.gca().cla() 

                        plt.subplots_adjust(bottom=0.2)
                        plt.xticks(rotation=25)
                        ax=plt.gca()
                        ax.set_xticks(list_date_time_for_pred[1:])

                        xfmt = md.DateFormatter('%H:%M:%S') 
                        ax.xaxis.set_major_formatter(xfmt)

                        plt.plot(list_date_time_for_pred[1:], list_pred_abs_price, linewidth=3, linestyle="--",
                                color="blue", marker='o', label=r"Predicted price")
                        plt.plot(list_date_time[1:], list_price_in_state[1:], linewidth=3, linestyle="-",
                                color="red", marker='o', label=r"True price") 
                        plt.xlabel(r"Agent Predicted Time (seconds)")
                        plt.ylabel(r"Predicted and true price (US$)")
                        plt.title(f'Real time trading. RMSE online = {rmse_online.round(2)} US$. SIGNAL = {signal_for_trade}')

                        plt.legend(loc="upper left") 
                        plt.pause(0.1)
                        plt.show()

                    time.sleep(int(abs(pred_time)) * max_time_window_sec) 
            
            global_steps += 1
            
            with lock:
                stop_train_trade_signal = np.load('weights_biases_numpy_arrays/stop_train_trade.npy', allow_pickle=True)
            if stop_train_trade_signal:
                break
            
        print('total_steps_trade =', total_steps)




