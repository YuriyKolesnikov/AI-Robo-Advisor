# Launch of an autonomous robo-advisor
from train import train_agent
from trade import trede_agent
from multiprocessing import Process
from multiprocessing import Lock
from utils import primary_initialization_window_time_rmse

individual_agent_number_0 = 0
individual_agent_number_1 = 1
# Number of workers/agents individual_agent_number_
number_workers = 2
# Initialize priority time window of seconds and the initial error
primary_window_time = 90  
primary_initialization_window_time_rmse(seconds=primary_window_time, number_workers=number_workers)

lock = Lock() 

if __name__ == '__main__':
  
    training_process_1 = Process(target=train_agent, args=(lock, individual_agent_number_0))
    training_process_2 = Process(target=train_agent, args=(lock, individual_agent_number_1))
    
    trading_process = Process(target=trede_agent, args=(lock,number_workers))
    #########################
    training_process_1.start()
    training_process_2.start()

    trading_process.start()
    #########################
    training_process_1.join()
    training_process_2.join()

    trading_process.join()
