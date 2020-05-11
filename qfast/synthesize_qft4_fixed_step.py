from decomposition_distribution import fixed_depth_exploration
import pickle
import logging

logging.basicConfig(filename='qfast_fixed_step_fun.log', level=logging.DEBUG)
logging.warning('Watch out!') #testing
logger = logging.getLogger( "qfast" )

#file_in = open('pickle_in_fixed_step.p', 'rb')
file_in = open('pickle_in_fixed_step_1.p', 'rb')
in_ = pickle.load(file_in)
     
result = fixed_depth_exploration( in_['target'], in_['num_qubits'], in_['gate_size'],
in_['fun_vals'], in_['loc_vals'],
in_['lm'],
in_['exploration_distance'], in_['learning_rate'])

success, fun_vals, loc_vals = result
print("Our result:")
print(fun_vals)

file_out = open('pickle_out_fixed_step.p', 'rb')
out_ = pickle.load(file_out)

print("Given result:")
print(out_['fun_vals'])
