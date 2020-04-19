from decomposition_distribution import *
import pickle

file_in = open('pickle_in.p', 'rb')
in_ = pickle.load(file_in)
     
fun_vals = refinement( in_['block.utry'], in_['block.num_qubits'], in_['gate_size'],
in_['fun_vals'], in_['loc_fixed'],
in_['params["refinement_distance"]'],
in_['params["refinement_learning_rate"]'])


print("Our result:")
print(fun_vals)


file_out = open('pickle_out.p', 'rb')
out_ = pickle.load(file_out)

for key in out_:
    print("Given result:")
    print(out_[key])
