from decomposition_distribution import *
import pickle
import os
import logging
from hostlist import expand_hostlist
import sys
import signal

logging.basicConfig(filename='qfastlog_queue.log', level=logging.DEBUG)
logging.warning('Watch out!') #testing
logger = logging.getLogger( "qfast_queue" )

#task_index  = int( os.environ['SLURM_PROCID'] )
#n_tasks     = int( os.environ['SLURM_NPROCS'] )
#tf_hostlist = [ ("%s:22222" % host) for host in expand_hostlist( os.environ['SLURM_JOB_NODELIST']) ]  
#print("host")
#print(tf_hostlist)

file_in = open('pickle_in.p', 'rb')
in_ = pickle.load(file_in)
     
fun_vals = refinement( in_['block.utry'], in_['block.num_qubits'], in_['gate_size'],
in_['fun_vals'], in_['loc_fixed'],
in_['params["refinement_distance"]'],
in_['params["refinement_learning_rate"]'])


logger.info("Our result:")
logger.info(fun_vals)


file_out = open('pickle_out.p', 'rb')
out_ = pickle.load(file_out)

for key in out_:
    logger.info("Given result:")
    logger.info(out_[key])

logger.info("reach the end")


sys.exit(0)

