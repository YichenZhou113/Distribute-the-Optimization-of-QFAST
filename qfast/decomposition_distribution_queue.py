"""
This module implements the main decomposition functions.
"""

import logging
import tensorflow as tf
import numpy      as np
import itertools  as it

from fixedgate import FixedGate
from genericgate import GenericGate
from locationmodel import LocationModel
from block import Block
from metrics import hilbert_schmidt_distance
from pauli import get_unitary_from_pauli_coefs, reset_tensor_cache

import pickle
import sys
import os
import signal
from hostlist import expand_hostlist

logger = logging.getLogger( "qfast_queue" )

task_index  = int( os.environ['SLURM_PROCID'] )
logger.info("outside task_index: %f" %task_index)
n_tasks     = int( os.environ['SLURM_NPROCS'] )
logger.info("n_tasks: %f " %n_tasks)
tf_hostlist = [ ("%s:22222" % host) for host in
                expand_hostlist( os.environ['SLURM_NODELIST']) ]
logger.info("tf_hostlist:  ")
logger.info(tf_hostlist)

cluster = tf.train.ClusterSpec( {"localhost" : tf_hostlist } )
#chief_index = 0
chief_index = task_index

server  = tf.train.Server( cluster.as_cluster_def(),
                           job_name   = "localhost",
                           task_index = chief_index )
n_cpus = n_tasks

def decomposition ( block, **kwargs ):
    """
    Decomposes a Block into a sequence of smaller blocks which
    implement the input.

    Args:
        block (Block): Target Block

    Keyword Args:
        start_depth (int): The numbezr of gates to start exploring from

        depth_step (int): The number of added gates in each
                          exploration step

        exploration_distance (float): Exploration's goal distance

        exploration_learning_rate (float): Learning rate of
                                           exploration's optimizer

        refinement_distance (float): Refinement's goal distance

        refinement_learning_rate (float): Learning rate of refinement's
                                          optimizer

        native_block_size (int): Minimum block size

    Returns:
        (List[Block]): Decomposed blocks
    """

    if block.num_qubits <= 2:
        return [ block ]

    params = {}
    params["start_depth" ] = 1
    params["depth_step" ] = 1
    params["exploration_distance" ] = 0.01
    params["exploration_learning_rate" ] = 0.01
    params["refinement_distance" ] = 1e-7
    params["refinement_learning_rate" ] = 1e-6
    params["native_block_size"] = 0
    params.update( kwargs )

    if block.num_qubits <= params["native_block_size"]:
        return [ block ]

    gate_size = get_decomposition_size( block.num_qubits )

    if gate_size < params["native_block_size"]:
        gate_size = params["native_block_size"]

    lm = LocationModel( block.num_qubits, gate_size )

    fun_vals, loc_vals = exploration( block.utry, block.num_qubits, gate_size,
                                      params["start_depth"],
                                      params["depth_step"], lm,
                                      params["exploration_distance"],
                                      params["exploration_learning_rate"] )

    loc_fixed = lm.fix_locations( loc_vals )

    pickle_in = {'block.utry': block.utry, 'block.num_qubits': block.num_qubits, 'gate_size': gate_size, 'fun_vals': fun_vals, 'loc_fixed': loc_fixed, 'params["refinement_distance"]': params["refinement_distance"], 'params["refinement_learning_rate"]': params["refinement_learning_rate"]}

    pickle.dump(pickle_in, open('pickle_in.p', 'wb'))
    fun_vals = refinement( block.utry, block.num_qubits, gate_size,
                           fun_vals, loc_fixed,
                           params["refinement_distance"],
                           params["refinement_learning_rate"] )
    pickle_out = {'fun_vals': fun_vals}
    pickle.dump(pickle_out, open('pickle_out.p', 'wb'))

    return convert_to_block_list( block.get_location(), fun_vals, loc_fixed )

def get_decomposition_size ( num_qubits ):
    """
    Given a block size, computes the block size to decompose to

    Args:
        num_qubits (int): Number of qubits in block

    Returns:
        (int): Decomposed block size
    """

    if not isinstance( num_qubits, int ):
        raise TypeError( "Input must be an integer." )

    if num_qubits <= 0:
        raise ValueError( "Number of qubits must be positive." )

    return int( np.ceil( num_qubits / 2 ) )


def exploration ( target, num_qubits, gate_size, start_depth, depth_step,
                  lm, exploration_distance = 0.01, learning_rate = 0.01 ):
    """
    Synthesizes a circuit that implements the target unitary. Explores
    both circuit structure (gate location) and gate functions.

    Args:
        target (np.ndarray): Target unitary

        num_qubits (int): The target unitary's number of qubits

        gate_size (int): number of active qubits in a gate

        start_depth (int): The number of gates to start searching from

        depth_step (int): The number of added gates in each step

        lm (LocationModel): The model that maps loc_vals to locations

        exploration_distance (float): Exploration's goal distance

        learning_rate (float): Learning rate of the optimizer

    Returns:
        (Tuple[List[List[float]], List[List[float]]]):
            The final fun_vals and loc_vals
    """

    depth = start_depth
    fun_vals = [ None ] * depth
    loc_vals = [ None ] * depth
    logger.info( "Starting to search with %d gates." % depth )

    # Stride search over layer_count
    while ( True ):
        result = fixed_depth_exploration( target, num_qubits, gate_size,
                                          fun_vals, loc_vals, lm,
                                          exploration_distance, learning_rate )

        success, fun_vals, loc_vals = result

        if success:
            logger.info( "Found a circuit with %d gates." % depth )
            break

        fun_vals += [ None ] * depth_step
        loc_vals += [ None ] * depth_step
        depth += depth_step
        logger.info( "Added a gate, now at %d gates." % depth )

    # Remember good results
    found_fun_vals = fun_vals
    found_loc_vals = loc_vals

    # Search backwards for a shorter circuit
    for i in range( depth_step - 1 ):
        depth -= 1
        logger.info( "Removing a layer, now at %d gates." % depth )


        result = fixed_depth_exploration( target, num_qubits, gate_size,
                                          fun_vals, loc_vals, lm,
                                          exploration_distance, learning_rate )

        success, fun_vals, loc_vals = result

        if success:
            found_fun_vals = fun_vals
            found_loc_vals = loc_vals
            logger.info( "Found a circuit with %d gates." % depth )

    return found_fun_vals, found_loc_vals


def fixed_depth_exploration ( target, num_qubits, gate_size, fun_vals,
                              loc_vals, lm, exploration_distance = 0.01,
                              learning_rate = 0.01 ):
    """
    Attempts to synthesize the target unitary with a fixed number
    of gates of size gate_size.

    Args:
        target (np.ndarray): Target unitary

        num_qubits (int): The target unitary's number of qubits

        gate_size (int): number of active qubits in a gate

        fun_vals (List[List[float]]): Gate function values

        loc_vals (List[List[float]]): Gate location values

        lm (LocationModel): The model that maps loc_vals to locations

        exploration_distance (float): Exploration's goal distance

        learning_rate (float): Learning rate of the optimizer

    Returns:
        (Tuple[bool, List[List[float]], List[List[float]]]):
            True if succeeded in hitting exploration distance
            and the final fun_vals and loc_vals
    """

    tf.reset_default_graph()
    reset_tensor_cache()

    layers = [ GenericGate( "Gate%d" % i, num_qubits, gate_size, lm,
                            fun_vals[i], loc_vals[i],
                            parity = (i + 1) % lm.num_buckets )
               for i in range( len( fun_vals ) ) ]

    tensor = layers[0].get_tensor()
    for layer in layers[1:]:
        tensor = tf.matmul( layer.get_tensor(), tensor )

    loss_fn   = hilbert_schmidt_distance( target, tensor )
    optimizer = tf.train.AdamOptimizer( learning_rate )
    train_op  = optimizer.minimize( loss_fn )
    init_op   = tf.global_variables_initializer()

    loss_values = []

    with tf.Session() as sess:
        sess.run( init_op )

        while ( True ):
            for i in range( 20 ):
                loss = sess.run( [ train_op, loss_fn ] )[1]

            if loss < exploration_distance:
                logger.info( "Ending exploration at %f distance." % loss )
                return ( True,
                         [ l.get_fun_vals( sess ) for l in layers ],
                         [ l.get_loc_vals( sess ) for l in layers ] )

            loss_values.append( loss )
            logger.debug( "Loss: %f" % loss )

            if len( loss_values ) > 100:
                min_value = np.min( loss_values[-100:] )
                max_value = np.max( loss_values[-100:] )

                # Plateau Detected
                if max_value - min_value < learning_rate / 10:
                    logger.debug( "Ending exploration at %f distance." % loss )
                    return ( False,
                             [ l.get_fun_vals( sess ) for l in layers ],
                             [ l.get_loc_vals( sess ) for l in layers ] )

            if len( loss_values ) > 500:
                min_value = np.min( loss_values[-500:] )
                max_value = np.max( loss_values[-500:] )

                # Plateau Detected
                if max_value - min_value < learning_rate:
                    logger.debug( "Ending exploration at %f distance." % loss )
                    return ( False,
                             [ l.get_fun_vals( sess ) for l in layers ],
                             [ l.get_loc_vals( sess ) for l in layers ] )

def create_done_queue(i):
    with tf.device("/job:localhost/replica:0/task:%d/device:CPU:0" % (i)):
        return tf.FIFOQueue(5, tf.int32, shared_name="done_queue"+str(i))
        
def refinement ( target, num_qubits, gate_size, fun_vals, loc_fixed,
                 refinement_distance = 1e-7, learning_rate = 1e-6 ):
    """
    Refines synthesized circuit to better implement the target unitary.
    This is achieved by using fixed circuit structure (gate location)
    and a more fine-grained optimizer.

    Args:
        target (np.ndarray): Target unitary

        num_qubits (int): The target unitary's number of qubits

        gate_size (int): number of active qubits in a gate

        fun_vals (List[List[float]]): Gate function values

        loc_fixed (List[Tuple[int]]): Gate locations

        refinement_distance (float): Refinement's goal distance

        learning_rate (float): Learning rate of the optimizer

    Returns:
        (List[List[float]]): Refined gate function values
    """
#    #can read the arguments like this. python3 synthesize_qft4_refinement.py -n 1
#    print("All arguments")
#    print(sys.argv) # synthesize_qft4_refinement.py -n 1
#    print("-n arguments")
#    print(sys.argv[-1]) # 1
##    n_cpus = sys.argv[-1];

    if task_index != 0:
        #1 set up server Done in front
        #2 set up session_config
        logger.info("set up ps session_config task index %f TODO:" %task_index)
        sess = tf.Session(server.target,
            config=tf.ConfigProto(
            device_count={ "CPU": n_cpus },
            inter_op_parallelism_threads=n_cpus,
            intra_op_parallelism_threads=1,
        ))
        logger.info("set up ps session_config task index %f Done:" %task_index)
        '''#3 initialize variable on this shard
        tf.reset_default_graph()
        reset_tensor_cache()
        layers = []
        
        
        with tf.device("/job:localhost/replica:0/task:0/device:CPU:0"):
            layers.append(FixedGate( "Gate%d" % 0, num_qubits, gate_size, loc = loc_fixed[0], fun_vals = fun_vals[0] ))
            tensor = layers[0].get_tensor()
       
        each_num = np.ceil(len(fun_vals)/n_cpus)
        for i in range(1, len(fun_vals)):
            with tf.device("/job:localhost/replica:0/task:%d/device:CPU:0" % (i // each_num)):
                layers.append( FixedGate( "Gate%d" % i, num_qubits, gate_size, loc = loc_fixed[i], fun_vals = fun_vals[i] ))
                tensor = tf.matmul(layers[-1].get_tensor(), tensor)

       # with tf.device(tf.train.replica_device_setter(ps_tasks = n_cpus)):
        #with tf.device("/job:localhost/replica:0/task:0/device:CPU:0"):
        with tf.device(tf.train.replica_device_setter(worker_device="/job:localhost/replica:0/task:%d/device:CPU:0" % task_index, cluster=cluster)):
            loss_fn   = hilbert_schmidt_distance( target, tensor )
            optimizer = tf.train.AdamOptimizer( learning_rate )
            train_op  = optimizer.minimize( loss_fn )
            init_op   = tf.global_variables_initializer()

        loss_values = []
        #4 sess.run(initizlizer)
        with tf.Session(server.target, config=tf.ConfigProto( device_count={ "CPU": n_cpus }, inter_op_parallelism_threads=n_cpus, intra_op_parallelism_threads=1,)) as sess:
            logger.info("server.target:%s" %server.target)
            sess.run( init_op )
            loss = sess.run( loss_fn )'''
        #5 create done queue
        queue = create_done_queue(task_index)
        
        #6 dequeu all worker
        sess.run(queue.dequeue())
        logger.info("ps %d received done chief" %task_index)
     

    else:
        logger.info("task index %f inside refinement:" %task_index)
        tf.reset_default_graph()
        reset_tensor_cache()
        layers = []
        
        
        with tf.device("/job:localhost/replica:0/task:0/device:CPU:0"):
            layers.append(FixedGate( "Gate%d" % 0, num_qubits, gate_size, loc = loc_fixed[0], fun_vals = fun_vals[0] ))
            tensor = layers[0].get_tensor()
       
        each_num = np.ceil(len(fun_vals)/n_cpus)
        for i in range(1, len(fun_vals)):
            with tf.device("/job:localhost/replica:0/task:%d/device:CPU:0" % (i // each_num)):
                layers.append( FixedGate( "Gate%d" % i, num_qubits, gate_size, loc = loc_fixed[i], fun_vals = fun_vals[i] ))
                tensor = tf.matmul(layers[-1].get_tensor(), tensor)
        
        """
        layers = [ FixedGate( "Gate%d" % i, num_qubits, gate_size,
                              loc = loc_fixed[i], fun_vals = fun_vals[i] )
                   for i in range( len( fun_vals ) ) ]

        tensor = layers[0].get_tensor()
        
        for layer in layers[1:]:
            tensor = tf.matmul( layer.get_tensor(), tensor )
        """

       # with tf.device(tf.train.replica_device_setter(ps_tasks = n_cpus)):
        #with tf.device("/job:localhost/replica:0/task:0/device:CPU:0"):
        with tf.device(tf.train.replica_device_setter(worker_device="/job:localhost/replica:0/task:%d/device:CPU:0" % task_index, cluster=cluster)):
            loss_fn   = hilbert_schmidt_distance( target, tensor )
            optimizer = tf.train.AdamOptimizer( learning_rate )
            train_op  = optimizer.minimize( loss_fn )
            init_op   = tf.global_variables_initializer()

        loss_values = []
        
       # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
       # run_metadata = tf.RunMetadata()
        
        with tf.Session(server.target,
            config=tf.ConfigProto(
            device_count={ "CPU": n_cpus },
            inter_op_parallelism_threads=n_cpus,
            intra_op_parallelism_threads=1,
        )) as sess:
        #with tf.Session() as sess:
            #writer = tf.summary.FileWriter('./graphs', sess.graph) #visualize tensorflow graph
            #sess.run( init_op, options=run_options, run_metadata=run_metadata )
            #loss = sess.run( loss_fn, options=run_options, run_metadata=run_metadata )
            
            logger.info("server.target:%s" %server.target)
            sess.run( init_op )
            loss = sess.run( loss_fn )
            

            logger.info( "Starting refinement at %f distance,task index %f" % (loss,task_index))

            for i in range( 10000 ):
                for j in range( 50 ):
                    loss = sess.run( [ train_op, loss_fn ] )[1]
                    
                    #loss = sess.run( [ train_op, loss_fn ], options=run_options, run_metadata=run_metadata )[1]
                
                if loss < refinement_distance:
                    break

                loss_values.append( loss )
                logger.log( 0, loss )

                if len( loss_values ) > 100:
                    min_value = np.min( loss_values[-100:] )
                    max_value = np.max( loss_values[-100:] )

                    # Plateau Detected
                    if max_value - min_value < 1e-5:
                        break

                if len( loss_values ) > 500:
                    min_value = np.min( loss_values[-500:] )
                    max_value = np.max( loss_values[-500:] )

                    # Plateau Detected
                    if max_value - min_value < 1e-3:
                        break
                        

            logger.info( "Ending refinement at %f distance.task index %f" % (loss,task_index) )

    #        for device in run_metadata.step_stats.dev_stats:
    #            device_name = device.device
    #            print(device.device)
    #            for node in device.node_stats:
    #                print("   ", node.node_name)"
            logger.info("reach the end in refinement")
            result = [ l.get_fun_vals( sess ) for l in layers ]
            q_result = []
            for r in range(n_cpus):
                if r==0: 
                    continue
                else:
                    q_result.append(create_done_queue(r))
            
            for q in q_result:
                sess.run(q.enqueue(1))
            return result


def convert_to_block_list ( block_loc, fun_vals, loc_fixed ):
    """
    Converts the function parameters to unitary matrices and composes
    location into a larger circuit. Returns the resulting block list.

    Args:
        block_loc (Tuple[int]): The location in a larger circuit that
                                this circuit corresponds to.

        fun_vals (List[List[float]]): Gate function values

        locs_fixed (List[Tuple[int]]): Gate locations

    Returns:
        (List[Block]): Resulting block list
    """

    block_list = []

    for loc, fun_params in zip( loc_fixed, fun_vals ):

        # Convert to unitary
        utry = get_unitary_from_pauli_coefs( fun_params )

        # Compose location
        new_loc = tuple( [ block_loc[i] for i in loc ] )

        block_list.append( Block( utry, new_loc ) )

    return block_list

