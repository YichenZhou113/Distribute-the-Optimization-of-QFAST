"""
Example showing how to synthesize a 4-qubit QFT program using QFAST.
"""

import numpy as np
import logging
from circuit import *
from instantiation import *
from recombination import *

logging.basicConfig(filename='qfast.log', level=logging.DEBUG)
logging.warning('Start!') #testing
logger = logging.getLogger( "qfast" )

# The QFT unitary matrix.
utry = np.loadtxt('../benchmarks/qft4.unitary', dtype = np.complex128 )

def synthesize ( utry ):
    """
    Synthesize a unitary matrix and return qasm code using QFAST with
    qiskit's kak native tool.

    Args:
        utry (np.ndarray): The unitary matrix to synthesize

    Returns:
        (str): Qasm code implementing utry
    """

    circ = Circuit( utry )
    block_size = get_native_tool( "kak" ).get_native_block_size()

    # Perform decomposition until the circuit is represented in blocks
    # of at most size block_size
    circ.hierarchically_decompose( block_size )

    # Perform instantiation converting blocks to small qasm circuits
    qasm_list = [ instantiation( "kak", block.utry ) for block in circ.blocks ]
    locations = circ.get_locations()

    # Recombine all small circuits into a large one
    out_qasm = recombination( qasm_list, locations )

    return out_qasm


# Synthesize the qft4 unitary and print results
qasm = synthesize( utry )
print( qasm )
