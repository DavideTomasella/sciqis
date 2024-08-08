from l2_quantumComputer import QState, QGate, QUGate
import numpy as np
import matplotlib.pyplot as plt
import time

# Script for Austin performance analysis
def computer():
    # No longer working after the refactoring
    state_00 = QState.create_state([QState._0()]*10)
    had = QGate.create_N_gate([QGate._hadamard()]+[QGate._n_id()]*9)
    cnot = QGate.create_N_gate([QGate._0x0(), QGate._n_id(), QGate._n_id()]+[QGate._n_id()]*7) + QGate.create_N_gate([QGate._1x1(), QGate._n_id(), QGate._X()]+[QGate._n_id()]*7)
    #cnot2 = QGate.create_gate([QGate._0(), QGate._id(), QGate._id()]+[QGate._id()]*7) + QGate.create_gate([QGate._1(), QGate._id(), QGate._X()]+[QGate._id()]*7)
    cnot*(had*state_00)

def computer2():
    N=10
    new_state_00 = QState.create_state([QState._0()]*N)
    new_had = QGate.hadamard([0,2],n=N)
    new_cnot = QGate.cnot(0,1,n=N)
    s=new_cnot*(new_had*new_state_00)
    print(s)

if __name__ == '__main__':
    for i in range(1):
        #austin -i 100 --pipe C:\Users\davtom\AppData\Local\miniforge3\envs\sciqis\python.exe ".\test_l3.py" >test2.austin
        computer2()
        #print(i)

        

        