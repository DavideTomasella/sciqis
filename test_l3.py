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
    if False:
        for i in range(1):
            #austin -i 100 --pipe C:\Users\davtom\AppData\Local\miniforge3\envs\sciqis\python.exe ".\test_l3.py" >test2.austin
            computer2()
            #print(i)
    else:
        computer2()

        rng_gen=np.random.default_rng()
        runs=1000
        N_gates=3
        N_pars=3
        N_tests = 5
        rngs=rng_gen.uniform(0,2*np.pi,N_gates*N_pars*runs).reshape(runs,N_pars,N_gates)
        #Init
        init_state=QState.create_state([QState._0()]*N_gates)
        test_state=np.zeros((N_tests,runs),dtype=QState)
        fidelities=np.zeros((N_tests,runs))
        P_haar=lambda F,N: (N-1)*(1-F)**(N-2)
        had=QGate.hadamard(range(N_gates),n=N_gates)
        identity=QGate.create_id(n=N_gates)
        #start timer
        t0=time.time()
        for run in range(runs):
            Ry=QGate.Ry(range(N_gates),rngs[run,0,:],n=N_gates)
            Rx=QGate.Rx(range(N_gates),rngs[run,1,:],n=N_gates)
            Rz=QGate.Rz(range(N_gates),rngs[run,2,:],n=N_gates)

            test_state[0,run]=Ry*had*init_state
            test_state[1,run]=Rx*had*init_state
            test_state[2,run]=Rz*had*init_state
            test_state[3,run]=Rx*Ry*had*init_state
            test_state[4,run]=Rz*Rx*Ry*had*init_state
            for i in range(N_tests):
                fidelities[i,run]=init_state.measure_prob(test_state[i,run])
            if not run%(runs//10): print(fidelities[:,run],test_state[:,run])
        print("Simulation ended in: %.3fs" % (time.time()-t0))
        nbin=21
        plt.hist(fidelities.T,range=(0,1),bins=nbin, log=True)
        plt.semilogy(np.arange(0,1,0.01),runs/nbin*P_haar(np.arange(0,1,0.01),2),label="Haar")
        plt.show()
        

        