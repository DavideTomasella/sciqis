import numpy as np
import matplotlib.pyplot as plt
import time
from l2_quantumComputer import QState,QGate

if __name__ == "__main__":
    rng_gen=np.random.default_rng()
    runs=1000
    N_states=2
    N_pars=10
    N_tests = 7
    rngs=rng_gen.uniform(0,2*np.pi,size=(runs,N_pars,N_states))
    #Init
    init_state=QState.create_state([QState._0()]*N_states)
    test_state=np.zeros((N_tests,runs),dtype=QState)
    fidelities=np.zeros((N_tests,runs))
    P_haar=lambda F,N: (N)*(1-F)**(N-1)
    had=QGate.hadamard(range(N_states),n=N_states)
    identity=QGate.create_id(n=N_states)
    #start timer
    t0=time.time()
    for run in range(runs):
        Ry=QGate.Ry(range(N_states),rngs[run,0,:],n=N_states)
        Rx=QGate.Rx(range(N_states),rngs[run,1,:],n=N_states)
        Rz=QGate.Rz(range(N_states),rngs[run,2,:],n=N_states)
        Ry2=QGate.Ry(range(N_states),rngs[run,3,:],n=N_states)
        Rx2=QGate.Rx(range(N_states),rngs[run,4,:],n=N_states)
        Rz2=QGate.Rz(range(N_states),rngs[run,5,:],n=N_states)
        Ry3=QGate.Ry(range(N_states),rngs[run,6,:],n=N_states)
        Rx3=QGate.Rx(range(N_states),rngs[run,7,:],n=N_states)
        Rz3=QGate.Rz(range(N_states),rngs[run,8,:],n=N_states)

        test_state[0,run]=Ry*had*init_state
        test_state[1,run]=Rx*had*init_state
        test_state[2,run]=Rz*had*init_state
        test_state[3,run]=Rx*Ry*had*init_state
        test_state[4,run]=Rx*Rz*Ry*had*init_state
        test_state[5,run]=Rx2*Rz2*Ry2*Rx*Rz*Ry*had*init_state
        test_state[6,run]=Rx3*Rz3*Ry3*Rx2*Rz2*Ry2*Rx*Rz*Ry*had*init_state
        for i in range(N_tests):
            fidelities[i,run]=init_state.measure_prob(test_state[i,run])
        if not run%(runs//10): print(fidelities[:,run],test_state[:,run])
    print("Simulation ended in: %.3fs" % (time.time()-t0))
    nbin=51
    plt.figure()
    for i in range(N_tests):
        plt.hist(fidelities[i,:],range=(0,1),bins=nbin, log=True,alpha=0.5,label="Test %d" % i)
    plt.semilogy(np.arange(0,1,0.01),runs/nbin*P_haar(np.arange(0,1,0.01),N_states),label="Haar")
    plt.legend()
    plt.show()
