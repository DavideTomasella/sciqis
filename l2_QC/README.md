# QState, QGate, QUGate
> The code works by definiting QState and QGate elements and using them in a "linear algebra style"
> ```
> in_state = QState._0()
> gate = QGate.hadamard()
> out_state = gate*in_state
> ```
> The measurement of a state relies on defining the projection measurement as gates or directly calculating the collapse probability
> ```
> projection_0 = QGate.measurement(mode=0,outcome=0)
> projection_1 = QGate.measurement(mode=0,outcome=1)
> collapse_state = (projection_0*out_state).normalize()
> collapse_prob0 = out_state.meas_prob(QState._0())
> ```
> We can create multi-state and multi-gate through an efficient implementation by `pyKronecker`
> ```
> N_state = QState.create_state([QState._0()]*N)
> N_gate = QGate.hadamard([0,1],n=N)
> N_gate2 = QGate.cnot(0,1,n=N)
> out = N_gate2*N_gate*N_state
> ```
> We can measure part of the quantum state by listing the modes that are collapse and their outcome so to implement feedforward systems
> ```
> N_state = QState.create_state([QState._0()]*N)
> N_gate = QGate.hadamard(mode=[0,1],n=N)
> N_gate2 = QGate.cnot(c=0,q=1,n=N)
> collapse_state, prob = (N_gate2*N_gate*N_state).collapse_after_measurement()
> ```