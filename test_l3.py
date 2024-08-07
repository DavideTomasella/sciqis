from l2_quantumComputer import QState, QGate, QUGate

# Script for Austin performance analysis
def computer():
    state_00 = QState.create_state([QState._0()]*10)
    had = QGate.create_gate([QGate._hadamard()]+[QGate._id()]*9)
    cnot = QGate.create_gate([QGate._0x0(), QGate._id(), QGate._id()]+[QGate._id()]*7) + QGate.create_gate([QGate._1x1(), QGate._id(), QGate._x()]+[QGate._id()]*7)
    #cnot2 = QGate.create_gate([QGate._0(), QGate._id(), QGate._id()]+[QGate._id()]*7) + QGate.create_gate([QGate._1(), QGate._id(), QGate._x()]+[QGate._id()]*7)
    cnot*(had*state_00)

for i in range(10000):
    computer()
    #print(i)