import numpy as np
import typing as tp
import cmath
import functools as ft
from kron_vec_product import kron_vec_prod, kron_brute_force
import pykronecker as pk

class QState:
    __tolerance = 1e-9

    @staticmethod
    def create_state(args:tp.Union[np.ndarray[complex],tp.Self])->tp.Self:
        if isinstance(args[0], QState):
            args = list(map(lambda x: x.value, args))
        # I need to create diag matrices because the kronecker product is not working with 1D arrays.
        # TODO handle not square matrices, see online so we don't have to create the diag matrices.
        args=list(map(lambda x: np.diag(x), args))
        v = np.ones((np.prod([a.shape[1] for a in args]), ))

        kron_res = pk.KroneckerProduct(args) @ v # 70us @ 10 elements
        #kron_res = kron_vec_prod(args, v) # 170us @ 10 elements
        #kron_res = ft.reduce(np.kron, args)@v # 19ms @ 10 elementss
        return QState(kron_res)
    
    def __init__(self, state_or_value : tp.Union[tp.Self, tp.List[complex],np.ndarray[complex]] = np.ndarray([1, 0])):
        if isinstance(state_or_value, QState):
            state_or_value = state_or_value.value
        elif type(state_or_value) == list:
            state_or_value = np.array(state_or_value)
        self._value = state_or_value
        self._dim = np.size(state_or_value)

    @classmethod
    def _0(self):
        return self([1,0])
        
    @classmethod
    def _1(self):
        return self([0,1])

    @classmethod
    def _p(self):
        return self([1,1]/np.sqrt(2))

    @classmethod
    def _m(self):
        return self([1,-1]/np.sqrt(2))

    @classmethod
    def _r(self):
        return self([1j,0])

    @classmethod
    def _l(self):
        return self([0,1j])

    @property
    def dim(self):
        return self._dim

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, state_value:np.ndarray[complex]):
        if not QState._check_norm(state_value) >= 0:
            raise ValueError("State norm must be one.")            
        if np.size(state_value) < 1:
            raise ValueError("State dimension must be greater than 0")
        self._dim = np.size(state_value)
        self._value = state_value.copy()

    @staticmethod
    def _check_norm(state_value:np.ndarray[complex]):
        return cmath.isclose(np.linalg.norm(state_value),1, rel_tol=QState.__tolerance)

    def __mod__(self, other: tp.Self) -> tp.Self:
        return QState(np.kron(self._value, other._value))
    
    #def __mul__(self, other: tp.Self) -> complex:
    #    return np.dot(np.conj(self._value.T), other._value)
    def ket(self) -> tp.Self:
        return self._value
    
    def bra(self) -> tp.Self:
        return np.conj(self._value.T)
    
    def proj(self, other: tp.Self) -> complex:
        return other.bra()*self.ket()
    
    def out(self, other: tp.Self) -> np.ndarray[complex]:
        return self.ket()*other.bra()
    
    def measure_prob(self, other: tp.Self) -> complex:
        return np.abs(self.proj(other))**2
    
    def __str__(self):
        return str(self._value)

    def __repr__(self):
        return "QS:"+self.__str__()

class QGate:
    __tolerance = 1e-9
    
    @staticmethod
    def create_gate(args:tp.List[tp.Union[np.ndarray[complex],tp.Self]])->tp.Self:
        if len(args) < 1:
            raise ValueError("At least one gate is required.")
        if isinstance(args[0], QGate):
            #concatenate x.matrix.As of each args
            args = [A for As in (map(lambda x: x.matrix.As, args)) for A in As]
            
        # I need to create diag matrices because the kronecker product is not working with 1D arrays.
        #args=list(map(lambda x: np.diag(x), args))
        #v = np.ones((np.prod([a.shape[1] for a in args]), ))

        #kron_res = kron_vec_prod(args, v) # 
        #kron_res = ft.reduce(np.kron, args) # 53ms @ 10 elementss
        kron_res = pk.KroneckerProduct(args) # 500ms @ 10 elements
        #print(kron_res)
        return QGate(kron_res)

    def __init__(self, matrix_or_gate:tp.Union[tp.Self,np.ndarray[complex],pk.base.KroneckerOperator]):
        #if type(matrix) == list:
        #    matrix = np.array(matrix)
        if isinstance(matrix_or_gate, QGate):
            matrix_or_gate = matrix_or_gate.matrix
        if isinstance(matrix_or_gate, np.ndarray):
            matrix_or_gate = pk.KroneckerProduct([matrix_or_gate])
        self._matrix = matrix_or_gate
        self._dim = len(matrix_or_gate)

    @classmethod
    def _id(self, dim:int=2):
        return self(np.eye(dim))
    
    @classmethod
    def _0x0(self):
        return self(np.array([[1,0],[0,0]]))
    
    @classmethod
    def _1x1(self):
        return self(np.array([[0,0],[0,1]]))
    
    @classmethod
    def _x(self):
        return self(np.array([[0,1],[1,0]]))
    
    @classmethod
    def _y(self):
        return self(np.array([[0,-1j],[1j,0]]))
    
    @classmethod
    def _z(self):
        return self(np.array([[1,0],[0,-1]]))
    
    @classmethod
    def _not(self):
        return self._x()
    
    @classmethod
    def _hadamard(self):
        return self(1/np.sqrt(2)*np.array([[1,1],[1,-1]]))

    @classmethod
    def _phase(self):
        return self(np.array([[1,0],[0,1j]]))
    
    @classmethod
    def _Rz(self, theta:float):
        return self(np.array([[np.exp(1j*theta/2),0],[0,np.exp(-1j*theta/2)]]))
        return self(np.array([[1,0],[0,np.exp(1j*theta)]]))
    
    @classmethod
    def _Rx(self, theta:float):
        return self(np.array([[np.cos(theta/2),-1j*np.sin(theta/2)],[-1j*np.sin(theta/2),np.cos(theta/2)]]))
    
    @classmethod
    def _Ry(self, theta:float):
        return self(np.array([[np.cos(theta/2),-np.sin(theta/2)],[np.sin(theta/2),np.cos(theta/2)]]))
    
    @classmethod
    def cnot(self,c,q,n=2):
        #build the matrix using the kronecker product of the identity and the x gate
        #for dim=2 QGate.create_gate([QGate._0(), QGate._id()]) + QGate.create_gate([QGate._1(), QGate._x()])
        #in general ...
        tmp_0 = [QGate._id()]*n
        tmp_0[c] = QGate._0x0()
        tmp_1 = [QGate._id()]*n
        tmp_1[c] = QGate._1x1()
        tmp_1[q] = QGate._x()
        return self.create_gate(tmp_0) + self.create_gate(tmp_1)

    @property
    def dim(self):
        return self._dim

    @property
    def matrix(self):
        return self._matrix

    @matrix.setter
    def matrix(self, gate_matrix:pk.base.KroneckerOperator):
        self._dim = len(gate_matrix)
        self._matrix = gate_matrix.copy()

    def __mod__(self, other: tp.Self) -> tp.Self:
        #return QGate(pk.KroneckerProduct([self._matrix, other._matrix]))
        # no changes so we can create the matrix in 2 different ways.
        kron_res = pk.KroneckerProduct([np.array(self._matrix), np.array(other._matrix)])
        return QGate(np.kron(self._matrix, other._matrix))
    
    def __mul__(self, other) -> tp.Union[QState, tp.Self]:
        if isinstance(other, QState):
            return QState(self._matrix @ other.value)
        elif isinstance(other, QGate):
            try:
                return QGate(self.matrix @ other._matrix)
            except:
                return QGate(np.ndarray(self.matrix) @ np.ndarray(other.matrix))
    
    def __add__(self, other: tp.Self) -> tp.Self:
        return QGate(self._matrix + other._matrix)
        return QGate(pk.KroneckerSum([self._matrix ,other._matrix]))

    def __str__(self):
        return str(self._matrix)
    
    def __repr__(self):
        return "QG:"+self.__str__()

class QUGate(QGate):
    __tolerance = 1e-9
    
    def __init__(self, matrix_or_gate:tp.Union[QGate,np.ndarray[complex],pk.base.KroneckerOperator]):
        if isinstance(matrix_or_gate, QGate):
            matrix_or_gate = matrix_or_gate.matrix
        if isinstance(matrix_or_gate, np.ndarray):
            if not self._check_unitary(matrix_or_gate):
                raise ValueError("Matrix must be unitary.")
            matrix_or_gate = pk.KroneckerProduct([matrix_or_gate])
        #if type(matrix_or_gate) == list:
        #    matrix_or_gate = np.array(matrix_or_gate)
        super().__init__(matrix_or_gate)
        #for gate in args:
        #    self %= QUGate(gate)
        
    @QGate.matrix.setter
    def matrix(self, matrix:pk.base.KroneckerOperator):
        super().matrix = matrix

    @staticmethod
    def _check_unitary(matrix:np.ndarray[complex]):
        return cmath.isclose(np.abs(np.linalg.det(matrix)),1, rel_tol=QUGate.__tolerance)
    
    def __mod__(self, other: QGate) -> QGate:
        if isinstance(other, QUGate):
            return QUGate(np.kron(self._matrix, other._matrix))
        return super().__mod__(other)
    
    def __mul__(self, other) -> tp.Union[QState, QGate]:
        if isinstance(other, QUGate):
            try:
                # This is the speed up when I'm creating the matrices with the library pykronecker.
                return QUGate(self.matrix @ other._matrix)
            except:
                # Otherwise do partial calculation and keep trying to use the library for the future.
                return QUGate(np.ndarray(self.matrix) @ np.ndarray(other.matrix))
        return super().__mul__(other)
    
    def __add__(self, other: QGate) -> QGate:
        if isinstance(other, QUGate):
            return QUGate(self._matrix + other._matrix)
        return super().__add__(other)
        return QUGate(pk.KroneckerSum([self._matrix ,other._matrix]))
    
# In[]
if __name__ == "__main__":

    g = QGate.create_gate([QGate._hadamard(),QGate._id(),QGate._id()])

    q = QState([1,0,0])
    print(QState._0())
    print(q.dim)
    print(q.value)
    q=QState(np.array([1,0]))
    print(q.dim)
    print(q.value)

    g=QGate(np.array([[1,0],[0,0.9]]))
    print(g)
    u=QUGate(np.array([[1,0],[0,1]]))
    print(u)

    print(QState._0()%QState._1())
    print(QUGate._id()%QUGate._id())
    print(QGate._id()%QGate([[1,0],[0,0.9]]))

    
    print(QUGate._hadamard() % QUGate._id())


    state_0000 = QState._0() % QState._0() % QState._0() #% QState._0()
    state_0001 = QState._0() % QState._0() % QState._0() #% QState._1()
    print(state_0000)
    gate = QUGate._hadamard() % QUGate._hadamard() % QUGate._hadamard() #% QUGate._id()
    print(gate)
    state2=gate*state_0000
    print(state2)
    print(state2.measure_prob(state_0000))
    print(state2.measure_prob(state_0001))


    state_00 = QState._0() % QState._0()
    had = QGate._hadamard() % QGate._id()
    cnot = QUGate(QGate._0x0() % QGate._id() + QGate._1x1() % QGate._x())
    print(cnot)
    print(had*state_00)
    print(cnot*(had*state_00))
    print(cnot*had*state_00)

    
    state_00 = QState._0() % QState._0() % QState._0()
    had = QGate._hadamard() % QGate._id() % QGate._id()
    cnot = QUGate(QGate._0x0() % QGate._id() % QGate._id() + QGate._1x1() % QGate._id() % QGate._x())

    print(cnot)
    print(had*state_00)
    print(cnot*(had*state_00))
    print(cnot*had*state_00)

    print(QState.create_state([QState._0()]*3))
    print(QGate.create_gate([QGate._hadamard(),QGate._id(), QGate._id()]))
    print(QGate.create_gate([QGate._hadamard(),QGate._id()% QGate._id()]))
    new_state_00 = QState.create_state([QState._0()]*3)
    new_had = QGate.create_gate([QGate._hadamard(),QGate._id(), QGate._id()])
    new_cnot = QGate.create_gate([QGate._0x0(), QGate._id(), QGate._id()]) + QGate.create_gate([QGate._1x1(), QGate._id(), QGate._x()])
    
    print(new_cnot)
    print(new_had*new_state_00)
    print(new_cnot*(new_had*new_state_00))
    print(new_cnot*new_had*new_state_00)