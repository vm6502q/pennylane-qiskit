import pytest

import numpy as np
import pennylane as qml
from scipy.linalg import block_diag

from pennylane_qiskit import AerDevice, BasicAerDevice

from conftest import U, U2, A

np.random.seed(42)


# global variables and functions
I = np.identity(2)
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])
H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
S = np.diag([1, 1j])
T = np.diag([1, np.exp(1j * np.pi / 4)])
SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
CZ = np.diag([1, 1, 1, -1])
toffoli = np.diag([1 for i in range(8)])
toffoli[6:8, 6:8] = np.array([[0, 1], [1, 0]])
CSWAP = block_diag(I, I, SWAP)


phase_shift = lambda phi: np.array([[1, 0], [0, np.exp(1j * phi)]])
rx = lambda theta: np.cos(theta / 2) * I + 1j * np.sin(-theta / 2) * X
ry = lambda theta: np.cos(theta / 2) * I + 1j * np.sin(-theta / 2) * Y
rz = lambda theta: np.cos(theta / 2) * I + 1j * np.sin(-theta / 2) * Z
rot = lambda a, b, c: rz(c) @ (ry(b) @ rz(a))
crz = lambda theta: np.array(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, np.exp(-1j * theta / 2), 0],
        [0, 0, 0, np.exp(1j * theta / 2)],
    ]
)


single_qubit = [
    (qml.PauliX, X),
    (qml.PauliY, Y),
    (qml.PauliZ, Z),
    (qml.Hadamard, H),
    (qml.S, S),
    (qml.T, T),
]

single_qubit_inverse = [
    (qml.S, S.conj().T),
    (qml.T, T.conj().T),
]

single_qubit_param = [(qml.PhaseShift, phase_shift), (qml.RX, rx), (qml.RY, ry), (qml.RZ, rz)]
two_qubit = [(qml.CNOT, CNOT), (qml.SWAP, SWAP), (qml.CZ, CZ)]
two_qubit_param = [(qml.CRZ, crz)]
three_qubit = [(qml.Toffoli, toffoli), (qml.CSWAP, CSWAP)]


@pytest.mark.parametrize("analytic", [True])
@pytest.mark.parametrize("shots", [8192])
@pytest.mark.usefixtures("skip_unitary")
class TestStateApply:
    """Test application of PennyLane operations to state simulators."""

    def test_qubit_state_vector(self, init_state, device, tol):
        """Test PauliX application"""
        dev = device(1)
        state = init_state(1)

        dev.apply(qml.QubitStateVector(state, wires=[0]))
        dev._obs_queue = []
        dev.pre_measure()

        res = np.abs(dev.state) ** 2
        expected = np.abs(state) ** 2
        assert np.allclose(res, expected, **tol)

    @pytest.mark.parametrize("operation,mat", single_qubit)
    def test_single_qubit_no_parameters(self, init_state, device, operation, mat, tol):
        """Test PauliX application"""
        dev = device(1)
        state = init_state(1)

        dev.apply(qml.QubitStateVector(state, wires=[0]))
        dev.apply(operation(wires=[0]))
        dev._obs_queue = []
        dev.pre_measure()

        res = np.abs(dev.state) ** 2
        expected = np.abs(mat @ state) ** 2
        assert np.allclose(res, expected, **tol)

    @pytest.mark.parametrize("operation,mat", single_qubit_inverse)
    def test_single_qubit_no_parametersi_inverse(self, init_state, device, operation, mat, tol):
        """Test PauliX application"""
        dev = device(1)
        state = init_state(1)

        dev.apply(qml.QubitStateVector(state, wires=[0]))
        dev.apply(operation(wires=[0]).inv())
        dev._obs_queue = []
        dev.pre_measure()

        res = np.abs(dev.state) ** 2
        expected = np.abs(mat @ state) ** 2
        assert np.allclose(res, expected, **tol)

    @pytest.mark.parametrize("theta", [0.5432, -0.232])
    @pytest.mark.parametrize("operation,func", single_qubit_param)
    def test_single_qubit_parameters(self, init_state, device, operation, func, theta, tol):
        """Test PauliX application"""
        dev = device(1)
        state = init_state(1)

        dev.apply(qml.QubitStateVector(state, wires=[0]))
        dev.apply(operation, [0], [theta])
        dev._obs_queue = []
        dev.pre_measure()

        res = np.abs(dev.state) ** 2
        expected = np.abs(func(theta) @ state) ** 2
        assert np.allclose(res, expected, **tol)

    @pytest.mark.parametrize("operation,mat", two_qubit)
    def test_two_qubit_no_parameters(self, init_state, device, operation, mat, tol):
        """Test PauliX application"""
        dev = device(2)
        state = init_state(2)

        dev.apply(qml.QubitStateVector(state, wires=[0, 1]))
        dev.apply(operation(wires=[0, 1]))
        dev._obs_queue = []
        dev.pre_measure()

        res = np.abs(dev.state) ** 2
        expected = np.abs(mat @ state) ** 2
        assert np.allclose(res, expected, **tol)

    @pytest.mark.parametrize("mat", [U, U2])
    def test_qubit_unitary(self, init_state, device, mat, tol):
        N = int(np.log2(len(mat)))
        dev = device(N)
        state = init_state(N)

        dev.apply(qml.QubitStateVector(state, wires=list(range(N))))
        dev.apply(qml.QubitUnitary(mat, wires=list(range(N))))
        dev._obs_queue = []
        dev.pre_measure()

        res = np.abs(dev.state) ** 2
        expected = np.abs(mat @ state) ** 2
        assert np.allclose(res, expected, **tol)

    @pytest.mark.parametrize("operation, mat", three_qubit)
    def test_three_qubit_no_parameters(self, init_state, device, operation, mat, tol):
        dev = device(3)
        state = init_state(3)

        dev.apply(qml.QubitStateVector(state, wires=[0, 1, 2]))
        dev.apply(qml.QubitUnitary(mat, wires=[0, 1, 2]))

        dev._obs_queue = []
        dev.pre_measure()

        res = np.abs(dev.state) ** 2
        expected = np.abs(mat @ state) ** 2
        assert np.allclose(res, expected, **tol)

    @pytest.mark.parametrize("theta", [0.5432, -0.232])
    @pytest.mark.parametrize("operation,func", two_qubit_param)
    def test_single_qubit_parameters(self, init_state, device, operation, func, theta, tol):
        """Test PauliX application"""
        dev = device(2)
        state = init_state(2)

        dev.apply(qml.QubitStateVector(state, wires=[0, 1]))
        dev.apply(operation(theta, wires=[0, 1]))
        dev._obs_queue = []
        dev.pre_measure()

        res = np.abs(dev.state) ** 2
        expected = np.abs(func(theta) @ state) ** 2
        assert np.allclose(res, expected, **tol)

@pytest.mark.parametrize("analytic", [True])
@pytest.mark.parametrize("shots", [8192])
@pytest.mark.usefixtures("run_only_for_unitary")
class TestStateApplyUnitarySimulator:
    """Test application of PennyLane operations to the unitary simulator."""

    def test_invalid_qubit(self, init_state, device):
        """Test that an exception is raised if the
        unitary matrix is applied on a unitary simulator."""
        dev = device(1)
        state = init_state(1)

        with pytest.raises(qml.QuantumFunctionError, match="The QubitStateVector operation is not supported on the unitary simulator backend"):
           dev.apply(qml.QubitStateVector(state, wires=[0]))

@pytest.mark.parametrize("shots", [8192])
@pytest.mark.parametrize("analytic", [False])
@pytest.mark.usefixtures("skip_unitary")
class TestHardwareApply:
    """Test application of PennyLane operations on hardware simulators."""

    def test_qubit_state_vector(self, init_state, device, tol):
        """Test PauliX application"""
        dev = device(1)
        state = init_state(1)

        dev.apply(qml.QubitStateVector(state, wires=[0]))
        dev._obs_queue = []
        dev.pre_measure()

        res = np.fromiter(dev.probabilities().values(), dtype=np.float64)
        expected = np.abs(state) ** 2
        assert np.allclose(res, expected, **tol)

    def test_invalid_qubit_state_vector(self, device):
        """Test that an exception is raised if the state
        vector is the wrong size"""
        dev = device(2)
        state = np.array([0, 123.432])

        with pytest.raises(ValueError, match=r"State vector must be of length 2\*\*wires"):
            dev.apply(qml.QubitStateVector(state, wires=[0, 1]))

    @pytest.mark.parametrize("operation,mat", single_qubit)
    def test_single_qubit_no_parameters(self, init_state, device, operation, mat, tol):
        """Test PauliX application"""
        dev = device(1)
        state = init_state(1)

        dev.apply(qml.QubitStateVector(state, wires=[0]))
        dev.apply(operation(wires=[0]))
        dev._obs_queue = []
        dev.pre_measure()

        res = np.fromiter(dev.probabilities().values(), dtype=np.float64)
        expected = np.abs(mat @ state) ** 2
        assert np.allclose(res, expected, **tol)

    @pytest.mark.parametrize("theta", [0.5432, -0.232])
    @pytest.mark.parametrize("operation,func", single_qubit_param)
    def test_single_qubit_parameters(self, init_state, device, operation, func, theta, tol):
        """Test PauliX application"""
        dev = device(1)
        state = init_state(1)

        dev.apply(qml.QubitStateVector(state, wires=[0]))
        dev.apply(operation(theta, wires=[0]))
        dev._obs_queue = []
        dev.pre_measure()

        res = np.fromiter(dev.probabilities().values(), dtype=np.float64)
        expected = np.abs(func(theta) @ state) ** 2
        assert np.allclose(res, expected, **tol)

    @pytest.mark.parametrize("operation,mat", two_qubit)
    def test_two_qubit_no_parameters(self, init_state, device, operation, mat, tol):
        """Test PauliX application"""
        dev = device(2)
        state = init_state(2)

        dev.apply(qml.QubitStateVector(state, wires=[0, 1]))
        dev.apply(operation(wires=[0, 1]))

        dev._obs_queue = []
        dev.pre_measure()

        res = np.fromiter(dev.probabilities().values(), dtype=np.float64)
        expected = np.abs(mat @ state) ** 2
        assert np.allclose(res, expected, **tol)

    @pytest.mark.parametrize("mat", [U, U2])
    def test_qubit_unitary(self, init_state, device, mat, tol):
        N = int(np.log2(len(mat)))
        dev = device(N)
        state = init_state(N)

        dev.apply(qml.QubitStateVector(state, wires=list(range(N))))
        dev.apply(qml.QubitUnitary(mat, wires=list(range(N))))
        dev._obs_queue = []
        dev.pre_measure()

        res = np.fromiter(dev.probabilities().values(), dtype=np.float64)
        expected = np.abs(mat @ state) ** 2
        assert np.allclose(res, expected, **tol)

    def test_invalid_qubit_state_unitary(self, device):
        """Test that an exception is raised if the
        unitary matrix is the wrong size"""
        dev = device(2)
        state = np.array([[0, 123.432], [-0.432, 023.4]])

        with pytest.raises(ValueError, match="State vector must be of length"):
            dev.apply(qml.QubitStateVector(state, wires=[0, 1]))

    @pytest.mark.parametrize("operation, mat", three_qubit)
    def test_three_qubit_no_parameters(self, init_state, device, operation, mat, tol):
        dev = device(3)
        state = init_state(3)

        dev.apply(qml.QubitStateVector(state, wires=[0, 1, 2]))
        dev.apply(qml.QubitUnitary(mat, wires=[0, 1, 2]))
        dev._obs_queue = []
        dev.pre_measure()

        res = np.fromiter(dev.probabilities().values(), dtype=np.float64)
        expected = np.abs(mat @ state) ** 2
        assert np.allclose(res, expected, **tol)

    @pytest.mark.parametrize("theta", [0.5432, -0.232])
    @pytest.mark.parametrize("operation,func", two_qubit_param)
    def test_single_qubit_parameters(self, init_state, device, operation, func, theta, tol):
        """Test PauliX application"""
        dev = device(2)
        state = init_state(2)

        dev.apply(qml.QubitStateVector(state, wires=[0, 1]))
        dev.apply(operation(theta, wires=[0, 1]))
        dev._obs_queue = []
        dev.pre_measure()

        res = np.fromiter(dev.probabilities().values(), dtype=np.float64)
        expected = np.abs(func(theta) @ state) ** 2
        assert np.allclose(res, expected, **tol)
