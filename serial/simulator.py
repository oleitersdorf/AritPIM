import numpy as np
from util import constants


class SerialSimulator:
    """
    Simulates a single array that adheres to the standard abstract PIM model
    """

    def __init__(self, num_rows: int, num_cols: int):
        """
        Initializes the array simulator with the given number of rows and columns
        :param num_rows: the number of rows
        :param num_cols: the number of columns
        """
        self.memory = np.zeros((num_cols, num_rows), dtype=bool)
        self.latency = 0
        self.log = []
        self.maxUsed = 0

    def perform(self, gateType: constants.GateType, inputs, outputs):
        """
        Performs the given logic gate on the simulated array. For logic operations, the result is ANDed with the
            previous output value (to simulate the effect of logic initialization steps).
        :param gateType: the type of gate to perform
        :param inputs: the list (python list or numpy array) of input indices
        :param outputs: the list (python list or numpy array) of output indices
        """

        # Check no intersection in inputs and outputs
        assert(len(np.intersect1d(inputs, outputs)) == 0)
        # Check inputs are unique
        assert(len(np.unique(inputs)) == len(inputs))
        # Check outputs are unique
        assert(len(np.unique(outputs)) == len(outputs))

        if gateType == constants.GateType.NOT:
            self.memory[outputs[0]] = np.bitwise_and(self.memory[outputs[0]],
                                        np.bitwise_not(self.memory[inputs[0]]))
            self.emit(f'T{str(self.latency) + ":":<4}\t{outputs[0]:<3} = NOT({inputs[0]})')

        elif gateType == constants.GateType.NOR:
            self.memory[outputs[0]] = np.bitwise_and(self.memory[outputs[0]],
                                        np.bitwise_not(np.bitwise_or(self.memory[inputs[0]], self.memory[inputs[1]])))
            self.emit(f'T{str(self.latency) + ":":<4}\t{outputs[0]:<3} = NOR({inputs[0]}, {inputs[1]})')

        elif gateType == constants.GateType.INIT0:
            self.memory[outputs[0]] = False
            self.emit(f'T{str(self.latency) + ":":<4}\t{outputs[0]:<3} = INIT0')

        elif gateType == constants.GateType.INIT1:
            self.memory[outputs[0]] = True
            self.emit(f'T{str(self.latency) + ":":<4}\t{outputs[0]:<3} = INIT1')

        self.latency += 1
        self.maxUsed = max(self.maxUsed, outputs[0])

    def emit(self, message):
        """
        Emits a message to the log buffer of the simulator
        :param message: the message to emit
        """
        self.log.append(message)

    def getLog(self):
        """
        Returns the current log of the simulator
        :return: the log as a list of strings
        """
        return self.log
