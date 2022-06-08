import numpy as np
from util import constants


class ParallelSimulator:
    """
    Simulates a single array that adheres to the standard abstract PIM model, with partitions
    The addressing is in a strided form, where an intra-partition index is specified alongside which partitions to
    apply the operation on
    """

    def __init__(self, num_rows: int, num_cols: int, num_partitions: int):
        """
        Initializes the array simulator with the given number of rows and columns
        :param num_rows: the number of rows
        :param num_cols: the number of columns
        :param num_partitions: the number of partitions. Assumes num_cols is divisible by num_partitions.
        """
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_partitions = num_partitions
        self.memory = np.zeros((num_cols, num_rows), dtype=bool)
        self.latency = 0
        self.energy = 0
        self.log = []
        self.maxUsed = 0

    def perform(self, gateType: constants.GateType, inputs, outputs, input_partitions=None, output_partitions=None):
        """
        Performs the given logic gate on the simulated array. For logic operations, the result is ANDed with the
            previous output value (to simulate the effect of logic initialization steps).
        :param gateType: the type of gate to perform
        :param inputs: the list (python list or numpy array) of intra-partition input indices
        :param outputs: the list (python list or numpy array) of intra-partition output indices
        :param input_partitions: the list of the partitions of the input partitions to apply the gate on. If None,
        then applies to all partitions.
        :param output_partitions: the list of the partitions of the output partitions to apply the gate on. If None,
        then equals input_partitions (fully parallel within inter_partitions)
        """

        if input_partitions is None:
            input_partitions = np.arange(self.num_partitions)
        if output_partitions is None:
            output_partitions = input_partitions
        input_partitions = input_partitions.reshape(-1)
        output_partitions = output_partitions.reshape(-1)

        # Sort in increasing order
        sort_idx = np.argsort(input_partitions)
        input_partitions = input_partitions[sort_idx]
        output_partitions = output_partitions[sort_idx]

        # Verify adherence to the minimal model

        if gateType == constants.GateType.NOT or gateType == constants.GateType.NOR:

            # Uniform distances
            distances = output_partitions - input_partitions
            assert((distances == distances[0]).all())

            # Periodic (check that input_partitions is arithmetic)
            assert(np.all(np.diff(input_partitions, 2) == 0))

            # No partition intersections
            if len(input_partitions) > 1:
                assert(np.abs(distances[0]) < np.abs(np.diff(input_partitions)[0]))

            # No intra-partition intersections
            if distances[0] == 0:
                # Check no intersection in inputs and outputs
                assert (len(np.intersect1d(inputs, outputs)) == 0)
                # Check inputs are unique
                assert (len(np.unique(inputs)) == len(inputs))
                # Check outputs are unique
                assert (len(np.unique(outputs)) == len(outputs))

        else:

            # Periodic (check that output_partitions is arithmetic)
            np.all(np.diff(output_partitions, 2) == 0)

        # The start indices of the partitions used
        input_partition_starts = (input_partitions * (self.num_cols // self.num_partitions))
        output_partition_starts = (output_partitions * (self.num_cols // self.num_partitions))

        if gateType == constants.GateType.NOT:
            self.memory[output_partition_starts + outputs[0]] = np.bitwise_and(self.memory[output_partition_starts + outputs[0]],
                np.bitwise_not(self.memory[input_partition_starts + inputs[0]]))

            pstr = (f"for i in range({input_partitions[0]}, {input_partitions[-1] + 1}, {np.diff(input_partitions)[0]})"
                if len(input_partitions) > 1 else f"for i in range({input_partitions[0]}, {input_partitions[0] + 1}, 1)").ljust(30) \
                + f" and j = i + {output_partitions[0] - input_partitions[0]}"
            self.emit(f'T{str(self.latency) + ":":<4}\tp_j.{outputs[0]:<2} = {f"NOT(p_i.{inputs[0]})".ljust(25)}' + pstr)

        elif gateType == constants.GateType.NOR:
            self.memory[output_partition_starts + outputs[0]] = np.bitwise_and(self.memory[output_partition_starts + outputs[0]],
                np.bitwise_not(np.bitwise_or(self.memory[input_partition_starts + inputs[0]], self.memory[input_partition_starts + inputs[1]])))

            pstr = (f"for i in range({input_partitions[0]}, {input_partitions[-1] + 1}, {np.diff(input_partitions)[0]})"
                if len(input_partitions) > 1 else f"for i in range({input_partitions[0]}, {input_partitions[0] + 1}, 1)").ljust(30) \
                + f" and j = i + {output_partitions[0] - input_partitions[0]}"
            self.emit(f'T{str(self.latency) + ":":<4}\tp_j.{outputs[0]:<2} = {f"NOR(p_i.{inputs[0]}, p_i.{inputs[1]})".ljust(25)}' + pstr)

        elif gateType == constants.GateType.INIT0:
            self.memory[output_partition_starts + outputs] = False

            pstr = (f"for i in range({output_partitions[0]}, {output_partitions[-1] + 1}, {np.diff(output_partitions)[0]})"
                if len(output_partitions) > 1 else f"for i in range({output_partitions[0]}, {output_partitions[0] + 1}, 1)").ljust(30) + f" and j = i + 0"
            self.emit(f'T{str(self.latency) + ":":<4}\tp_j.{outputs[0]:<2} = {"INIT0".ljust(25)}' + pstr)

        elif gateType == constants.GateType.INIT1:
            self.memory[output_partition_starts + outputs] = True

            pstr = (f"for i in range({output_partitions[0]}, {output_partitions[-1] + 1}, {np.diff(output_partitions)[0]})"
                if len(output_partitions) > 1 else f"for i in range({output_partitions[0]}, {output_partitions[0] + 1}, 1)").ljust(30) + f" and j = i + 0"
            self.emit(f'T{str(self.latency) + ":":<4}\tp_j.{outputs[0]:<2} = {"INIT1".ljust(25)}' + pstr)

        self.latency += 1
        self.energy += len(output_partitions)
        self.maxUsed = max(self.maxUsed, outputs[0])

    def read(self, x_addr, partitions=None):
        """
        Reads the bits corresponding to the intra-partition index on the given partitions
        :param x_addr: the intra-partition index
        :param partitions: the list of partitions to read from
        :return the data at those indices (numpy array of dimension len(partitions) x num_rows)
        """
        if partitions is None:
            partitions = np.arange(self.num_partitions)
        return self.memory[(partitions * (self.num_cols // self.num_partitions)) + x_addr]

    def write(self, x_addr, data, partitions=None):
        """
        Writes the bits corresponding to the intra-partition index on the given partitions
        :param x_addr: the intra-partition index
        :param data: the data to write at those indices (numpy array of dimension len(partitions) x num_rows)
        :param partitions: the list of partitions to read from
        """
        if partitions is None:
            partitions = np.arange(self.num_partitions)
        self.memory[(partitions * (self.num_cols // self.num_partitions)) + x_addr] = data

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
