import numpy as np
from util import constants
from math import ceil, log2
import simulator


class ParallelArithmetic:
    """
    The proposed algorithms for bit-parallel arithmetic
    """

    class IntermediateAllocator:
        """
        Helper that assists in the allocation of intermediate cells
        """

        def __init__(self, cells: np.ndarray, partitions: np.ndarray):
            """
            Initializes the allocator
            :param cells: a np list of the intra-partition indices of the available cells
            :param partitions: a np list of the partition indices of the available cells
            """

            self.cells = cells
            self.partitions = partitions
            self.cells_inverse = {cells[i]: i for i in range(len(cells))}
            self.partitions_inverse = {partitions[i]: i for i in range(len(partitions))}
            self.allocated = np.zeros((len(partitions), len(cells)),
                                      dtype=bool)  # vector containing 1 if allocated, 0 otherwise

        def malloc(self, num_cells: int, partitions: np.ndarray):
            """
            Allocates num_cells cells on the given partitions (same intra-partition indices for all partitions)
            :param num_cells: the number of cells to allocate
            :param partitions: the partitions to allocate on
            :return: np array of length num_cells containing the allocated intra-partition indices, or int if num_cells = 1
            """

            assert (num_cells >= 1)

            allocation = []

            if isinstance(partitions, np.int64):
                partitions = np.array([partitions], dtype=int)

            partition_indices = np.array([self.partitions_inverse[x] for x in partitions])

            # Search for available cells (first searching between previous allocations, then extending if necessary)
            for i in range(len(self.cells)):
                if np.sum(self.allocated[partition_indices, i], axis=0) == 0:
                    allocation.append(i)
                    # Mark the cells as allocated
                    self.allocated[partition_indices, i] = True
                if len(allocation) == num_cells:
                    break

            # Assert that there were enough cells
            assert (len(allocation) == num_cells)

            # Return the allocated cells
            if num_cells > 1:
                return np.array(self.cells[allocation], dtype=int)
            else:
                return self.cells[allocation[0]]

        def free(self, cells, partitions: np.ndarray):
            """
            Frees the given cells in the given partitions
            :param cells: np array containing the intra-partition cells to free, or int (if num_cells was 1)
            :param partitions: the partitions to free
            """

            if isinstance(partitions, np.int64):
                partitions = np.array([partitions], dtype=int)

            partition_indices = np.array([self.partitions_inverse[x] for x in partitions])

            if isinstance(cells, np.ndarray):
                for x in cells:
                    self.allocated[partition_indices, self.cells_inverse[x]] = False
            else:
                self.allocated[partition_indices, self.cells_inverse[cells]] = False

    @staticmethod
    def fixedAddition(sim: simulator.ParallelSimulator, x_addr: int, y_addr: int, z_addr: int,
            inter, partitions=None, cin_addr=None, cin_partition=None, cout_addr=None, cout_partition=None):
        """
        Performs a fixed-point addition on the given columns. Supports both unsigned and signed numbers.
        :param sim: the simulation environment
        :param x_addr: the intra-partition address of input x (N-bit)
        :param y_addr: the intra-partition address of input y (N-bit)
        :param z_addr: the intra-partition address of output z (N-bit)
        :param inter: addresses for inter. Either np array or IntermediateAllocator. Relevant to "partitions".
        :param partitions: the partitions to operate on
        :param cin_addr: the intra-partition address of optional input carry (1-bit). "-1" designates constant 1 input carry.
        :param cin_partition: the partition address of optional input carry (1-bit). Lowest partition by default.
        :param cout_addr: the intra-partition address of optional output carry (1-bit)
        :param cout_partition: the partition address of optional output carry (1-bit). Highest partition by default.
        """

        if partitions is None:
            partitions = np.arange(sim.num_partitions)
        N = len(partitions)
        log2_N = ceil(log2(N))

        if isinstance(inter, np.ndarray):
            inter = ParallelArithmetic.IntermediateAllocator(inter, partitions)

        # Pre-compute the propagate and not propagate bits
        pbit_addr = inter.malloc(1, partitions)
        notpbit_addr = inter.malloc(1, partitions)
        ParallelArithmetic.__or(sim, x_addr, y_addr, pbit_addr, inter, partitions, notz_addr=notpbit_addr)

        # Pre-compute the not generate bits
        notgbit_addr = inter.malloc(1, partitions)
        ParallelArithmetic.__nand(sim, x_addr, y_addr, notgbit_addr, inter, partitions)

        # Deal with carry-in: if carry-in and propagate at LSB, then set generate of LSB to one
        notcin_addr = None
        if cin_addr is not None:

            if cin_addr == -1:
                sim.perform(constants.GateType.NOT, [pbit_addr], [notgbit_addr], partitions[0])
            else:
                # Choose default cin_partition if not explicitly chosen
                cin_partition = partitions[0] if cin_partition is None else cin_partition
                # generate = OR(generate, AND(carry-in, propagate))
                # NOT generate = NOT OR(generate, AND(carry-in, propagate)) = AND(NOT generate, NOT AND (carry-in, propagate))
                temp = inter.malloc(1, partitions[0])
                notcin_addr = inter.malloc(1, partitions[0])
                sim.perform(constants.GateType.INIT1, [], [notcin_addr], partitions[0])
                sim.perform(constants.GateType.NOT, [cin_addr], [notcin_addr], cin_partition, partitions[0])
                sim.perform(constants.GateType.INIT1, [], [temp], partitions[0])
                sim.perform(constants.GateType.NOR, [notpbit_addr, notcin_addr], [temp], partitions[0])
                sim.perform(constants.GateType.NOT, [temp], [notgbit_addr], partitions[0])
                inter.free(temp, partitions[0])

        # Perform the prefix operation
        for i in range(log2_N - 1):

            # Perform operation from partitions[np.arange((1 << i), N - (1 << i), 1 << (i + 1))]
            # to partitions[np.arange(1 << (i + 1), N, 1 << (i + 1))]
            inp = partitions[np.arange((1 << i), N - (1 << i), 1 << (i + 1))]
            outp = partitions[np.arange(1 << (i + 1), N, 1 << (i + 1))]

            sim.perform(constants.GateType.NOT, [notgbit_addr], [pbit_addr], input_partitions=inp, output_partitions=outp)

            sim.perform(constants.GateType.NOT, [pbit_addr], [notgbit_addr], input_partitions=outp)

            sim.perform(constants.GateType.INIT1, [], [pbit_addr], input_partitions=outp)
            sim.perform(constants.GateType.NOT, [notpbit_addr], [pbit_addr], input_partitions=outp)

            sim.perform(constants.GateType.NOT, [notpbit_addr], [pbit_addr], input_partitions=inp, output_partitions=outp)

            sim.perform(constants.GateType.INIT1, [], [notpbit_addr], input_partitions=outp)
            sim.perform(constants.GateType.NOT, [pbit_addr], [notpbit_addr], input_partitions=outp)

        for i in range(log2_N):

            # Perform operation from partitions[np.arange(0, N, 1 << (log2_N - i))]
            # to partitions[np.arange((N >> (i + 1)), N + (N >> (i + 1)), 1 << (log2_N - i))]
            inp = partitions[np.arange(0, N - (1 << (log2_N - i - 1)), 1 << (log2_N - i))]
            outp = partitions[np.arange((1 << (log2_N - i - 1)), N, 1 << (log2_N - i))]

            sim.perform(constants.GateType.NOT, [notgbit_addr], [pbit_addr], input_partitions=inp, output_partitions=outp)

            sim.perform(constants.GateType.NOT, [pbit_addr], [notgbit_addr], input_partitions=outp)

            sim.perform(constants.GateType.INIT1, [], [pbit_addr], input_partitions=outp)
            sim.perform(constants.GateType.NOT, [notpbit_addr], [pbit_addr], input_partitions=outp)

            sim.perform(constants.GateType.NOT, [notpbit_addr], [pbit_addr], input_partitions=inp, output_partitions=outp)

            sim.perform(constants.GateType.INIT1, [], [notpbit_addr], input_partitions=outp)
            sim.perform(constants.GateType.NOT, [pbit_addr], [notpbit_addr], input_partitions=outp)

        inter.free(pbit_addr, partitions)
        inter.free(notpbit_addr, partitions)

        carry_loc = inter.malloc(1, partitions)

        # Shift the carries to the right
        sim.perform(constants.GateType.INIT1, [], [carry_loc], partitions)
        if cin_addr is None:
            sim.perform(constants.GateType.INIT0, [], [carry_loc], partitions[0])
        else:
            if cin_addr == -1:
                sim.perform(constants.GateType.INIT1, [], [carry_loc], partitions[0])
            else:
                sim.perform(constants.GateType.NOT, [notcin_addr], [carry_loc], partitions[0])
                inter.free(notcin_addr, partitions[0])
        sim.perform(constants.GateType.NOT, [notgbit_addr], [carry_loc], partitions[np.arange(0, N - 1, 2)], partitions[np.arange(1, N, 2)])
        sim.perform(constants.GateType.NOT, [notgbit_addr], [carry_loc], partitions[np.arange(1, N - 1, 2)], partitions[np.arange(2, N, 2)])
        if cout_addr is not None:
            # Choose default cout_partition if not explicitly chosen
            cout_partition = partitions[-1] if cout_partition is None else cout_partition
            sim.perform(constants.GateType.INIT1, [], [cout_addr], cout_partition)
            sim.perform(constants.GateType.NOT, [notgbit_addr], [cout_addr], partitions[-1], cout_partition)

        inter.free(notgbit_addr, partitions)

        # Compute the final sum as XOR(x, y, carry_loc)
        ParallelArithmetic.__xnor(sim, x_addr, y_addr, z_addr, inter, partitions)
        ParallelArithmetic.__xnor(sim, z_addr, carry_loc, z_addr, inter, partitions)

        inter.free(carry_loc, partitions)

    @staticmethod
    def fixedSubtraction(sim: simulator.ParallelSimulator, x_addr: int, y_addr: int, z_addr: int,
            inter, partitions=None, cout_addr=None, cout_partition=None):
        """
        Performs a fixed-point subtraction on the given columns. Supports both unsigned and signed numbers.
        :param sim: the simulation environment
        :param x_addr: the intra-partition address of input x (N-bit)
        :param y_addr: the intra-partition address of input y (N-bit)
        :param z_addr: the intra-partition address of output z (N-bit)
        :param inter: addresses for inter. Either np array or IntermediateAllocator. Relevant to "partitions".
        :param partitions: the partitions to operate on
        :param cout_addr: the intra-partition address of optional output carry (1-bit)
        :param cout_partition: the partition address of optional output carry (1-bit). Highest partition by default.
        """

        if partitions is None:
            partitions = np.arange(sim.num_partitions)

        if isinstance(inter, np.ndarray):
            inter = ParallelArithmetic.IntermediateAllocator(inter, partitions)

        noty_addr = inter.malloc(1, partitions)

        sim.perform(constants.GateType.INIT1, [], [noty_addr], partitions)
        sim.perform(constants.GateType.NOT, [y_addr], [noty_addr], partitions)

        ParallelArithmetic.fixedAddition(sim, x_addr, noty_addr, z_addr, inter,
             cin_addr=-1, cout_addr=cout_addr, cout_partition=cout_partition, partitions=partitions)

        inter.free(noty_addr, partitions)

    @staticmethod
    def fixedMultiplication(sim: simulator.ParallelSimulator, x_addr: int, y_addr: int, z_addr: int, w_addr: int,
            inter, partitions=None):
        """
        Performs a fixed-point multiplication on the given columns. Supports only unsigned numbers.
        Note: Can be extended to signed by first performing absolute value on inputs, and then conditionally inverting
            the output if the input signs were different.
        Note: Can be reduced for half-precision multiplication (only lower N bits) by removing the final addition.
        :param sim: the simulation environment
        :param x_addr: the intra-partition address of input x (N-bit)
        :param y_addr: the intra-partition address of input y (N-bit)
        :param z_addr: the intra-partition address of the lower N bits of x * y (N-bit)
        :param w_addr: the intra-partition address of the upper N bits of x * y (N-bit)
        :param inter: addresses for inter. Either np array or IntermediateAllocator.
        :param partitions: the partitions to operate on
        """

        if partitions is None:
            partitions = np.arange(sim.num_partitions)
        N = len(partitions)
        log2_N = ceil(log2(N))

        if isinstance(inter, np.ndarray):
            inter = ParallelArithmetic.IntermediateAllocator(inter, partitions)

        # Pre-compute the NOT of x
        xnot_addr = inter.malloc(1, partitions)
        sim.perform(constants.GateType.INIT1, [], [xnot_addr], partitions)
        sim.perform(constants.GateType.NOT, [x_addr], [xnot_addr], partitions)

        sbit_addr = w_addr  # inter.malloc(1, partitions)
        cbit_addr = inter.malloc(1, partitions)
        sim.perform(constants.GateType.INIT0, [], [sbit_addr], partitions)
        sim.perform(constants.GateType.INIT0, [], [cbit_addr], partitions)

        for k in range(N):

            # Copy y[k] to all partitions using the broadcast technique
            pbit_addr = inter.malloc(1, partitions)
            temp_addr = inter.malloc(1, partitions)
            sim.perform(constants.GateType.INIT1, [], [temp_addr], partitions)
            sim.perform(constants.GateType.INIT1, [], [pbit_addr], partitions)
            sim.perform(constants.GateType.NOT, [y_addr], [temp_addr], partitions[k], partitions[0])
            sim.perform(constants.GateType.NOT, [temp_addr], [pbit_addr], partitions[0])
            for i in range(log2_N):
                sim.perform(constants.GateType.NOT, [pbit_addr], [temp_addr],
                    partitions[np.arange(0, N - (1 << (log2_N - i - 1)), 1 << (log2_N - i))],
                    partitions[np.arange((1 << (log2_N - i - 1)), N, 1 << (log2_N - i))])
                sim.perform(constants.GateType.NOT, [temp_addr], [pbit_addr],
                            partitions[np.arange((1 << (log2_N - i - 1)), N, 1 << (log2_N - i))])
            inter.free(temp_addr, partitions)

            # Compute partial products
            sim.perform(constants.GateType.NOT, [xnot_addr], [pbit_addr], partitions)  # X-MAGIC

            # Compute full-adder
            temps_addr = inter.malloc(4, partitions)
            sim.perform(constants.GateType.INIT1, [], [temps_addr[0]], partitions)
            sim.perform(constants.GateType.NOR, [sbit_addr, cbit_addr], [temps_addr[0]], partitions)
            sim.perform(constants.GateType.INIT1, [], [temps_addr[1]], partitions)
            sim.perform(constants.GateType.NOR, [sbit_addr, temps_addr[0]], [temps_addr[1]], partitions)
            sim.perform(constants.GateType.INIT1, [], [temps_addr[2]], partitions)
            sim.perform(constants.GateType.NOR, [cbit_addr, temps_addr[0]], [temps_addr[2]], partitions)
            sim.perform(constants.GateType.INIT1, [], [temps_addr[3]], partitions)
            sim.perform(constants.GateType.NOR, [temps_addr[1], temps_addr[2]], [temps_addr[3]], partitions)
            sim.perform(constants.GateType.INIT1, [], [temps_addr[1]], partitions)
            sim.perform(constants.GateType.NOR, [pbit_addr, temps_addr[3]], [temps_addr[1]], partitions)
            sim.perform(constants.GateType.INIT1, [], [cbit_addr], partitions)
            sim.perform(constants.GateType.NOR, [temps_addr[0], temps_addr[1]], [cbit_addr], partitions)
            sim.perform(constants.GateType.INIT1, [], [temps_addr[0]], partitions)
            sim.perform(constants.GateType.NOR, [temps_addr[1], temps_addr[3]], [temps_addr[0]], partitions)
            sim.perform(constants.GateType.INIT1, [], [temps_addr[2]], partitions)
            sim.perform(constants.GateType.NOR, [pbit_addr, temps_addr[1]], [temps_addr[2]], partitions)
            # Compute sum of full-adder while shifting
            sim.perform(constants.GateType.INIT1, [], [z_addr], partitions[k])
            sim.perform(constants.GateType.INIT1, [], [sbit_addr], partitions[:N-1])
            sim.perform(constants.GateType.NOR, [temps_addr[0], temps_addr[2]], [z_addr], partitions[0], partitions[k])
            sim.perform(constants.GateType.NOR, [temps_addr[0], temps_addr[2]], [sbit_addr], partitions[np.arange(1, N, 2)], partitions[np.arange(0, N - 1, 2)])
            sim.perform(constants.GateType.NOR, [temps_addr[0], temps_addr[2]], [sbit_addr], partitions[np.arange(2, N, 2)], partitions[np.arange(1, N - 1, 2)])

            inter.free(temps_addr, partitions)

            inter.free(pbit_addr, partitions)

        inter.free(xnot_addr, partitions)

        # Proposed improvement for computation of upper N bits
        ParallelArithmetic.fixedAddition(sim, cbit_addr, sbit_addr, w_addr, inter, partitions)

        # inter.free(sbit_addr, partitions)
        inter.free(cbit_addr, partitions)

    @staticmethod
    def fixedDivision(sim: simulator.ParallelSimulator, w_addr: int, z_addr: int, d_addr: int, q_addr: int, r_addr: int,
            inter, partitions=None):
        """
        Performs a fixed-point division on the given columns. Supports only unsigned numbers.
        Note: Can be extended to signed by first performing absolute value on inputs, and then conditionally inverting
            the output if the input signs were different.
        :param sim: the simulation environment
        :param w_addr: the intra-partition address of input w (upper N bits of dividend)
        :param z_addr: the intra-partition address of input z (lower N bits of dividend)
        :param d_addr: the intra-partition address of input d (N-bit)
        :param q_addr: the intra-partition address of output q (N-bit)
        :param r_addr: the intra-partition address of output r (N-bit)
        :param inter: addresses for inter. Either np array or IntermediateAllocator.
        :param partitions: the partitions to operate on
        """

        if partitions is None:
            partitions = np.arange(sim.num_partitions)
        N = len(partitions)
        log2_N = ceil(log2(N))

        if isinstance(inter, np.ndarray):
            inter = ParallelArithmetic.IntermediateAllocator(inter, partitions)

        # Initialize sum and carry bits

        # Initialize sum to shifted bits of the dividend
        temp_addr = inter.malloc(1, partitions)
        sbit_addr = inter.malloc(1, partitions)
        sim.perform(constants.GateType.INIT1, [], [temp_addr], partitions)
        sim.perform(constants.GateType.INIT1, [], [sbit_addr], partitions)
        sim.perform(constants.GateType.NOT, [z_addr], [temp_addr], partitions[-1], partitions[0])
        sim.perform(constants.GateType.NOT, [w_addr], [temp_addr], partitions[0:N-1:2], partitions[1:N:2])
        sim.perform(constants.GateType.NOT, [w_addr], [temp_addr], partitions[1:N-1:2], partitions[2:N:2])
        sim.perform(constants.GateType.NOT, [temp_addr], [sbit_addr], partitions)
        inter.free(temp_addr, partitions)

        # Initialize carry to zero
        cbit_addr = r_addr  # inter.malloc(1, partitions)
        sim.perform(constants.GateType.INIT0, [], [cbit_addr], partitions[1:])

        # Initialize the separate MSB
        prev_msb_addr = inter.malloc(1, partitions[-3:])  # actually stored in partitions[-1],
        # the other two are used as aligned intermediates for XOR computation
        temp_addr = inter.malloc(1, partitions[-1])
        sim.perform(constants.GateType.INIT1, [], [prev_msb_addr], partitions[-1])
        sim.perform(constants.GateType.INIT1, [], [temp_addr], partitions[-1])
        sim.perform(constants.GateType.NOT, [w_addr], [temp_addr], partitions[-1])
        sim.perform(constants.GateType.NOT, [temp_addr], [prev_msb_addr], partitions[-1])
        inter.free(temp_addr, partitions[-1])

        sim.perform(constants.GateType.INIT1, [], [q_addr], partitions)

        # Perform division iterations
        for i in range(N):

            # Broadcast bit from q to all partitions (into qtag_addr)
            qtag_addr = inter.malloc(1, partitions)
            notqtag_addr = inter.malloc(1, partitions)
            if i > 0:
                sim.perform(constants.GateType.INIT1, [], [qtag_addr], partitions)
                sim.perform(constants.GateType.INIT1, [], [notqtag_addr], partitions)
                sim.perform(constants.GateType.NOT, [q_addr], [notqtag_addr], partitions[N-(i-1)-1], partitions[0])
                sim.perform(constants.GateType.NOT, [notqtag_addr], [qtag_addr], partitions[0])

                # Set q' as carry-in
                # sim.perform(constants.GateType.INIT1, [], [cbit_addr], partitions[0])
                sim.perform(constants.GateType.NOT, [notqtag_addr], [cbit_addr], partitions[0])

                for j in range(log2_N):
                    sim.perform(constants.GateType.NOT, [qtag_addr], [notqtag_addr],
                                partitions[np.arange(0, N - (1 << (log2_N - j - 1)), 1 << (log2_N - j))],
                                partitions[np.arange((1 << (log2_N - j - 1)), N, 1 << (log2_N - j))])
                    sim.perform(constants.GateType.NOT, [notqtag_addr], [qtag_addr],
                                partitions[np.arange((1 << (log2_N - j - 1)), N, 1 << (log2_N - j))])
            else:
                sim.perform(constants.GateType.INIT1, [], [qtag_addr], partitions)
                sim.perform(constants.GateType.INIT0, [], [notqtag_addr], partitions)
                sim.perform(constants.GateType.INIT1, [], [cbit_addr], partitions[0])
            inter.free(notqtag_addr, partitions)

            # Compute xor_addr = XOR(d, q')
            xor_addr = inter.malloc(1, partitions)
            ParallelArithmetic.__xor(sim, qtag_addr, d_addr, xor_addr, inter, partitions, nota_addr=notqtag_addr)

            # Compute full-adder with inputs xor_addr, sbit_addr, and cbit_addr
            # Start storing the partial expressions for the next bit of q in partial_msb_addr
            partial_msb_addr = inter.malloc(1, partitions[-1])
            temps_addr = inter.malloc(4, partitions)
            sim.perform(constants.GateType.INIT1, [], [temps_addr[0]], partitions)
            sim.perform(constants.GateType.NOR, [sbit_addr, cbit_addr], [temps_addr[0]], partitions)
            sim.perform(constants.GateType.INIT1, [], [temps_addr[1]], partitions)
            sim.perform(constants.GateType.NOR, [sbit_addr, temps_addr[0]], [temps_addr[1]], partitions)
            sim.perform(constants.GateType.INIT1, [], [temps_addr[2]], partitions)
            sim.perform(constants.GateType.NOR, [cbit_addr, temps_addr[0]], [temps_addr[2]], partitions)
            sim.perform(constants.GateType.INIT1, [], [temps_addr[3]], partitions)
            sim.perform(constants.GateType.NOR, [temps_addr[1], temps_addr[2]], [temps_addr[3]], partitions)
            sim.perform(constants.GateType.INIT1, [], [temps_addr[1]], partitions)
            sim.perform(constants.GateType.NOR, [xor_addr, temps_addr[3]], [temps_addr[1]], partitions)

            # Compute carry while shifting
            sim.perform(constants.GateType.INIT1, [], [cbit_addr], partitions[1:])
            sim.perform(constants.GateType.INIT0, [], [cbit_addr], partitions[0])
            sim.perform(constants.GateType.INIT1, [], [partial_msb_addr], partitions[-1])
            sim.perform(constants.GateType.NOR, [temps_addr[0], temps_addr[1]], [partial_msb_addr], partitions[-1])
            sim.perform(constants.GateType.NOR, [temps_addr[0], temps_addr[1]], [cbit_addr], partitions[np.arange(0, N - 1, 2)],
                        partitions[np.arange(1, N, 2)])
            sim.perform(constants.GateType.NOR, [temps_addr[0], temps_addr[1]], [cbit_addr], partitions[np.arange(1, N - 1, 2)],
                        partitions[np.arange(2, N, 2)])

            # Compute sum bits
            sim.perform(constants.GateType.INIT1, [], [temps_addr[0]], partitions)
            sim.perform(constants.GateType.NOR, [temps_addr[1], temps_addr[3]], [temps_addr[0]], partitions)
            sim.perform(constants.GateType.INIT1, [], [temps_addr[2]], partitions)
            sim.perform(constants.GateType.NOR, [xor_addr, temps_addr[1]], [temps_addr[2]], partitions)
            sim.perform(constants.GateType.INIT1, [], [sbit_addr], partitions)
            sim.perform(constants.GateType.NOR, [temps_addr[0], temps_addr[2]], [sbit_addr], partitions)

            inter.free(temps_addr, partitions)
            inter.free(xor_addr, partitions)

            # Perform the carry-lookahead operation

            # Pre-compute the propagate and not propagate bits
            pbit_addr = inter.malloc(1, partitions)
            notpbit_addr = inter.malloc(1, partitions)
            ParallelArithmetic.__or(sim, sbit_addr, cbit_addr, pbit_addr, inter, partitions, notz_addr=notpbit_addr)

            # Pre-compute the not generate bits
            notgbit_addr = inter.malloc(1, partitions)
            ParallelArithmetic.__nand(sim, sbit_addr, cbit_addr, notgbit_addr, inter, partitions)

            # Compute the reduction
            for j in range(log2_N):

                inp = partitions[np.flip(N - 1 - np.arange(1 << j, N, 1 << (j + 1)))]
                outp = partitions[np.flip(N - 1 - np.arange(0, N - (1 << j), 1 << (j + 1)))]

                sim.perform(constants.GateType.NOT, [notgbit_addr], [pbit_addr], input_partitions=inp,
                            output_partitions=outp)

                sim.perform(constants.GateType.NOT, [pbit_addr], [notgbit_addr], input_partitions=outp)

                sim.perform(constants.GateType.INIT1, [], [pbit_addr], input_partitions=outp)
                sim.perform(constants.GateType.NOT, [notpbit_addr], [pbit_addr], input_partitions=outp)

                sim.perform(constants.GateType.NOT, [notpbit_addr], [pbit_addr], input_partitions=inp,
                            output_partitions=outp)

                sim.perform(constants.GateType.INIT1, [], [notpbit_addr], input_partitions=outp)
                sim.perform(constants.GateType.NOT, [pbit_addr], [notpbit_addr], input_partitions=outp)

            inter.free(pbit_addr, partitions)
            inter.free(notpbit_addr, partitions)

            # q[N-i-1] = NOT(XOR(NOT(notgbit_addr at partitions[-1]), partial_msb_addr at partitions[-1],
            #           prev_msb_addr at partitions[-1], qtag_addr at partitions[-1]))
            # Also, prev_msb_addr at partitions[-1] = XOR(sbit_addr at partitions[-1], cbit_addr at partitions[-1])
            # Uses partitions[-3], partitions[-2], and partitions[-1] for the computation

            # Start by computing
            # 1. XNOR(NOT(notgbit_addr at partitions[-1]), partial_msb_addr at partitions[-1])
            # 2. XNOR(sbit_addr at partitions[-1], cbit_addr at partitions[-1])
            # 3. XNOR(prev_msb_addr at partitions[-1], qtag_addr at partitions[-1])
            # in parallel, using partitions[-3], partitions[-2], partitions[-1] (respectively)

            xor_a_addr = prev_msb_addr
            xor_b_addr = qtag_addr

            # Setup XNOR(NOT(notgbit_addr at partitions[-1]), partial_msb_addr at partitions[-1])
            sim.perform(constants.GateType.INIT1, [], [xor_a_addr], partitions[-3:-1])
            sim.perform(constants.GateType.NOT, [notgbit_addr], [xor_a_addr], partitions[-1], partitions[-3])
            sim.perform(constants.GateType.INIT1, [], [xor_b_addr], partitions[-3:-1])
            sim.perform(constants.GateType.NOT, [partial_msb_addr], [xor_b_addr], partitions[-1], partitions[-3])

            inter.free(notgbit_addr, partitions)
            inter.free(partial_msb_addr, partitions[-1])

            # Setup XNOR(sbit_addr at partitions[-1], cbit_addr at partitions[-1])
            sim.perform(constants.GateType.NOT, [sbit_addr], [xor_a_addr], partitions[-1], partitions[-2])
            sim.perform(constants.GateType.NOT, [cbit_addr], [xor_b_addr], partitions[-1], partitions[-2])

            # Setup XNOR(prev_msb_addr at partitions[-1], qtag_addr at partitions[-1])
            # No need due to choice of xor_a_addr, xor_b_addr

            # Perform the three XNORs in parallel, store in xor_b_addr
            ParallelArithmetic.__xnor(sim, xor_a_addr, xor_b_addr, xor_b_addr, inter, partitions[-3:])

            # Setup XOR(xor_b_addr of partitions[-1], xor_b_addr of partitions[-3]) in partitions[-1]
            sim.perform(constants.GateType.INIT1, [], [xor_a_addr], partitions[-1])
            sim.perform(constants.GateType.NOT, [xor_b_addr], [xor_a_addr], partitions[-3], partitions[-1])

            # Perform the the XOR, store in q[N-i-1]
            temps_addr = inter.malloc(3, partitions[-1])
            sim.perform(constants.GateType.INIT1, [], [temps_addr[0]], partitions[-1])
            sim.perform(constants.GateType.NOR, [xor_a_addr, xor_b_addr], [temps_addr[0]], partitions[-1])
            sim.perform(constants.GateType.INIT1, [], [temps_addr[1]], partitions[-1])
            sim.perform(constants.GateType.NOR, [xor_a_addr, temps_addr[0]], [temps_addr[1]], partitions[-1])
            sim.perform(constants.GateType.INIT1, [], [temps_addr[2]], partitions[-1])
            sim.perform(constants.GateType.NOR, [xor_b_addr, temps_addr[0]], [temps_addr[2]], partitions[-1])
            # sim.perform(constants.GateType.INIT1, [], [q_addr], partitions[N-i-1])
            sim.perform(constants.GateType.NOR, [temps_addr[1], temps_addr[2]], [q_addr], partitions[-1], partitions[N-i-1])
            inter.free(temps_addr, partitions[-1])

            # Update prev_msb_addr
            sim.perform(constants.GateType.INIT1, [], [prev_msb_addr], partitions[-1])
            sim.perform(constants.GateType.NOT, [xor_b_addr], [prev_msb_addr], partitions[-2], partitions[-1])

            inter.free(qtag_addr, partitions)

            if i < N - 1:

                # Shift the carries
                temp_addr = inter.malloc(1, partitions)
                sim.perform(constants.GateType.INIT1, [], [temp_addr], partitions)
                sim.perform(constants.GateType.NOT, [cbit_addr], [temp_addr], partitions[np.arange(0, N - 1, 2)],
                            partitions[np.arange(1, N, 2)])
                sim.perform(constants.GateType.NOT, [cbit_addr], [temp_addr], partitions[np.arange(1, N - 1, 2)],
                            partitions[np.arange(2, N, 2)])
                sim.perform(constants.GateType.INIT1, [], [cbit_addr], partitions)
                sim.perform(constants.GateType.NOT, [temp_addr], [cbit_addr], partitions[1:])

                # Shift the sums
                sim.perform(constants.GateType.INIT1, [], [temp_addr], partitions)
                sim.perform(constants.GateType.NOT, [sbit_addr], [temp_addr], partitions[np.arange(0, N - 1, 2)],
                            partitions[np.arange(1, N, 2)])
                sim.perform(constants.GateType.NOT, [sbit_addr], [temp_addr], partitions[np.arange(1, N - 1, 2)],
                            partitions[np.arange(2, N, 2)])
                sim.perform(constants.GateType.NOT, [z_addr], [temp_addr], partitions[-(2 + i)], partitions[0])
                sim.perform(constants.GateType.INIT1, [], [sbit_addr], partitions)
                sim.perform(constants.GateType.NOT, [temp_addr], [sbit_addr], partitions)
                inter.free(temp_addr, partitions)

        inter.free(prev_msb_addr, partitions[-3:])

        # Compute R = S + C

        ParallelArithmetic.fixedAddition(sim, sbit_addr, cbit_addr, r_addr, inter, partitions)

        # Compute sbit_addr = AND(d, NOT q0)
        temp_addr = inter.malloc(1, partitions)
        sim.perform(constants.GateType.INIT1, [], [sbit_addr], partitions)
        sim.perform(constants.GateType.INIT1, [], [temp_addr], partitions)
        sim.perform(constants.GateType.NOT, [q_addr], [sbit_addr], partitions[0])
        for j in range(log2_N):
            sim.perform(constants.GateType.NOT, [sbit_addr], [temp_addr],
                        partitions[np.arange(0, N - (1 << (log2_N - j - 1)), 1 << (log2_N - j))],
                        partitions[np.arange((1 << (log2_N - j - 1)), N, 1 << (log2_N - j))])
            sim.perform(constants.GateType.NOT, [temp_addr], [sbit_addr],
                        partitions[np.arange((1 << (log2_N - j - 1)), N, 1 << (log2_N - j))])

        sim.perform(constants.GateType.INIT1, [], [temp_addr], partitions)
        sim.perform(constants.GateType.NOT, [d_addr], [temp_addr], partitions)
        sim.perform(constants.GateType.NOT, [temp_addr], [sbit_addr], partitions)
        inter.free(temp_addr, partitions)

        # Compute R += AND(d, NOT q0)

        ParallelArithmetic.fixedAddition(sim, sbit_addr, r_addr, r_addr, inter, partitions)

        inter.free(sbit_addr, partitions)
        # inter.free(cbit_addr, partitions)

    @staticmethod
    def floatingAdditionUnsignedIEEE(sim: simulator.ParallelSimulator, x_addr: int, y_addr: int, z_addr: int,
            inter, partitions=None):
        """
        Performs a floating-point addition on the given columns. Supports only unsigned numbers.
        Note: Assumes stored (exponent, mantissa), with sizes chosen according to the IEEE standard for
            15-bit, 31-bit, or 63-bit numbers (no sign).
        :param sim: the simulation environment
        :param x_addr: the intra-partition address of input x
        :param y_addr: the intra-partition address of input y
        :param z_addr: the intra-partition address of output z
        :param inter: addresses for inter. Either np array or IntermediateAllocator.
        :param partitions: the partition to operate on. Assumes divided according to (exponent, mantissa).
        """

        if partitions is None:
            partitions = np.arange(sim.num_partitions)
        N = len(partitions)

        Ns, Ne, Nm = constants.getIEEE754Split(N + 1)

        ParallelArithmetic.floatingAdditionUnsigned(sim, Ne, Nm, x_addr, y_addr, z_addr, inter, partitions)

    @staticmethod
    def floatingAdditionUnsigned(sim: simulator.ParallelSimulator, Ne: int, Nm: int, x_addr: int, y_addr: int, z_addr: int,
            inter, partitions=None):
        """
        Performs a floating-point addition on the given columns. Supports only unsigned numbers.
        :param sim: the simulation environment
        :param Ne: the number of exponent bits
        :param Nm: the number of mantissa bits
        :param x_addr: the intra-partition address of input x
        :param y_addr: the intra-partition address of input y
        :param z_addr: the intra-partition address of output z
        :param inter: addresses for inter. Either np array or IntermediateAllocator.
        :param partitions: the partition to operate on. Assumes divided according to (exponent, mantissa).
        """

        if partitions is None:
            partitions = np.arange(sim.num_partitions)

        e_partitions = partitions[:Ne]
        m_partitions = partitions[Ne:]
        N = Ne + Nm
        log2_N = ceil(log2(N))

        if isinstance(inter, np.ndarray):
            inter = ParallelArithmetic.IntermediateAllocator(inter, partitions)

        # Shift xm to the left (to make room for the hidden bit)
        xm_addr = inter.malloc(1, partitions[Ne - 3:])
        temp_addr = inter.malloc(1, partitions[Ne - 1:-1])
        sim.perform(constants.GateType.INIT1, [], [xm_addr], partitions[Ne - 1:])
        sim.perform(constants.GateType.INIT1, [], [temp_addr], partitions[Ne - 1:-1])
        sim.perform(constants.GateType.NOT, [x_addr], [temp_addr], partitions[(Ne - 1) + 1::2], partitions[(Ne - 1):-1:2])
        sim.perform(constants.GateType.NOT, [x_addr], [temp_addr], partitions[(Ne - 1) + 2::2], partitions[(Ne - 1) + 1:-1:2])
        sim.perform(constants.GateType.NOT, [temp_addr], [xm_addr], partitions[Ne - 1:-1])
        inter.free(temp_addr, partitions[Ne - 1:-1])
        # Compute the x hidden bit (1 if the exponent is non-zero)
        ParallelArithmetic.__reduceOR(sim, x_addr, xm_addr, inter, e_partitions, partitions[-1])

        # Shift ym to the left (to make room for the hidden bit)
        ym_addr = inter.malloc(1, partitions[Ne - 3:])
        temp_addr = inter.malloc(1, partitions[Ne - 1:-1])
        sim.perform(constants.GateType.INIT1, [], [ym_addr], partitions[Ne - 1:])
        sim.perform(constants.GateType.INIT1, [], [temp_addr], partitions[Ne - 1:-1])
        sim.perform(constants.GateType.NOT, [y_addr], [temp_addr], partitions[(Ne - 1) + 1::2], partitions[(Ne - 1):-1:2])
        sim.perform(constants.GateType.NOT, [y_addr], [temp_addr], partitions[(Ne - 1) + 2::2], partitions[(Ne - 1) + 1:-1:2])
        sim.perform(constants.GateType.NOT, [temp_addr], [ym_addr], partitions[Ne - 1:-1])
        inter.free(temp_addr, partitions[Ne - 1:-1])
        # Compute the y hidden bit (1 if the exponent is non-zero)
        ParallelArithmetic.__reduceOR(sim, y_addr, ym_addr, inter, e_partitions, partitions[-1])

        # Calculate exponent difference using fixed-point subtraction
        # Store the result in deltaE_addr of partitions[:Ne+1]
        deltaE_addr = inter.malloc(1, partitions[:Ne + 1])
        swap_addr = inter.malloc(1, partitions)
        carry_addr = inter.malloc(1, partitions[0])
        ParallelArithmetic.fixedSubtraction(sim, x_addr, y_addr, deltaE_addr, inter, partitions=partitions[:Ne],
            cout_addr=carry_addr, cout_partition=partitions[0])
        # Compute deltaE[Ne] and swap
        sim.perform(constants.GateType.INIT1, [], [swap_addr], partitions[Ne])
        sim.perform(constants.GateType.NOT, [carry_addr], [swap_addr], partitions[0], partitions[Ne])
        sim.perform(constants.GateType.INIT1, [], [deltaE_addr], partitions[Ne])
        sim.perform(constants.GateType.NOT, [carry_addr], [deltaE_addr], partitions[0], partitions[Ne])
        inter.free(carry_addr, partitions[0])

        # Broadcast swap condition
        notswap_addr = inter.malloc(1, partitions)
        sim.perform(constants.GateType.INIT1, [], [notswap_addr], partitions)
        sim.perform(constants.GateType.NOT, [swap_addr], [notswap_addr], partitions[Ne], partitions[0])
        sim.perform(constants.GateType.INIT1, [], [swap_addr], partitions)
        sim.perform(constants.GateType.NOT, [notswap_addr], [swap_addr], partitions[0])
        for j in range(log2_N):
            sim.perform(constants.GateType.NOT, [swap_addr], [notswap_addr],
                        partitions[np.arange(0, N - (1 << (log2_N - j - 1)), 1 << (log2_N - j))],
                        partitions[np.arange((1 << (log2_N - j - 1)), N, 1 << (log2_N - j))])
            sim.perform(constants.GateType.NOT, [notswap_addr], [swap_addr],
                        partitions[np.arange((1 << (log2_N - j - 1)), N, 1 << (log2_N - j))])

        # Choose the output exponent
        ParallelArithmetic.__mux(sim, swap_addr, y_addr, x_addr, z_addr, inter, e_partitions, nota_addr=notswap_addr)

        # Perform the conditional swap operation
        ParallelArithmetic.__swap(sim, swap_addr, notswap_addr, xm_addr, ym_addr, inter, partitions[Ne - 1:])

        inter.free(swap_addr, partitions)
        inter.free(notswap_addr, partitions)

        # Compute deltaE = abs(deltaE)
        ParallelArithmetic.__abs(sim, deltaE_addr, deltaE_addr, inter, partitions[:Ne + 1])

        # Perform variable shifting

        # If Ne > ceil(log2(Nm + 2)), then if any higher bit is 1, set ym to zero
        if Ne > ceil(log2(Nm + 2)):
            # Compute the OR of the top bits
            ParallelArithmetic.__reduceOR(sim, deltaE_addr, deltaE_addr, inter, e_partitions[ceil(log2(Nm + 2)):], e_partitions[ceil(log2(Nm + 2))])
            # If the OR is one, then zero the mantissa
            for i in range(Nm + 1):
                sim.perform(constants.GateType.NOT, [deltaE_addr], [ym_addr], e_partitions[ceil(log2(Nm + 2))], partitions[Ne - 1 + i])
        sim.perform(constants.GateType.INIT0, [], [ym_addr], partitions[Ne - 3:Ne - 1])
        ParallelArithmetic.__variableShift(sim, ym_addr, deltaE_addr, inter, partitions[Ne - 2:], e_partitions[:ceil(log2(Nm + 2))],
            sticky_addr=ym_addr, sticky_partition=partitions[Ne - 3])

        inter.free(deltaE_addr, partitions[:Ne + 1])

        # Current legend
        # ym_addr of partitions[Ne - 3:] stores (sticky (1 bit), guard (1 bit), ym (Nm + 1 bit))

        # Add the mantissas, store instead of xm
        mantissa_carry_addr = inter.malloc(1, partitions[0])
        ParallelArithmetic.fixedAddition(sim, xm_addr, ym_addr, xm_addr, inter, partitions[Ne-1:],
             cout_addr=mantissa_carry_addr, cout_partition=partitions[0])
        # Copy the sticky/guard bits
        temp_addr = inter.malloc(1, partitions[Ne - 3:Ne - 1])
        sim.perform(constants.GateType.INIT1, [], [temp_addr], partitions[Ne - 3:Ne - 1])
        sim.perform(constants.GateType.NOT, [ym_addr], [temp_addr], partitions[Ne - 3:Ne - 1])
        sim.perform(constants.GateType.INIT1, [], [xm_addr], partitions[Ne - 3:Ne - 1])
        sim.perform(constants.GateType.NOT, [temp_addr], [xm_addr], partitions[Ne - 3:Ne - 1])
        inter.free(temp_addr, partitions[Ne - 3:Ne - 1])

        inter.free(ym_addr, partitions[Ne - 3:])

        # Perform right-shift normalization
        # ParallelArithmetic.__fixedAddBit(sim, z_addr, z_addr, inter, partitions=partitions[:Ne + 1], cin_addr=mantissa_carry_addr, cin_partition=partitions[0]) # performed as part of exponent overflow addition below
        ParallelArithmetic.__variableShift(sim, xm_addr, mantissa_carry_addr, inter, partitions[Ne-2:], partitions[0:1],
            sticky_addr=xm_addr, sticky_partition=partitions[Ne - 3])

        # Perform the round-to-nearest-tie-to-even
        # guard = AND(guard, OR(sticky_addr, zm[0]))
        temps_addr = inter.malloc(2, partitions[Ne - 3:Ne - 1])
        sim.perform(constants.GateType.INIT1, [], [temps_addr[0]], partitions[Ne - 3:Ne - 1])
        sim.perform(constants.GateType.NOT, [xm_addr], [temps_addr[0]], partitions[Ne - 1], partitions[Ne - 3])
        sim.perform(constants.GateType.INIT1, [], [temps_addr[1]], partitions[Ne - 3])
        sim.perform(constants.GateType.NOT, [temps_addr[0]], [temps_addr[1]], partitions[Ne - 3])
        sim.perform(constants.GateType.NOR, [xm_addr, temps_addr[1]], [temps_addr[0]], partitions[Ne - 3], partitions[Ne - 2])
        sim.perform(constants.GateType.NOT, [temps_addr[0]], [xm_addr], partitions[Ne - 2])  # X-MAGIC
        inter.free(temps_addr, partitions[Ne - 3:Ne - 1])
        # Add the rounding correction bit to the mantissa
        # Store the carry-out (whether the rounding caused an overflow since the mantissa was all 1) in xm_addr of partitions[Ne - 2]
        overflow_addr = inter.malloc(1, partitions[0])
        ParallelArithmetic.__fixedAddBit(sim, xm_addr, xm_addr, inter,
             partitions=partitions[Ne - 1:], cin_addr=xm_addr, cin_partition=partitions[Ne - 2],
             cout_addr=overflow_addr, cout_partition=partitions[0])
        # If such overflow occurred, increment the exponent
        # Perform the addition with the addition of the mantissa_carry_addr
        temp_carry_addr = inter.malloc(1, partitions[0])
        ParallelArithmetic.__fullAdder(sim, mantissa_carry_addr, overflow_addr, z_addr, z_addr, temp_carry_addr, inter, partitions[0])
        ParallelArithmetic.__fixedAddBit(sim, z_addr, z_addr, inter, partitions=partitions[1:Ne + 1], cin_addr=temp_carry_addr, cin_partition=partitions[0])
        inter.free(temp_carry_addr, partitions[0])

        inter.free(mantissa_carry_addr, partitions[0])
        inter.free(overflow_addr, partitions[0])

        # Store the mantissa in the final place by shifting one to the right
        temp_addr = inter.malloc(1, m_partitions)
        sim.perform(constants.GateType.INIT1, [], [z_addr], m_partitions)
        sim.perform(constants.GateType.INIT1, [], [temp_addr], m_partitions)
        sim.perform(constants.GateType.NOT, [xm_addr], [temp_addr], partitions[(Ne - 1):-1:2], partitions[(Ne - 1) + 1::2])
        sim.perform(constants.GateType.NOT, [xm_addr], [temp_addr], partitions[(Ne - 1) + 1:-1:2], partitions[(Ne - 1) + 2::2])
        sim.perform(constants.GateType.NOT, [temp_addr], [z_addr], m_partitions)
        inter.free(temp_addr, m_partitions)

        inter.free(xm_addr, partitions[Ne - 3:])

    @staticmethod
    def floatingAdditionSignedIEEE(sim: simulator.ParallelSimulator, x_addr: int, y_addr: int, z_addr: int,
            inter, partitions=None):
        """
        Performs a floating-point addition on the given columns. Supports only signed numbers.
        Note: Assumes stored (sign, exponent, mantissa), with sizes chosen according to the IEEE standard for
            16-bit, 32-bit, or 64-bit numbers.
        :param sim: the simulation environment
        :param x_addr: the intra-partition address of input x
        :param y_addr: the intra-partition address of input y
        :param z_addr: the intra-partition address of output z
        :param inter: addresses for inter. Either np array or IntermediateAllocator.
        :param partitions: the partition to operate on. Assumes divided according to (sign, exponent, mantissa).
        """

        if partitions is None:
            partitions = np.arange(sim.num_partitions)
        N = len(partitions)

        Ns, Ne, Nm = constants.getIEEE754Split(N)

        ParallelArithmetic.floatingAdditionSigned(sim, Ne, Nm, x_addr, y_addr, z_addr, inter, partitions)

    @staticmethod
    def floatingSubtractionSignedIEEE(sim: simulator.ParallelSimulator, x_addr: int, y_addr: int, z_addr: int,
            inter, partitions=None):
        """
        Performs a floating-point subtraction on the given columns. Supports only signed numbers.
        Note: Assumes stored (sign, exponent, mantissa), with sizes chosen according to the IEEE standard for
            16-bit, 32-bit, or 64-bit numbers.
        :param sim: the simulation environment
        :param x_addr: the intra-partition address of input x
        :param y_addr: the intra-partition address of input y
        :param z_addr: the intra-partition address of output z
        :param inter: addresses for inter. Either np array or IntermediateAllocator.
        :param partitions: the partition to operate on. Assumes divided according to (sign, exponent, mantissa).
        """

        if partitions is None:
            partitions = np.arange(sim.num_partitions)
        N = len(partitions)

        Ns, Ne, Nm = constants.getIEEE754Split(N)

        if isinstance(inter, np.ndarray):
            inter = ParallelArithmetic.IntermediateAllocator(inter, partitions)

        # Flip the sign of y
        temps_addr = inter.malloc(2, partitions[0])
        sim.perform(constants.GateType.INIT1, [], [temps_addr[0]], partitions[0])
        sim.perform(constants.GateType.NOT, [y_addr], [temps_addr[0]], partitions[0])
        sim.perform(constants.GateType.INIT1, [], [temps_addr[1]], partitions[0])
        sim.perform(constants.GateType.NOT, [temps_addr[0]], [temps_addr[1]], partitions[0])
        sim.perform(constants.GateType.INIT1, [], [y_addr], partitions[0])
        sim.perform(constants.GateType.NOT, [temps_addr[1]], [y_addr], partitions[0])
        inter.free(temps_addr, partitions[0])

        ParallelArithmetic.floatingAdditionSigned(sim, Ne, Nm, x_addr, y_addr, z_addr, inter, partitions)

        # Flip the sign of y
        temps_addr = inter.malloc(2, partitions[0])
        sim.perform(constants.GateType.INIT1, [], [temps_addr[0]], partitions[0])
        sim.perform(constants.GateType.NOT, [y_addr], [temps_addr[0]], partitions[0])
        sim.perform(constants.GateType.INIT1, [], [temps_addr[1]], partitions[0])
        sim.perform(constants.GateType.NOT, [temps_addr[0]], [temps_addr[1]], partitions[0])
        sim.perform(constants.GateType.INIT1, [], [y_addr], partitions[0])
        sim.perform(constants.GateType.NOT, [temps_addr[1]], [y_addr], partitions[0])
        inter.free(temps_addr, partitions[0])

    @staticmethod
    def floatingAdditionSigned(sim: simulator.ParallelSimulator, Ne: int, Nm: int, x_addr: int, y_addr: int, z_addr: int,
            inter, partitions=None):
        """
        Performs a floating-point addition on the given columns. Supports only signed numbers.
        :param sim: the simulation environment
        :param Ne: the number of exponent bits
        :param Nm: the number of mantissa bits
        :param x_addr: the intra-partition address of input x
        :param y_addr: the intra-partition address of input y
        :param z_addr: the intra-partition address of output z
        :param inter: addresses for inter. Either np array or IntermediateAllocator.
        :param partitions: the partition to operate on. Assumes divided according to (sign, exponent, mantissa).
        """

        if partitions is None:
            partitions = np.arange(sim.num_partitions)

        Ns = 1
        s_partitions = partitions[:Ns]
        e_partitions = partitions[Ns:Ns + Ne]
        m_partitions = partitions[Ns + Ne:Ns + Ne + Nm]
        N = Ns + Ne + Nm
        log2_N = ceil(log2(N))

        if isinstance(inter, np.ndarray):
            inter = ParallelArithmetic.IntermediateAllocator(inter, partitions)

        # Shift xm to the left (to make room for the hidden bit)
        xm_addr = inter.malloc(1, partitions[Ns + Ne - 4:])
        temp_addr = inter.malloc(1, partitions[Ns + Ne - 1:-1])
        sim.perform(constants.GateType.INIT1, [], [xm_addr], partitions[Ns + Ne - 1:])
        sim.perform(constants.GateType.INIT1, [], [temp_addr], partitions[Ns + Ne - 1:-1])
        sim.perform(constants.GateType.NOT, [x_addr], [temp_addr], partitions[(Ns + Ne - 1) + 1::2], partitions[(Ns + Ne - 1):-1:2])
        sim.perform(constants.GateType.NOT, [x_addr], [temp_addr], partitions[(Ns + Ne - 1) + 2::2], partitions[(Ns + Ne - 1) + 1:-1:2])
        sim.perform(constants.GateType.NOT, [temp_addr], [xm_addr], partitions[Ns + Ne - 1:-1])
        inter.free(temp_addr, partitions[Ns + Ne - 1:-1])
        # Compute the x hidden bit (1 if the exponent is non-zero)
        ParallelArithmetic.__reduceOR(sim, x_addr, xm_addr, inter, e_partitions, partitions[-1])

        # Shift ym to the left (to make room for the hidden bit)
        ym_addr = inter.malloc(1, partitions[Ns + Ne - 4:])
        temp_addr = inter.malloc(1, partitions[Ns + Ne - 1:-1])
        sim.perform(constants.GateType.INIT1, [], [ym_addr], partitions[Ns + Ne - 1:])
        sim.perform(constants.GateType.INIT1, [], [temp_addr], partitions[Ns + Ne - 1:-1])
        sim.perform(constants.GateType.NOT, [y_addr], [temp_addr], partitions[(Ns + Ne - 1) + 1::2], partitions[(Ns + Ne - 1):-1:2])
        sim.perform(constants.GateType.NOT, [y_addr], [temp_addr], partitions[(Ns + Ne - 1) + 2::2], partitions[(Ns + Ne - 1) + 1:-1:2])
        sim.perform(constants.GateType.NOT, [temp_addr], [ym_addr], partitions[Ns + Ne - 1:-1])
        inter.free(temp_addr, partitions[Ns + Ne - 1:-1])
        # Compute the y hidden bit (1 if the exponent is non-zero)
        ParallelArithmetic.__reduceOR(sim, y_addr, ym_addr, inter, e_partitions, partitions[-1])

        # Calculate exponent difference using fixed-point subtraction
        # Store the result in deltaE_addr of partitions[Ns:Ns + Ne+1]
        deltaE_addr = inter.malloc(1, partitions[Ns:Ns + Ne + 1])
        swap_addr = inter.malloc(1, partitions)
        carry_addr = inter.malloc(1, partitions[0])
        ParallelArithmetic.fixedSubtraction(sim, x_addr, y_addr, deltaE_addr, inter, partitions=partitions[Ns:Ns+Ne],
            cout_addr=carry_addr, cout_partition=partitions[0])
        # Compute deltaE[Ne] and swap
        sim.perform(constants.GateType.INIT1, [], [swap_addr], partitions[Ns + Ne])
        sim.perform(constants.GateType.NOT, [carry_addr], [swap_addr], partitions[0], partitions[Ns + Ne])
        sim.perform(constants.GateType.INIT1, [], [deltaE_addr], partitions[Ns + Ne])
        sim.perform(constants.GateType.NOT, [carry_addr], [deltaE_addr], partitions[0], partitions[Ns + Ne])
        inter.free(carry_addr, partitions[0])

        # Broadcast swap condition
        notswap_addr = inter.malloc(1, partitions)
        sim.perform(constants.GateType.INIT1, [], [notswap_addr], partitions)
        sim.perform(constants.GateType.NOT, [swap_addr], [notswap_addr], partitions[Ns + Ne], partitions[0])
        sim.perform(constants.GateType.INIT1, [], [swap_addr], partitions)
        sim.perform(constants.GateType.NOT, [notswap_addr], [swap_addr], partitions[0])
        for j in range(log2_N):
            sim.perform(constants.GateType.NOT, [swap_addr], [notswap_addr],
                        partitions[np.arange(0, N - (1 << (log2_N - j - 1)), 1 << (log2_N - j))],
                        partitions[np.arange((1 << (log2_N - j - 1)), N, 1 << (log2_N - j))])
            sim.perform(constants.GateType.NOT, [notswap_addr], [swap_addr],
                        partitions[np.arange((1 << (log2_N - j - 1)), N, 1 << (log2_N - j))])

        # Choose the output exponent
        ParallelArithmetic.__mux(sim, swap_addr, y_addr, x_addr, z_addr, inter, e_partitions, nota_addr=notswap_addr)

        # Perform the conditional swap operation
        ParallelArithmetic.__swap(sim, swap_addr, notswap_addr, xm_addr, ym_addr, inter, partitions[Ns + Ne - 1:])

        # Compute deltaE = abs(deltaE)
        ParallelArithmetic.__abs(sim, deltaE_addr, deltaE_addr, inter, partitions[Ns:Ns + Ne + 1])

        # Perform variable shifting

        # If Ne > ceil(log2(Nm + 3)), then if any higher bit is 1, set ym to zero
        if Ne > ceil(log2(Nm + 3)):
            # Compute the OR of the top bits
            ParallelArithmetic.__reduceOR(sim, deltaE_addr, deltaE_addr, inter, e_partitions[ceil(log2(Nm + 3)):], e_partitions[ceil(log2(Nm + 3))])
            # If the OR is one, then zero the mantissa
            for i in range(Nm + 1):
                sim.perform(constants.GateType.NOT, [deltaE_addr], [ym_addr], e_partitions[ceil(log2(Nm + 3))], partitions[Ns + Ne - 1 + i])
        sim.perform(constants.GateType.INIT0, [], [ym_addr], partitions[Ns + Ne - 4:Ns + Ne - 1])
        ParallelArithmetic.__variableShift(sim, ym_addr, deltaE_addr, inter, partitions[Ns + Ne - 3:], e_partitions[:ceil(log2(Nm + 3))],
            sticky_addr=ym_addr, sticky_partition=partitions[Ns + Ne - 4])

        inter.free(deltaE_addr, partitions[Ns:Ns + Ne + 1])

        # Current legend
        # ym_addr of partitions[Ne - 4:] stores (sticky (1 bit), round (1 bit), guard (1 bit), ym (Nm + 1 bit))

        # Perform XOR on the signs of the inputs
        sdiff_all_addr = inter.malloc(1, partitions)
        sim.perform(constants.GateType.INIT1, [], [sdiff_all_addr], partitions)
        notsdiff_addr = inter.malloc(1, partitions)
        sim.perform(constants.GateType.INIT1, [], [notsdiff_addr], partitions)
        ParallelArithmetic.__xor(sim, x_addr, y_addr, sdiff_all_addr, inter, s_partitions, notz_addr=notsdiff_addr)
        # Broadcast sdiff_addr to all partitions
        for j in range(log2_N):
            sim.perform(constants.GateType.NOT, [sdiff_all_addr], [notsdiff_addr],
                        partitions[np.arange(0, N - (1 << (log2_N - j - 1)), 1 << (log2_N - j))],
                        partitions[np.arange((1 << (log2_N - j - 1)), N, 1 << (log2_N - j))])
            sim.perform(constants.GateType.NOT, [notsdiff_addr], [sdiff_all_addr],
                        partitions[np.arange((1 << (log2_N - j - 1)), N, 1 << (log2_N - j))])

        # Compute zm = z_m' + (y_m' if (x_s == y_s) else -y_m') = x_m' + (y_m' XOR sdiff) + sdiff. Store instead of xm.
        sim.perform(constants.GateType.INIT0, [], [xm_addr], partitions[Ns + Ne - 4: Ns + Ne - 1])
        mantissa_carry_addr = inter.malloc(1, partitions[Ns])
        ParallelArithmetic.__xor(sim, ym_addr, sdiff_all_addr, ym_addr, inter, partitions[Ns + Ne - 4:])
        ParallelArithmetic.fixedAddition(sim, xm_addr, ym_addr, xm_addr, inter, partitions[Ns + Ne - 4:],
             cin_addr=sdiff_all_addr, cin_partition=partitions[-1], cout_addr=mantissa_carry_addr, cout_partition=partitions[Ns])

        # Keep only partitions[0] of sdiff_all_addr (into sdiff_addr)
        inter.free(sdiff_all_addr, partitions)
        sdiff_addr = inter.malloc(1, partitions[0])
        sim.perform(constants.GateType.INIT1, [], [sdiff_addr], partitions[0])
        sim.perform(constants.GateType.NOT, [notsdiff_addr], [sdiff_addr], partitions[0])
        inter.free(notsdiff_addr, partitions)

        inter.free(ym_addr, partitions[Ns + Ne - 4:])

        # If sdiff and not mantissa_carry, then negative_m (if negative_m = 1, then zm is negative);
        # thus, negative_M = sdiff AND (NOT mantissa_carry)) = NOT(notsdiff OR mantissa_carry) = NOR(notsdiff, mantissa_carry)
        negativeM_all_addr = inter.malloc(1, partitions)
        notnegativeM_addr = inter.malloc(1, partitions)
        sim.perform(constants.GateType.INIT1, [], [negativeM_all_addr], partitions)
        sim.perform(constants.GateType.NOR, [notsdiff_addr, mantissa_carry_addr], [negativeM_all_addr], partitions[Ns], partitions[0])
        sim.perform(constants.GateType.INIT1, [], [notnegativeM_addr], partitions)
        sim.perform(constants.GateType.NOT, [negativeM_all_addr], [notnegativeM_addr], partitions[0])
        # Broadcast negative_M to all partitions
        for j in range(log2_N):
            sim.perform(constants.GateType.NOT, [negativeM_all_addr], [notnegativeM_addr],
                        partitions[np.arange(0, N - (1 << (log2_N - j - 1)), 1 << (log2_N - j))],
                        partitions[np.arange((1 << (log2_N - j - 1)), N, 1 << (log2_N - j))])
            sim.perform(constants.GateType.NOT, [notnegativeM_addr], [negativeM_all_addr],
                        partitions[np.arange((1 << (log2_N - j - 1)), N, 1 << (log2_N - j))])

        # Compute the absolute value of zm, figuring out if zm is negative using negative_M
        # That is, set zm = zm XOR negative_M + negative_M
        ParallelArithmetic.__xor(sim, xm_addr, negativeM_all_addr, xm_addr, inter, partitions[Ns + Ne - 4:])
        ParallelArithmetic.__fixedAddBit(sim, xm_addr, xm_addr, inter, partitions=partitions[Ns + Ne - 4:],
            cin_addr=negativeM_all_addr, cin_partition=partitions[-1])

        # Keep only partitions[0] of negativeM_all_addr (into negativeM_addr)
        inter.free(negativeM_all_addr, partitions)
        negativeM_addr = inter.malloc(1, partitions[0])
        sim.perform(constants.GateType.INIT1, [], [negativeM_addr], partitions[0])
        sim.perform(constants.GateType.NOT, [notnegativeM_addr], [negativeM_addr], partitions[0])
        inter.free(notnegativeM_addr, partitions)

        # if diff_signs, then mantissa_carry = False
        sim.perform(constants.GateType.NOT, [sdiff_addr], [mantissa_carry_addr], partitions[0], partitions[Ns])  # X-MAGIC

        # Perform right-shift normalization
        # ParallelArithmetic.__fixedAddBit(sim, z_addr, z_addr, inter, partitions=partitions[Ns:Ns + Ne + 1], cin_addr=mantissa_carry_addr, cin_partition=partitions[Ns]) # performed as part of exponent overflow addition below
        ParallelArithmetic.__variableShift(sim, xm_addr, mantissa_carry_addr, inter, partitions[Ns + Ne - 3:], partitions[Ns:Ns + 1],
            sticky_addr=xm_addr, sticky_partition=partitions[Ns + Ne - 4])
        # OR the MSB of xm with mantissa_carry (as if mantissa_carry was shifted in)
        temps_addr = inter.malloc(2, partitions[-1])
        sim.perform(constants.GateType.INIT1, [], [temps_addr[0]], partitions[-1])
        sim.perform(constants.GateType.NOT, [mantissa_carry_addr], [temps_addr[0]], partitions[Ns], partitions[-1])
        sim.perform(constants.GateType.INIT1, [], [temps_addr[1]], partitions[-1])
        sim.perform(constants.GateType.NOT, [temps_addr[0]], [temps_addr[1]], partitions[-1])
        ParallelArithmetic.__or(sim, temps_addr[1], xm_addr, xm_addr, inter, partitions[-1])
        inter.free(temps_addr, partitions[-1])

        # Add additional bit to z exponent
        zexp_addr = inter.malloc(1, partitions[Ns:Ns + Ne + 1])
        # Copy z to zexp
        temp_addr = inter.malloc(1, partitions[Ns:Ns + Ne + 1])
        sim.perform(constants.GateType.INIT1, [], [temp_addr], partitions[Ns:Ns + Ne + 1])
        sim.perform(constants.GateType.NOT, [z_addr], [temp_addr], partitions[Ns:Ns + Ne])
        sim.perform(constants.GateType.INIT1, [], [zexp_addr], partitions[Ns:Ns + Ne + 1])
        sim.perform(constants.GateType.NOT, [temp_addr], [zexp_addr], partitions[Ns:Ns + Ne + 1])
        inter.free(temp_addr, partitions[Ns:Ns + Ne + 1])

        # Perform left-shift normalization
        left_shift_addr = inter.malloc(1, partitions[Ns:Ns + Ne + 1])
        sim.perform(constants.GateType.INIT0, [], [left_shift_addr], partitions[Ns:Ns + Ne + 1])
        ParallelArithmetic.__normalizeShift(sim, xm_addr, left_shift_addr, inter, partitions[Ns + Ne - 3:], partitions[Ns:Ns+ceil(log2(Nm + 2))], direction=True)
        # Subtract from exponent
        ParallelArithmetic.fixedSubtraction(sim, zexp_addr, left_shift_addr, zexp_addr, inter, partitions[Ns:Ns + Ne + 1])
        inter.free(left_shift_addr, partitions[Ns:Ns + Ne + 1])

        # Perform the round-to-nearest-tie-to-even
        # sticky_addr = OR(round_addr, sticky_addr)
        temps_addr = inter.malloc(2, partitions[Ns + Ne - 4])
        sim.perform(constants.GateType.INIT1, [], [temps_addr[0]], partitions[Ns + Ne - 4])
        sim.perform(constants.GateType.INIT1, [], [temps_addr[1]], partitions[Ns + Ne - 4])
        sim.perform(constants.GateType.NOT, [xm_addr], [temps_addr[0]], partitions[Ns + Ne - 3], partitions[Ns + Ne - 4])
        sim.perform(constants.GateType.NOT, [temps_addr[0]], [temps_addr[1]], partitions[Ns + Ne - 4])
        sim.perform(constants.GateType.INIT1, [], [temps_addr[0]], partitions[Ns + Ne - 4])
        sim.perform(constants.GateType.NOR, [temps_addr[1], xm_addr], [temps_addr[0]], partitions[Ns + Ne - 4])
        sim.perform(constants.GateType.INIT1, [], [xm_addr], partitions[Ns + Ne - 4])
        sim.perform(constants.GateType.NOT, [temps_addr[0]], [xm_addr], partitions[Ns + Ne - 4])
        inter.free(temps_addr, partitions[Ns + Ne - 4])
        # guard = AND(guard, OR(sticky_addr, zm[0]))
        temps_addr = inter.malloc(2, partitions[Ns + Ne - 4:Ns + Ne - 1])
        sim.perform(constants.GateType.INIT1, [], [temps_addr[0]], partitions[Ns + Ne - 4:Ns + Ne - 1])
        sim.perform(constants.GateType.NOT, [xm_addr], [temps_addr[0]], partitions[Ns + Ne - 1], partitions[Ns + Ne - 4])
        sim.perform(constants.GateType.INIT1, [], [temps_addr[1]], partitions[Ns + Ne - 4])
        sim.perform(constants.GateType.NOT, [temps_addr[0]], [temps_addr[1]], partitions[Ns + Ne - 4])
        sim.perform(constants.GateType.NOR, [xm_addr, temps_addr[1]], [temps_addr[0]], partitions[Ns + Ne - 4], partitions[Ns + Ne - 2])
        sim.perform(constants.GateType.NOT, [temps_addr[0]], [xm_addr], partitions[Ns + Ne - 2])  # X-MAGIC
        inter.free(temps_addr, partitions[Ns + Ne - 4:Ns + Ne - 1])
        # Add the rounding correction bit to the mantissa
        # Store the carry-out (whether the rounding caused an overflow since the mantissa was all 1) in xm_addr of partitions[Ne - 2]
        overflow_addr = inter.malloc(1, partitions[Ns])
        ParallelArithmetic.__fixedAddBit(sim, xm_addr, xm_addr, inter,
             partitions=partitions[Ns + Ne - 1:], cin_addr=xm_addr, cin_partition=partitions[Ns + Ne - 2],
             cout_addr=overflow_addr, cout_partition=partitions[Ns])
        # If such overflow occurred, increment the exponent
        # Perform the addition with the addition of the mantissa_carry_addr
        # xm[-1] = OR(xm[-1], overflow)
        temps_addr = inter.malloc(2, partitions[-1])
        sim.perform(constants.GateType.INIT1, [], [temps_addr[0]], partitions[-1])
        sim.perform(constants.GateType.NOT, [overflow_addr], [temps_addr[0]], partitions[Ns], partitions[-1])
        sim.perform(constants.GateType.INIT1, [], [temps_addr[1]], partitions[-1])
        sim.perform(constants.GateType.NOT, [temps_addr[0]], [temps_addr[1]], partitions[-1])
        ParallelArithmetic.__or(sim, temps_addr[1], xm_addr, xm_addr, inter, partitions[-1])
        inter.free(temps_addr, partitions[-1])

        temp_carry_addr = inter.malloc(1, partitions[Ns])
        ParallelArithmetic.__fullAdder(sim, mantissa_carry_addr, overflow_addr, zexp_addr, zexp_addr, temp_carry_addr, inter, partitions[Ns])
        ParallelArithmetic.__fixedAddBit(sim, zexp_addr, zexp_addr, inter, partitions=partitions[Ns + 1:Ns + Ne + 1], cin_addr=temp_carry_addr, cin_partition=partitions[Ns])
        inter.free(temp_carry_addr, partitions[Ns])

        inter.free(mantissa_carry_addr, partitions[Ns])
        inter.free(overflow_addr, partitions[Ns])

        # Store the mantissa in the final place by shifting one to the right
        temp_addr = inter.malloc(1, m_partitions)
        sim.perform(constants.GateType.INIT1, [], [z_addr], m_partitions)
        sim.perform(constants.GateType.INIT1, [], [temp_addr], m_partitions)
        sim.perform(constants.GateType.NOT, [xm_addr], [temp_addr], partitions[(Ne - 1):-1:2], partitions[(Ne - 1) + 1::2])
        sim.perform(constants.GateType.NOT, [xm_addr], [temp_addr], partitions[(Ne - 1) + 1:-1:2], partitions[(Ne - 1) + 2::2])
        sim.perform(constants.GateType.NOT, [temp_addr], [z_addr], m_partitions)
        inter.free(temp_addr, m_partitions)

        # Store the exponent in the final place
        temp_addr = inter.malloc(1, e_partitions)
        sim.perform(constants.GateType.INIT1, [], [temp_addr], e_partitions)
        sim.perform(constants.GateType.NOT, [zexp_addr], [temp_addr], e_partitions)
        sim.perform(constants.GateType.INIT1, [], [z_addr], e_partitions)
        sim.perform(constants.GateType.NOT, [temp_addr], [z_addr], e_partitions)
        inter.free(temp_addr, e_partitions)

        # Computing the final sign

        # Idea: Control flow (before conversion to mux - for reference)
        # if xs == ys:
        #     zs = xs
        # else:
        #     if (NOT xs) AND ys:
        #         zs = negativeM XOR swap
        #     else:
        #         zs = not negativeM XOR swap

        # Data flow. Observations:
        # 1. AND((NOT xs), ys) = NOR(xs, not ys)
        # 2. The top else evaluates to:
        # notNegativeM XOR swap XOR NOR(xs, not ys)
        # 3. Overall, we find:
        # zs = xs if XNOR(xs, ys) else (notNegativeM XOR swap XOR NOR(xs, not ys))  (implemented using mux)

        # Data flow. Implementation:
        xor_addr = inter.malloc(1, s_partitions)

        not_ys_addr = inter.malloc(1, s_partitions)
        sim.perform(constants.GateType.INIT1, [], [not_ys_addr], s_partitions)
        sim.perform(constants.GateType.NOT, [y_addr], [not_ys_addr], s_partitions)

        sim.perform(constants.GateType.INIT1, [], [xor_addr], s_partitions)
        sim.perform(constants.GateType.NOR, [x_addr, not_ys_addr], [xor_addr], s_partitions)

        inter.free(not_ys_addr, s_partitions)

        ParallelArithmetic.__xor(sim, xor_addr, swap_addr, xor_addr, inter, s_partitions, notb_addr=notswap_addr)
        ParallelArithmetic.__xnor(sim, xor_addr, negativeM_addr, xor_addr, inter, s_partitions)
        ParallelArithmetic.__mux(sim, sdiff_addr, xor_addr, x_addr, z_addr, inter, s_partitions)

        inter.free(xor_addr, s_partitions)

        inter.free(swap_addr, partitions)
        inter.free(notswap_addr, partitions)
        inter.free(sdiff_addr, partitions[0])
        inter.free(negativeM_addr, partitions[0])

        # Set the output to zero if z hidden is zero, or the exponent is negative
        # should_zero_addr of partitions[0] = OR(NOT xm[-1], zexp[Ne])
        should_zero_addr = inter.malloc(1, partitions[0])
        temps_addr = inter.malloc(2, partitions[Ns + Ne])
        sim.perform(constants.GateType.INIT1, [], [temps_addr[0]], partitions[Ns + Ne])
        sim.perform(constants.GateType.NOT, [xm_addr], [temps_addr[0]], partitions[-1], partitions[Ns + Ne])
        sim.perform(constants.GateType.INIT1, [], [temps_addr[1]], partitions[Ns + Ne])
        sim.perform(constants.GateType.NOR, [zexp_addr, temps_addr[0]], [temps_addr[1]], partitions[Ns + Ne])
        sim.perform(constants.GateType.INIT1, [], [should_zero_addr], partitions[0])
        sim.perform(constants.GateType.NOT, [temps_addr[1]], [should_zero_addr], partitions[Ns + Ne], partitions[0])
        for z in e_partitions:
            sim.perform(constants.GateType.NOT, [should_zero_addr], [z_addr], partitions[0], z)  # X-MAGIC
        for z in m_partitions:
            sim.perform(constants.GateType.NOT, [should_zero_addr], [z_addr], partitions[0], z)  # X-MAGIC
        sim.perform(constants.GateType.NOT, [should_zero_addr], [z_addr], partitions[0], s_partitions)  # X-MAGIC
        inter.free(should_zero_addr, partitions[0])
        inter.free(temps_addr, partitions[Ns + Ne])

        inter.free(xm_addr, partitions[Ns + Ne - 4:])
        inter.free(zexp_addr, partitions[Ns:Ns + Ne + 1])

    @staticmethod
    def floatingMultiplicationIEEE(sim: simulator.ParallelSimulator, x_addr: int, y_addr: int, z_addr: int,
            inter, partitions=None):
        """
        Performs a floating-point multiplication on the given columns. Supports only signed numbers.
        Note: Assumes stored (sign, exponent, mantissa), with sizes chosen according to the IEEE standard for
            16-bit, 32-bit, or 64-bit numbers.
        :param sim: the simulation environment
        :param x_addr: the intra-partition address of input x
        :param y_addr: the intra-partition address of input y
        :param z_addr: the intra-partition address of output z
        :param inter: addresses for inter. Either np array or IntermediateAllocator.
        :param partitions: the partition to operate on. Assumes divided according to (sign, exponent, mantissa).
        """

        if partitions is None:
            partitions = np.arange(sim.num_partitions)
        N = len(partitions)

        Ns, Ne, Nm = constants.getIEEE754Split(N)

        ParallelArithmetic.floatingMultiplication(sim, Ne, Nm, x_addr, y_addr, z_addr, inter, partitions)

    @staticmethod
    def floatingMultiplication(sim: simulator.ParallelSimulator, Ne: int, Nm: int, x_addr: int, y_addr: int, z_addr: int,
            inter, partitions=None):
        """
        Performs a floating-point multiplication on the given columns. Supports only signed numbers.
        :param sim: the simulation environment
        :param Ne: the number of exponent bits
        :param Nm: the number of mantissa bits
        :param x_addr: the intra-partition address of input x
        :param y_addr: the intra-partition address of input y
        :param z_addr: the intra-partition address of output z
        :param inter: addresses for inter. Either np array or IntermediateAllocator.
        :param partitions: the partition to operate on. Assumes divided according to (sign, exponent, mantissa).
        """

        if partitions is None:
            partitions = np.arange(sim.num_partitions)

        Ns = 1
        s_partitions = partitions[:Ns]
        e_partitions = partitions[Ns:Ns + Ne]
        m_partitions = partitions[Ns + Ne:Ns + Ne + Nm]

        if isinstance(inter, np.ndarray):
            inter = ParallelArithmetic.IntermediateAllocator(inter, partitions)

        # Shift xm to the left (to make room for the hidden bit)
        xm_addr = inter.malloc(1,  partitions[Ns + Ne - 1:])
        temp_addr = inter.malloc(1, partitions[Ns + Ne - 1:-1])
        sim.perform(constants.GateType.INIT1, [], [xm_addr], partitions[Ns + Ne - 1:])
        sim.perform(constants.GateType.INIT1, [], [temp_addr], partitions[Ns + Ne - 1:-1])
        sim.perform(constants.GateType.NOT, [x_addr], [temp_addr], partitions[(Ns + Ne - 1) + 1::2], partitions[(Ns + Ne - 1):-1:2])
        sim.perform(constants.GateType.NOT, [x_addr], [temp_addr], partitions[(Ns + Ne - 1) + 2::2], partitions[(Ns + Ne - 1) + 1:-1:2])
        sim.perform(constants.GateType.NOT, [temp_addr], [xm_addr], partitions[Ns + Ne - 1:-1])
        inter.free(temp_addr, partitions[Ns + Ne - 1:-1])
        # Compute the x hidden bit (1 if the exponent is non-zero)
        ParallelArithmetic.__reduceOR(sim, x_addr, xm_addr, inter, e_partitions, partitions[-1])

        # Shift ym to the left (to make room for the hidden bit)
        ym_addr = inter.malloc(1,  partitions[Ns + Ne - 1:])
        temp_addr = inter.malloc(1, partitions[Ns + Ne - 1:-1])
        sim.perform(constants.GateType.INIT1, [], [ym_addr], partitions[Ns + Ne - 1:])
        sim.perform(constants.GateType.INIT1, [], [temp_addr], partitions[Ns + Ne - 1:-1])
        sim.perform(constants.GateType.NOT, [y_addr], [temp_addr], partitions[(Ns + Ne - 1) + 1::2], partitions[(Ns + Ne - 1):-1:2])
        sim.perform(constants.GateType.NOT, [y_addr], [temp_addr], partitions[(Ns + Ne - 1) + 2::2], partitions[(Ns + Ne - 1) + 1:-1:2])
        sim.perform(constants.GateType.NOT, [temp_addr], [ym_addr], partitions[Ns + Ne - 1:-1])
        inter.free(temp_addr, partitions[Ns + Ne - 1:-1])
        # Compute the y hidden bit (1 if the exponent is non-zero)
        ParallelArithmetic.__reduceOR(sim, y_addr, ym_addr, inter, e_partitions, partitions[-1])

        # Multiply the mantissas (with the hidden bits)
        zm_addr = inter.malloc(1, partitions[Ns + Ne - 1:])
        wm_addr = inter.malloc(1, partitions[Ns + Ne - 4:])
        ParallelArithmetic.fixedMultiplication(sim, xm_addr, ym_addr, zm_addr, wm_addr, inter, partitions[Ns + Ne - 1:])

        inter.free(xm_addr, partitions[Ns + Ne - 1:])
        inter.free(ym_addr, partitions[Ns + Ne - 1:])

        temp_addr = inter.malloc(1, partitions)
        sim.perform(constants.GateType.INIT1, [], [temp_addr], partitions)
        # Store the carry in mantissa_carry_addr of partitions[0]
        mantissa_carry_addr = inter.malloc(1, partitions[0])
        sim.perform(constants.GateType.NOT, [wm_addr], [temp_addr], partitions[-1], partitions[0])
        sim.perform(constants.GateType.INIT1, [], [mantissa_carry_addr], partitions[0])
        sim.perform(constants.GateType.NOT, [temp_addr], [mantissa_carry_addr], partitions[0])
        # Store the next bit from the product in partition Ns + Ne - 2
        sim.perform(constants.GateType.NOT, [zm_addr], [temp_addr], partitions[-1], partitions[Ns + Ne - 2])
        sim.perform(constants.GateType.INIT1, [], [wm_addr], partitions[Ns + Ne - 3:Ns + Ne - 1])
        sim.perform(constants.GateType.NOT, [temp_addr], [wm_addr], partitions[Ns + Ne - 2])
        # Store the guard bit from the product in partition Ns + Ne - 3
        sim.perform(constants.GateType.NOT, [zm_addr], [temp_addr], partitions[-2], partitions[Ns + Ne - 3])
        sim.perform(constants.GateType.NOT, [temp_addr], [wm_addr], partitions[Ns + Ne - 3])
        inter.free(temp_addr, partitions)
        # Store the OR of the other bits (sticky) in partition Ns + Ne - 4
        ParallelArithmetic.__reduceOR(sim, zm_addr, wm_addr, inter, partitions[Ns + Ne - 1:-2], partitions[Ns + Ne - 4])

        inter.free(zm_addr, partitions[Ns + Ne - 1:])

        # Current legend:
        # wm_addr of partitions[Ns + Ne - 4:] stores (sticky_bit (1 bit), guard_bit (1 bit), zm (Nm + 1 bit), carry (1 bit))
        # mantissa_carry_addr of partitions[0] stores the carry bit from the multiplication

        # Perform variable shift according to the carry
        ParallelArithmetic.__variableShift(sim, wm_addr, mantissa_carry_addr, inter,
            partitions[Ns + Ne - 3:], partitions[0], sticky_addr=wm_addr, sticky_partition=partitions[Ns + Ne - 4])

        # Perform the round-to-nearest-tie-to-even
        # guard = AND(guard, OR(sticky_addr, zm[0]))
        temps_addr = inter.malloc(2, partitions[Ns + Ne - 4:Ns + Ne - 2])
        sim.perform(constants.GateType.INIT1, [], [temps_addr[0]], partitions[Ns + Ne - 4:Ns + Ne - 2])
        sim.perform(constants.GateType.NOT, [wm_addr], [temps_addr[0]], partitions[Ns + Ne - 2], partitions[Ns + Ne - 4])
        sim.perform(constants.GateType.INIT1, [], [temps_addr[1]], partitions[Ns + Ne - 4])
        sim.perform(constants.GateType.NOT, [temps_addr[0]], [temps_addr[1]], partitions[Ns + Ne - 4])
        sim.perform(constants.GateType.NOR, [wm_addr, temps_addr[1]], [temps_addr[0]], partitions[Ns + Ne - 4], partitions[Ns + Ne - 3])
        sim.perform(constants.GateType.NOT, [temps_addr[0]], [wm_addr], partitions[Ns + Ne - 3])  # X-MAGIC
        inter.free(temps_addr, partitions[Ns + Ne - 4:Ns + Ne - 2])
        # Add the rounding correction bit to the mantissa
        # Store the carry-out (whether the rounding caused an overflow since the mantissa was all 1) in overflow_addr of partitions[-2]
        overflow_addr = inter.malloc(1, partitions[-2])
        ParallelArithmetic.__fixedAddBit(sim, wm_addr, wm_addr, inter,
            partitions=partitions[Ns + Ne - 2:-1], cin_addr=wm_addr, cin_partition=partitions[Ns + Ne - 3],
            cout_addr=overflow_addr, cout_partition=partitions[-2])
        # Set the hidden_bit to one in case of such overflow
        ParallelArithmetic.__or(sim, overflow_addr, wm_addr, wm_addr, inter, partitions[-2])

        # Store the mantissa in the final place by shifting two to the right
        temp_addr = inter.malloc(1, m_partitions)
        sim.perform(constants.GateType.INIT1, [], [z_addr], m_partitions)
        sim.perform(constants.GateType.INIT1, [], [temp_addr], m_partitions)
        sim.perform(constants.GateType.NOT, [wm_addr], [temp_addr], partitions[(Ns + Ne - 2):-2:3], partitions[(Ns + Ne - 2) + 2::3])
        sim.perform(constants.GateType.NOT, [wm_addr], [temp_addr], partitions[(Ns + Ne - 2) + 1:-2:3], partitions[(Ns + Ne - 2) + 3::3])
        sim.perform(constants.GateType.NOT, [wm_addr], [temp_addr], partitions[(Ns + Ne - 2) + 2:-2:3], partitions[(Ns + Ne - 2) + 4::3])
        sim.perform(constants.GateType.NOT, [temp_addr], [z_addr], m_partitions)
        inter.free(temp_addr, m_partitions)

        # Current legend:
        # z_addr of m_partitions stores the final mantissa
        # mantissa_carry_addr of partitions[0] stores the carry bit from the multiplication
        # overflow_addr of partitions[-2] stores whether the rounding increased the exponent

        # Will temporarily store the z exponent in zexp_addr of partitions[Ns:Ns+Ne+1] (as we require Ne + 1 bits)
        zexp_addr = inter.malloc(1, partitions[Ns:Ns+Ne+1])

        # Write -(1 << Ne - 1) to z exponent.
        sim.perform(constants.GateType.INIT0, [], [zexp_addr], partitions[Ns:Ns + Ne + 1])
        sim.perform(constants.GateType.INIT1, [], [zexp_addr], partitions[Ns:Ns + Ne + 1][0])
        sim.perform(constants.GateType.INIT1, [], [zexp_addr], partitions[Ns:Ns + Ne + 1][-2])
        sim.perform(constants.GateType.INIT1, [], [zexp_addr], partitions[Ns:Ns + Ne + 1][-1])

        # Add carry to the exponent
        # performed as part of addition below
        # ParallelArithmetic.__fixedAddBit(sim, zexp_addr, zexp_addr, inter,
        #     cin_addr=mantissa_carry_addr, cin_partition=partitions[0], partitions=partitions[Ns:Ns+Ne+1])

        # Add rounding correction to the exponent
        # performed as part of addition below
        # ParallelArithmetic.__fixedAddBit(sim, zexp_addr, zexp_addr, inter,
        #     cin_addr=overflow_addr, cin_partition=partitions[-2], partitions=partitions[Ns:Ns+Ne+1])

        # Compute the new exponent
        temps_addr = inter.malloc(2, partitions[:Ns + Ne + 1])
        sim.perform(constants.GateType.INIT0, [], [temps_addr[0]], partitions[Ns + Ne])
        sim.perform(constants.GateType.INIT1, [], [temps_addr[0]], e_partitions)
        sim.perform(constants.GateType.INIT1, [], [temps_addr[1]], e_partitions)
        sim.perform(constants.GateType.NOT, [x_addr], [temps_addr[1]], e_partitions)
        sim.perform(constants.GateType.NOT, [temps_addr[1]], [temps_addr[0]], e_partitions)
        ParallelArithmetic.fixedAddition(sim, zexp_addr, temps_addr[0], zexp_addr, inter,
            partitions=partitions[Ns:Ns + Ne + 1], cin_addr=mantissa_carry_addr, cin_partition=partitions[0])
        sim.perform(constants.GateType.INIT1, [], [temps_addr[0]], e_partitions)
        sim.perform(constants.GateType.INIT1, [], [temps_addr[1]], e_partitions)
        sim.perform(constants.GateType.NOT, [y_addr], [temps_addr[1]], e_partitions)
        sim.perform(constants.GateType.NOT, [temps_addr[1]], [temps_addr[0]], e_partitions)
        ParallelArithmetic.fixedAddition(sim, zexp_addr, temps_addr[0], zexp_addr, inter,
            partitions=partitions[Ns:Ns + Ne + 1], cin_addr=overflow_addr, cin_partition=partitions[-2])
        inter.free(temps_addr, partitions[:Ns + Ne + 1])

        inter.free(overflow_addr, partitions[-2])

        inter.free(mantissa_carry_addr, partitions[0])

        # Compute the sign using XOR
        ParallelArithmetic.__xor(sim, x_addr, y_addr, z_addr, inter, s_partitions)

        # Set the output to zero if one of the inputs was zero, or there was an underflow
        # should_zero_addr of partitions[0] = OR(NOT wm[-1], zexp[Ne])
        should_zero_addr = inter.malloc(1, partitions[0])
        temps_addr = inter.malloc(2, partitions[Ns + Ne])
        sim.perform(constants.GateType.INIT1, [], [temps_addr[0]], partitions[Ns + Ne])
        sim.perform(constants.GateType.NOT, [wm_addr], [temps_addr[0]], partitions[-2], partitions[Ns + Ne])
        sim.perform(constants.GateType.INIT1, [], [temps_addr[1]], partitions[Ns + Ne])
        sim.perform(constants.GateType.NOR, [zexp_addr, temps_addr[0]], [temps_addr[1]], partitions[Ns + Ne])
        sim.perform(constants.GateType.INIT1, [], [should_zero_addr], partitions[0])
        sim.perform(constants.GateType.NOT, [temps_addr[1]], [should_zero_addr], partitions[Ns + Ne], partitions[0])
        for z in partitions[Ns:Ns+Ne+1]:
            sim.perform(constants.GateType.NOT, [should_zero_addr], [zexp_addr], partitions[0], z)  # X-MAGIC
        for z in m_partitions:
            sim.perform(constants.GateType.NOT, [should_zero_addr], [z_addr], partitions[0], z)  # X-MAGIC
        sim.perform(constants.GateType.NOT, [should_zero_addr], [z_addr], partitions[0], s_partitions)  # X-MAGIC
        inter.free(should_zero_addr, partitions[0])
        inter.free(temps_addr, partitions[Ns + Ne])

        inter.free(wm_addr, partitions[Ns + Ne - 4:])

        # Store the exponent in the final place
        temp_addr = inter.malloc(1, e_partitions)
        sim.perform(constants.GateType.INIT1, [], [temp_addr], e_partitions)
        sim.perform(constants.GateType.INIT1, [], [z_addr], e_partitions)
        sim.perform(constants.GateType.NOT, [zexp_addr], [temp_addr], e_partitions)
        sim.perform(constants.GateType.NOT, [temp_addr], [z_addr], e_partitions)
        inter.free(temp_addr, e_partitions)

        inter.free(zexp_addr, partitions[Ns:Ns + Ne + 1])

    @staticmethod
    def floatingDivisionIEEE(sim: simulator.ParallelSimulator, x_addr: int, y_addr: int, z_addr: int,
            inter, partitions=None):
        """
        Performs a floating-point division on the given columns. Supports only signed numbers.
        Note: Assumes stored (sign, exponent, mantissa), with sizes chosen according to the IEEE standard for
            16-bit, 32-bit, or 64-bit numbers.
        :param sim: the simulation environment
        :param x_addr: the intra-partition address of input x
        :param y_addr: the intra-partition address of input y
        :param z_addr: the intra-partition address of output z
        :param inter: addresses for inter. Either np array or IntermediateAllocator.
        :param partitions: the partition to operate on. Assumes divided according to (sign, exponent, mantissa).
        """

        if partitions is None:
            partitions = np.arange(sim.num_partitions)
        N = len(partitions)

        Ns, Ne, Nm = constants.getIEEE754Split(N)

        ParallelArithmetic.floatingDivision(sim, Ne, Nm, x_addr, y_addr, z_addr, inter, partitions)

    @staticmethod
    def floatingDivision(sim: simulator.ParallelSimulator, Ne: int, Nm: int, x_addr: int, y_addr: int, z_addr: int,
            inter, partitions=None):
        """
        Performs a floating-point division on the given columns. Supports only signed numbers.
        :param sim: the simulation environment
        :param Ne: the number of exponent bits
        :param Nm: the number of mantissa bits
        :param x_addr: the intra-partition address of input x
        :param y_addr: the intra-partition address of input y
        :param z_addr: the intra-partition address of output z
        :param inter: addresses for inter. Either np array or IntermediateAllocator.
        :param partitions: the partition to operate on. Assumes divided according to (sign, exponent, mantissa).
        """

        if partitions is None:
            partitions = np.arange(sim.num_partitions)

        Ns = 1
        s_partitions = partitions[:Ns]
        e_partitions = partitions[Ns:Ns + Ne]
        m_partitions = partitions[Ns + Ne:Ns + Ne + Nm]

        if isinstance(inter, np.ndarray):
            inter = ParallelArithmetic.IntermediateAllocator(inter, partitions)

        # Shift xm to the left (to make room for the hidden bit)
        xm_addr = inter.malloc(1, partitions[Ns + Ne - 3:])
        temp_addr = inter.malloc(1, partitions[Ns + Ne - 1:-1])
        sim.perform(constants.GateType.INIT1, [], [xm_addr], partitions[Ns + Ne - 1:])
        sim.perform(constants.GateType.INIT1, [], [temp_addr], partitions[Ns + Ne - 1:-1])
        sim.perform(constants.GateType.NOT, [x_addr], [temp_addr], partitions[(Ns + Ne - 1) + 1::2], partitions[(Ns + Ne - 1):-1:2])
        sim.perform(constants.GateType.NOT, [x_addr], [temp_addr], partitions[(Ns + Ne - 1) + 2::2], partitions[(Ns + Ne - 1) + 1:-1:2])
        sim.perform(constants.GateType.NOT, [temp_addr], [xm_addr], partitions[Ns + Ne - 1:-1])
        inter.free(temp_addr, partitions[Ns + Ne - 1:-1])
        # Compute the x hidden bit (1 if the exponent is non-zero)
        ParallelArithmetic.__reduceOR(sim, x_addr, xm_addr, inter, e_partitions, partitions[-1])

        # Shift ym to the left (to make room for the hidden bit)
        ym_addr = inter.malloc(1, partitions[Ns + Ne - 3:])
        temp_addr = inter.malloc(1, partitions[Ns + Ne - 1:-1])
        sim.perform(constants.GateType.INIT1, [], [ym_addr], partitions[Ns + Ne - 1:])
        sim.perform(constants.GateType.INIT1, [], [temp_addr], partitions[Ns + Ne - 1:-1])
        sim.perform(constants.GateType.NOT, [y_addr], [temp_addr], partitions[(Ns + Ne - 1) + 1::2], partitions[(Ns + Ne - 1):-1:2])
        sim.perform(constants.GateType.NOT, [y_addr], [temp_addr], partitions[(Ns + Ne - 1) + 2::2], partitions[(Ns + Ne - 1) + 1:-1:2])
        sim.perform(constants.GateType.NOT, [temp_addr], [ym_addr], partitions[Ns + Ne - 1:-1])
        inter.free(temp_addr, partitions[Ns + Ne - 1:-1])
        # Compute the y hidden bit (1 if the exponent is non-zero)
        ParallelArithmetic.__reduceOR(sim, y_addr, ym_addr, inter, e_partitions, partitions[-1])

        # Shifts toward the division

        # Shift x three to the left
        temp_addr = inter.malloc(1, partitions[Ns + Ne - 4:])
        sim.perform(constants.GateType.INIT1, [], [temp_addr], partitions[Ns + Ne - 4:])
        sim.perform(constants.GateType.NOT, [xm_addr], [temp_addr], partitions[(Ns + Ne - 4) + 3::4], partitions[(Ns + Ne - 4):-3:4])
        sim.perform(constants.GateType.NOT, [xm_addr], [temp_addr], partitions[(Ns + Ne - 4) + 4::4], partitions[(Ns + Ne - 4) + 1:-3:4])
        sim.perform(constants.GateType.NOT, [xm_addr], [temp_addr], partitions[(Ns + Ne - 4) + 5::4], partitions[(Ns + Ne - 4) + 2:-3:4])
        sim.perform(constants.GateType.NOT, [xm_addr], [temp_addr], partitions[(Ns + Ne - 4) + 6::4], partitions[(Ns + Ne - 4) + 3:-3:4])
        sim.perform(constants.GateType.INIT1, [], [xm_addr], partitions[Ns + Ne - 4:])
        sim.perform(constants.GateType.NOT, [temp_addr], [xm_addr], partitions[Ns + Ne - 4:])
        inter.free(temp_addr, partitions[Ns + Ne - 4:])

        # Move LSB of x to partitions[-1] (as will be part of z)
        temp_addr = inter.malloc(1, partitions[-1])
        zlow_addr = inter.malloc(1, partitions[Ns + Ne - 3:])
        sim.perform(constants.GateType.INIT1, [], [zlow_addr], partitions[-1])
        sim.perform(constants.GateType.INIT1, [], [temp_addr], partitions[-1])
        sim.perform(constants.GateType.INIT0, [], [zlow_addr], partitions[Ns + Ne - 3:-1])
        sim.perform(constants.GateType.NOT, [xm_addr], [temp_addr], partitions[Ns + Ne - 4], partitions[-1])
        sim.perform(constants.GateType.NOT, [temp_addr], [zlow_addr], partitions[-1])
        inter.free(temp_addr, partitions[-1])

        # Shift y twice to the left
        temp_addr = inter.malloc(1, partitions[Ns + Ne - 3:])
        sim.perform(constants.GateType.INIT1, [], [temp_addr], partitions[Ns + Ne - 3:])
        sim.perform(constants.GateType.NOT, [ym_addr], [temp_addr], partitions[(Ns + Ne - 3) + 2::3], partitions[(Ns + Ne - 3):-2:3])
        sim.perform(constants.GateType.NOT, [ym_addr], [temp_addr], partitions[(Ns + Ne - 3) + 3::3], partitions[(Ns + Ne - 3) + 1:-2:3])
        sim.perform(constants.GateType.NOT, [ym_addr], [temp_addr], partitions[(Ns + Ne - 3) + 4::3], partitions[(Ns + Ne - 3) + 2:-2:3])
        sim.perform(constants.GateType.INIT1, [], [ym_addr], partitions[Ns + Ne - 3:])
        sim.perform(constants.GateType.NOT, [temp_addr], [ym_addr], partitions[Ns + Ne - 3:])
        inter.free(temp_addr, partitions[Ns + Ne - 3:])

        # Perform the division
        q_addr = inter.malloc(1, partitions[Ns + Ne - 4:])
        r_addr = inter.malloc(1, partitions[Ns + Ne - 3:])
        ParallelArithmetic.fixedDivision(sim, xm_addr, zlow_addr, ym_addr, q_addr, r_addr, inter, partitions[Ns + Ne - 3:])
        inter.free(xm_addr, partitions[Ns + Ne - 3:])
        inter.free(ym_addr, partitions[Ns + Ne - 3:])
        inter.free(zlow_addr, partitions[Ns + Ne - 3:])
        # Compute the sticky bit
        ParallelArithmetic.__reduceOR(sim, r_addr, q_addr, inter, partitions[Ns + Ne - 3:], partitions[Ns + Ne - 4])
        inter.free(r_addr, partitions[Ns + Ne - 3:])

        # Current legend:
        # q_addr of partitions[Ns + Ne - 4:] stores (sticky bit (1 bit), round_bit (1 bit), guard_bit (1 bit), zm (Nm + 1 bit))

        # Perform left-shift normalization
        norm_addr = inter.malloc(1, partitions[Ns + Ne - 5])
        sim.perform(constants.GateType.INIT1, [], [norm_addr], partitions[Ns + Ne - 5])
        sim.perform(constants.GateType.NOT, [q_addr], [norm_addr], partitions[-1], partitions[Ns + Ne - 5])
        ParallelArithmetic.__variableShift(sim, q_addr, norm_addr, inter, partitions[Ns + Ne - 4:], partitions[Ns + Ne - 5], direction=True)

        # Perform the round-to-nearest-tie-to-even
        # sticky_addr = OR(round_addr, sticky_addr)
        temps_addr = inter.malloc(2, partitions[Ns + Ne - 4])
        sim.perform(constants.GateType.INIT1, [], [temps_addr[0]], partitions[Ns + Ne - 4])
        sim.perform(constants.GateType.INIT1, [], [temps_addr[1]], partitions[Ns + Ne - 4])
        sim.perform(constants.GateType.NOT, [q_addr], [temps_addr[0]], partitions[Ns + Ne - 3], partitions[Ns + Ne - 4])
        sim.perform(constants.GateType.NOT, [temps_addr[0]], [temps_addr[1]], partitions[Ns + Ne - 4])
        sim.perform(constants.GateType.INIT1, [], [temps_addr[0]], partitions[Ns + Ne - 4])
        sim.perform(constants.GateType.NOR, [temps_addr[1], q_addr], [temps_addr[0]], partitions[Ns + Ne - 4])
        sim.perform(constants.GateType.INIT1, [], [q_addr], partitions[Ns + Ne - 4])
        sim.perform(constants.GateType.NOT, [temps_addr[0]], [q_addr], partitions[Ns + Ne - 4])
        inter.free(temps_addr, partitions[Ns + Ne - 4])
        # guard = AND(guard, OR(sticky_addr, zm[0]))
        temps_addr = inter.malloc(2, partitions[Ns + Ne - 4:Ns + Ne - 1])
        sim.perform(constants.GateType.INIT1, [], [temps_addr[0]], partitions[Ns + Ne - 4:Ns + Ne - 1])
        sim.perform(constants.GateType.NOT, [q_addr], [temps_addr[0]], partitions[Ns + Ne - 1], partitions[Ns + Ne - 4])
        sim.perform(constants.GateType.INIT1, [], [temps_addr[1]], partitions[Ns + Ne - 4])
        sim.perform(constants.GateType.NOT, [temps_addr[0]], [temps_addr[1]], partitions[Ns + Ne - 4])
        sim.perform(constants.GateType.NOR, [q_addr, temps_addr[1]], [temps_addr[0]], partitions[Ns + Ne - 4], partitions[Ns + Ne - 2])
        sim.perform(constants.GateType.NOT, [temps_addr[0]], [q_addr], partitions[Ns + Ne - 2])  # X-MAGIC
        inter.free(temps_addr, partitions[Ns + Ne - 4:Ns + Ne - 1])

        # Add the rounding correction bit to the mantissa
        # Store the carry-out (whether the rounding caused an overflow since the mantissa was all 1) in overflow_addr of partitions[-1]
        overflow_addr = inter.malloc(1, partitions[-1])
        ParallelArithmetic.__fixedAddBit(sim, q_addr, q_addr, inter,
            partitions=partitions[Ns + Ne - 1:], cin_addr=q_addr, cin_partition=partitions[Ns + Ne - 2],
            cout_addr=overflow_addr, cout_partition=partitions[-1])

        # Set the hidden_bit to one in case of such overflow
        ParallelArithmetic.__or(sim, overflow_addr, q_addr, q_addr, inter, partitions[-1])

        # Store the final mantissa by shifting once to the right
        temp_addr = inter.malloc(1, m_partitions)
        sim.perform(constants.GateType.INIT1, [], [z_addr], m_partitions)
        sim.perform(constants.GateType.INIT1, [], [temp_addr], m_partitions)
        sim.perform(constants.GateType.NOT, [q_addr], [temp_addr], partitions[(Ns + Ne - 1):-1:2], partitions[(Ns + Ne - 1) + 1::2])
        sim.perform(constants.GateType.NOT, [q_addr], [temp_addr], partitions[(Ns + Ne - 1) + 1:-1:2], partitions[(Ns + Ne - 1) + 2::2])
        sim.perform(constants.GateType.NOT, [temp_addr], [z_addr], m_partitions)
        inter.free(temp_addr, m_partitions)

        # Current legend:
        # z_addr of m_partitions stores the final mantissa
        # norm_addr of partitions[Ns + Ne - 5] stores whether or not left-normalization was performed
        # overflow_addr of partitions[-1] stores whether the rounding increased the exponent

        # Will temporarily store the z exponent in zexp_addr of partitions[Ns:Ns+Ne+1] (as we require Ne + 1 bits)
        zexp_addr = inter.malloc(1, partitions[Ns:Ns + Ne + 1])

        # Write (1 << Ne - 1) to z exponent.
        sim.perform(constants.GateType.INIT1, [], [zexp_addr], partitions[Ns:Ns + Ne - 1])
        sim.perform(constants.GateType.INIT0, [], [zexp_addr], partitions[Ns + Ne - 1:Ns + Ne + 1])
        # Subtract (whether normalization occurred) from the exponent
        # Set zexp[0] to NOT norm to effectively perform the subtraction
        sim.perform(constants.GateType.NOT, [norm_addr], [zexp_addr], partitions[Ns + Ne - 5], partitions[Ns])
        # Add rounding correction to the exponent
        # ParallelArithmetic.__fixedAddBit(sim, zexp_addr, zexp_addr, inter, cin_addr=overflow_addr, cin_partition=partitions[-1],
        #     partitions=partitions[Ns:Ns + Ne + 1])  # performed as part of next addition
        inter.free(norm_addr, partitions[Ns + Ne - 5])

        # Compute the new exponent
        temps_addr = inter.malloc(2, partitions[:Ns + Ne + 1])
        sim.perform(constants.GateType.INIT0, [], [temps_addr[0]], partitions[Ns + Ne])
        sim.perform(constants.GateType.INIT1, [], [temps_addr[0]], e_partitions)
        sim.perform(constants.GateType.INIT1, [], [temps_addr[1]], e_partitions)
        sim.perform(constants.GateType.NOT, [x_addr], [temps_addr[1]], e_partitions)
        sim.perform(constants.GateType.NOT, [temps_addr[1]], [temps_addr[0]], e_partitions)
        ParallelArithmetic.fixedAddition(sim, zexp_addr, temps_addr[0], zexp_addr, inter,
            partitions=partitions[Ns:Ns + Ne + 1], cin_addr=overflow_addr, cin_partition=partitions[-1])
        inter.free(overflow_addr, partitions[-1])
        sim.perform(constants.GateType.INIT1, [], [temps_addr[0]], e_partitions)
        sim.perform(constants.GateType.INIT1, [], [temps_addr[1]], e_partitions)
        sim.perform(constants.GateType.NOT, [y_addr], [temps_addr[1]], e_partitions)
        sim.perform(constants.GateType.NOT, [temps_addr[1]], [temps_addr[0]], e_partitions)
        ParallelArithmetic.fixedSubtraction(sim, zexp_addr, temps_addr[0], zexp_addr, inter,
            partitions=partitions[Ns:Ns + Ne + 1])
        inter.free(temps_addr, partitions[:Ns + Ne + 1])

        # Compute the sign using XOR
        ParallelArithmetic.__xor(sim, x_addr, y_addr, z_addr, inter, s_partitions)

        # Set the output to zero if one of the inputs was zero, or there was an underflow
        # should_zero_addr of partitions[0] = OR(NOT q[-1], zexp[Ne])
        should_zero_addr = inter.malloc(1, partitions[0])
        temps_addr = inter.malloc(2, partitions[Ns + Ne])
        sim.perform(constants.GateType.INIT1, [], [temps_addr[0]], partitions[Ns + Ne])
        sim.perform(constants.GateType.NOT, [q_addr], [temps_addr[0]], partitions[-1], partitions[Ns + Ne])
        sim.perform(constants.GateType.INIT1, [], [temps_addr[1]], partitions[Ns + Ne])
        sim.perform(constants.GateType.NOR, [zexp_addr, temps_addr[0]], [temps_addr[1]], partitions[Ns + Ne])
        sim.perform(constants.GateType.INIT1, [], [should_zero_addr], partitions[0])
        sim.perform(constants.GateType.NOT, [temps_addr[1]], [should_zero_addr], partitions[Ns + Ne], partitions[0])
        for z in partitions[Ns:Ns+Ne+1]:
            sim.perform(constants.GateType.NOT, [should_zero_addr], [zexp_addr], partitions[0], z)  # X-MAGIC
        for z in m_partitions:
            sim.perform(constants.GateType.NOT, [should_zero_addr], [z_addr], partitions[0], z)  # X-MAGIC
        sim.perform(constants.GateType.NOT, [should_zero_addr], [z_addr], partitions[0], s_partitions)  # X-MAGIC
        inter.free(should_zero_addr, partitions[0])
        inter.free(temps_addr, partitions[Ns + Ne])

        inter.free(q_addr, partitions[Ns + Ne - 4:])

        # Store the exponent in the final place
        temp_addr = inter.malloc(1, e_partitions)
        sim.perform(constants.GateType.INIT1, [], [temp_addr], e_partitions)
        sim.perform(constants.GateType.INIT1, [], [z_addr], e_partitions)
        sim.perform(constants.GateType.NOT, [zexp_addr], [temp_addr], e_partitions)
        sim.perform(constants.GateType.NOT, [temp_addr], [z_addr], e_partitions)
        inter.free(temp_addr, e_partitions)

        inter.free(zexp_addr, partitions[Ns:Ns + Ne + 1])

    @staticmethod
    def __and(sim: simulator.ParallelSimulator, a_addr: int, b_addr: int, z_addr: int, inter, partitions,
            nota_addr=None, notb_addr=None, notz_addr=None):
        """
        Performs z = AND(a, b) on the given columns (in all partitions in parallel)
        :param sim: the simulation environment
        :param a_addr: the intra-partition index of input a
        :param b_addr: the intra-partition index of input b
        :param z_addr: the intra-partition index of output z
        :param inter: addresses for inter. Either np array or IntermediateAllocator.
        :param partitions: the independent partitions to operate on
        :param nota_addr: the index of the optional input which stores the not of a
        :param notb_addr: the index of the optional input which stores the not of b
        :param notz_addr: the index of the optional output which stores the not of z
        """

        if isinstance(inter, np.ndarray):
            inter = ParallelArithmetic.IntermediateAllocator(inter, partitions)

        computed_not_a = nota_addr is None
        if computed_not_a:
            nota_addr = inter.malloc(1, partitions)
            sim.perform(constants.GateType.INIT1, [], [nota_addr], partitions)
            sim.perform(constants.GateType.NOT, [a_addr], [nota_addr], partitions)

        computed_not_b = notb_addr is None
        if computed_not_b:
            notb_addr = inter.malloc(1, partitions)
            sim.perform(constants.GateType.INIT1, [], [notb_addr], partitions)
            sim.perform(constants.GateType.NOT, [b_addr], [notb_addr], partitions)

        sim.perform(constants.GateType.INIT1, [], [z_addr], partitions)
        sim.perform(constants.GateType.NOR, [nota_addr, notb_addr], [z_addr], partitions)

        if notz_addr is not None:
            sim.perform(constants.GateType.INIT1, [], [notz_addr], partitions)
            sim.perform(constants.GateType.NOT, [z_addr], [notz_addr], partitions)

        if computed_not_a:
            inter.free(nota_addr, partitions)
        if computed_not_b:
            inter.free(notb_addr, partitions)

    @staticmethod
    def __nand(sim: simulator.ParallelSimulator, a_addr: int, b_addr: int, z_addr: int, inter, partitions,
            nota_addr=None, notb_addr=None, notz_addr=None):
        """
        Performs z = NAND(a, b) on the given columns (in all partitions in parallel)
        :param sim: the simulation environment
        :param a_addr: the intra-partition index of input a
        :param b_addr: the intra-partition index of input a
        :param z_addr: the intra-partition index of output z
        :param inter: addresses for inter. Either np array or IntermediateAllocator.
        :param partitions: the independent partitions to operate on
        :param nota_addr: the index of the optional input which stores the not of a
        :param notb_addr: the index of the optional input which stores the not of b
        :param notz_addr: the index of the optional output which stores the not of z
        """

        if isinstance(inter, np.ndarray):
            inter = ParallelArithmetic.IntermediateAllocator(inter, partitions)

        computed_not_a = nota_addr is None
        if computed_not_a:
            nota_addr = z_addr  # inter.malloc(1, partitions)  (use output cell)
            sim.perform(constants.GateType.INIT1, [], [nota_addr], partitions)
            sim.perform(constants.GateType.NOT, [a_addr], [nota_addr], partitions)

        computed_not_b = notb_addr is None
        if computed_not_b:
            notb_addr = inter.malloc(1, partitions)
            sim.perform(constants.GateType.INIT1, [], [notb_addr], partitions)
            sim.perform(constants.GateType.NOT, [b_addr], [notb_addr], partitions)

        allocated_not_z = notz_addr is None
        if allocated_not_z:
            notz_addr = inter.malloc(1, partitions)

        sim.perform(constants.GateType.INIT1, [], [notz_addr], partitions)
        sim.perform(constants.GateType.NOR, [nota_addr, notb_addr], [notz_addr], partitions)
        sim.perform(constants.GateType.INIT1, [], [z_addr], partitions)
        sim.perform(constants.GateType.NOT, [notz_addr], [z_addr], partitions)

        # if computed_not_a:
        #     inter.free(nota_addr, partitions)
        if computed_not_b:
            inter.free(notb_addr, partitions)
        if allocated_not_z:
            inter.free(notz_addr, partitions)

    @staticmethod
    def __or(sim: simulator.ParallelSimulator, a_addr: int, b_addr: int, z_addr: int, inter, partitions,
            nota_addr=None, notb_addr=None, notz_addr=None):
        """
        Performs z = OR(a, b) on the given columns (in all partitions in parallel)
        :param sim: the simulation environment
        :param a_addr: the intra-partition index of input a
        :param b_addr: the intra-partition index of input a
        :param z_addr: the intra-partition index of output z
        :param inter: addresses for inter. Either np array or IntermediateAllocator.
        :param partitions: the independent partitions to operate on
        :param nota_addr: the index of the optional input which stores the not of a
        :param notb_addr: the index of the optional input which stores the not of b
        :param notz_addr: the index of the optional output which stores the not of z
        """

        if isinstance(inter, np.ndarray):
            inter = ParallelArithmetic.IntermediateAllocator(inter, partitions)

        allocated_not_z = notz_addr is None
        if allocated_not_z:
            notz_addr = inter.malloc(1, partitions)

        sim.perform(constants.GateType.INIT1, [], [notz_addr], partitions)
        sim.perform(constants.GateType.NOR, [a_addr, b_addr], [notz_addr], partitions)
        sim.perform(constants.GateType.INIT1, [], [z_addr], partitions)
        sim.perform(constants.GateType.NOT, [notz_addr], [z_addr], partitions)

        if allocated_not_z:
            inter.free(notz_addr, partitions)

    @staticmethod
    def __xor(sim: simulator.ParallelSimulator, a_addr: int, b_addr: int, z_addr: int, inter, partitions,
            nota_addr=None, notb_addr=None, notz_addr=None):
        """
        Performs z = XOR(a, b) on the given columns (in all partitions in parallel)
        :param sim: the simulation environment
        :param a_addr: the intra-partition index of input a
        :param b_addr: the intra-partition index of input a
        :param z_addr: the intra-partition index of output z
        :param inter: addresses for inter. Either np array or IntermediateAllocator.
        :param partitions: the independent partitions to operate on
        :param nota_addr: the index of the optional input which stores the not of a
        :param notb_addr: the index of the optional input which stores the not of b
        :param notz_addr: the index of the optional output which stores the not of z
        """

        if isinstance(inter, np.ndarray):
            inter = ParallelArithmetic.IntermediateAllocator(inter, partitions)

        computed_not_a = nota_addr is None
        if computed_not_a:
            nota_addr = inter.malloc(1, partitions)
            sim.perform(constants.GateType.INIT1, [], [nota_addr], partitions)
            sim.perform(constants.GateType.NOT, [a_addr], [nota_addr], partitions)

        computed_not_b = notb_addr is None
        if computed_not_b:
            notb_addr = inter.malloc(1, partitions)
            sim.perform(constants.GateType.INIT1, [], [notb_addr], partitions)
            sim.perform(constants.GateType.NOT, [b_addr], [notb_addr], partitions)

        t_addr = inter.malloc(2, partitions)

        sim.perform(constants.GateType.INIT1, [], [t_addr[0]], partitions)
        sim.perform(constants.GateType.NOR, [a_addr, b_addr], [t_addr[0]], partitions)
        sim.perform(constants.GateType.INIT1, [], [t_addr[1]], partitions)
        sim.perform(constants.GateType.NOR, [nota_addr, notb_addr], [t_addr[1]], partitions)
        sim.perform(constants.GateType.INIT1, [], [z_addr], partitions)
        sim.perform(constants.GateType.NOR, [t_addr[0], t_addr[1]], [z_addr], partitions)

        inter.free(t_addr, partitions)

        if notz_addr is not None:
            sim.perform(constants.GateType.INIT1, [], [notz_addr], partitions)
            sim.perform(constants.GateType.NOT, [z_addr], [notz_addr], partitions)

        if computed_not_a:
            inter.free(nota_addr, partitions)
        if computed_not_b:
            inter.free(notb_addr, partitions)

    @staticmethod
    def __xnor(sim: simulator.ParallelSimulator, a_addr: int, b_addr: int, z_addr: int, inter, partitions,
              nota_addr=None, notb_addr=None, notz_addr=None):
        """
        Performs z = XNOR(a, b) on the given columns (in all partitions in parallel)
        :param sim: the simulation environment
        :param a_addr: the intra-partition index of input a
        :param b_addr: the intra-partition index of input b
        :param z_addr: the intra-partition index of output z
        :param inter: addresses for inter. Either np array or IntermediateAllocator.
        :param partitions: the independent partitions to operate on
        :param nota_addr: the index of the optional input which stores the not of a
        :param notb_addr: the index of the optional input which stores the not of b
        :param notz_addr: the index of the optional output which stores the not of z
        """

        if isinstance(inter, np.ndarray):
            inter = ParallelArithmetic.IntermediateAllocator(inter, partitions)

        t_addr = inter.malloc(3, partitions)

        sim.perform(constants.GateType.INIT1, [], [t_addr[0]], partitions)
        sim.perform(constants.GateType.NOR, [a_addr, b_addr], [t_addr[0]], partitions)
        sim.perform(constants.GateType.INIT1, [], [t_addr[1]], partitions)
        sim.perform(constants.GateType.NOR, [t_addr[0], a_addr], [t_addr[1]], partitions)
        sim.perform(constants.GateType.INIT1, [], [t_addr[2]], partitions)
        sim.perform(constants.GateType.NOR, [t_addr[0], b_addr], [t_addr[2]], partitions)
        sim.perform(constants.GateType.INIT1, [], [z_addr], partitions)
        sim.perform(constants.GateType.NOR, [t_addr[1], t_addr[2]], [z_addr], partitions)

        inter.free(t_addr, partitions)

        if notz_addr is not None:
            sim.perform(constants.GateType.INIT1, [], [notz_addr], partitions)
            sim.perform(constants.GateType.NOT, [z_addr], [notz_addr], partitions)

    @staticmethod
    def __fullAdder(sim: simulator.ParallelSimulator, a_addr: int, b_addr: int, c_addr: int, s_addr: int, cout_addr: int,
            inter, partitions):
        """
        Performs a full-adder on the given columns (in all partitions in parallel)
        :param sim: the simulation environment
        :param a_addr: the intra-partition index of input a
        :param b_addr: the intra-partition index of input b
        :param c_addr: the intra-partition index of input c
        :param s_addr: the intra-partition index of output s
        :param cout_addr: the intra-partition index of output cout
        :param inter: addresses for inter. Either np array or IntermediateAllocator.
        :param partitions: the independent partitions to operate on
        """

        if isinstance(inter, np.ndarray):
            inter = ParallelArithmetic.IntermediateAllocator(inter, partitions)

        t_addr = inter.malloc(4, partitions)

        sim.perform(constants.GateType.INIT1, [], [t_addr[0]], partitions)
        sim.perform(constants.GateType.NOR, [a_addr, b_addr], [t_addr[0]], partitions)

        sim.perform(constants.GateType.INIT1, [], [t_addr[1]], partitions)
        sim.perform(constants.GateType.NOR, [a_addr, t_addr[0]], [t_addr[1]], partitions)

        sim.perform(constants.GateType.INIT1, [], [t_addr[2]], partitions)
        sim.perform(constants.GateType.NOR, [b_addr, t_addr[0]], [t_addr[2]], partitions)

        sim.perform(constants.GateType.INIT1, [], [t_addr[3]], partitions)
        sim.perform(constants.GateType.NOR, [t_addr[1], t_addr[2]], [t_addr[3]], partitions)

        sim.perform(constants.GateType.INIT1, [], [t_addr[1]], partitions)
        sim.perform(constants.GateType.NOR, [t_addr[3], c_addr], [t_addr[1]], partitions)

        sim.perform(constants.GateType.INIT1, [], [t_addr[2]], partitions)
        sim.perform(constants.GateType.NOR, [t_addr[1], t_addr[3]], [t_addr[2]], partitions)

        sim.perform(constants.GateType.INIT1, [], [t_addr[3]], partitions)
        sim.perform(constants.GateType.NOR, [t_addr[1], c_addr], [t_addr[3]], partitions)

        sim.perform(constants.GateType.INIT1, [], [s_addr], partitions)
        sim.perform(constants.GateType.NOR, [t_addr[3], t_addr[2]], [s_addr], partitions)

        sim.perform(constants.GateType.INIT1, [], [cout_addr], partitions)
        sim.perform(constants.GateType.NOR, [t_addr[0], t_addr[1]], [cout_addr], partitions)

        inter.free(t_addr, partitions)

    @staticmethod
    def __mux(sim: simulator.ParallelSimulator, a_addr: int, b_addr: int, c_addr: int, z_addr: int, inter, partitions,
            nota_addr=None):
        """
        Performs a mux_a(b,c) on the given columns (in all partitions in parallel)
        :param sim: the simulation environment
        :param a_addr: the index of input a (the condition)
        :param b_addr: the index of input b (if a if true)
        :param c_addr: the index of input c (if a is false)
        :param z_addr: the index of the output
        :param inter: addresses for inter. Either np array or IntermediateAllocator.
        :param partitions: the independent partitions to operate on
        :param nota_addr: the index of the optional input which stores the not of a
        """

        if isinstance(inter, np.ndarray):
            inter = ParallelArithmetic.IntermediateAllocator(inter, partitions)

        computed_not_a = nota_addr is None
        if computed_not_a:
            nota_addr = inter.malloc(1, partitions)
            sim.perform(constants.GateType.INIT1, [], [nota_addr], partitions)
            sim.perform(constants.GateType.NOT, [a_addr], [nota_addr], partitions)

        temps_addr = inter.malloc(2, partitions)

        sim.perform(constants.GateType.INIT1, [], [temps_addr[0]], partitions)
        sim.perform(constants.GateType.NOR, [b_addr, nota_addr], [temps_addr[0]], partitions)
        sim.perform(constants.GateType.INIT1, [], [temps_addr[1]], partitions)
        sim.perform(constants.GateType.NOR, [c_addr, a_addr], [temps_addr[1]], partitions)
        sim.perform(constants.GateType.INIT1, [], [z_addr], partitions)
        sim.perform(constants.GateType.NOR, [temps_addr[0], temps_addr[1]], [z_addr], partitions)

        inter.free(temps_addr, partitions)

        if computed_not_a:
            inter.free(nota_addr, partitions)

    @staticmethod
    def __swap(sim: simulator.ParallelSimulator, a_addr: int, nota_addr: int, b_addr: int, c_addr: int, inter, partitions):
        """
        Performs a conditional swap on the given columns (in all partitions in parallel). If a is 1, then b and c swap.
        :param sim: the simulation environment
        :param a_addr: the index of input a (the condition)
        :param nota_addr: precomputed value for the not of a
        :param b_addr: the index of input b
        :param c_addr: the index of input c
        :param inter: addresses for inter. Either np array or IntermediateAllocator.
        :param partitions: the independent partitions to operate on
        """

        if isinstance(inter, np.ndarray):
            inter = ParallelArithmetic.IntermediateAllocator(inter, partitions)

        temps_addr = inter.malloc(4, partitions)

        sim.perform(constants.GateType.INIT1, [], [temps_addr[0]], partitions)
        sim.perform(constants.GateType.NOR, [b_addr, nota_addr], [temps_addr[0]], partitions)
        sim.perform(constants.GateType.INIT1, [], [temps_addr[1]], partitions)
        sim.perform(constants.GateType.NOR, [c_addr, a_addr], [temps_addr[1]], partitions)

        sim.perform(constants.GateType.INIT1, [], [temps_addr[2]], partitions)
        sim.perform(constants.GateType.NOR, [b_addr, a_addr], [temps_addr[2]], partitions)
        sim.perform(constants.GateType.INIT1, [], [temps_addr[3]], partitions)
        sim.perform(constants.GateType.NOR, [c_addr, nota_addr], [temps_addr[3]], partitions)

        sim.perform(constants.GateType.INIT1, [], [c_addr], partitions)
        sim.perform(constants.GateType.NOR, [temps_addr[0], temps_addr[1]], [c_addr], partitions)

        sim.perform(constants.GateType.INIT1, [], [b_addr], partitions)
        sim.perform(constants.GateType.NOR, [temps_addr[2], temps_addr[3]], [b_addr], partitions)

        inter.free(temps_addr, partitions)

    @staticmethod
    def __variableShift(sim: simulator.ParallelSimulator, x_addr: int, t_addr: int, inter,
            x_partitions, t_partitions, sticky_addr=None, sticky_partition=None, direction=False):
        """
        Performs the in-place variable shift operation on the given columns
        :param sim: the simulation environment
        :param x_addr: the intra-partition address of input & output x
        :param t_addr: the intra-partition address of input t
        :param inter: addresses for inter. Either np array or IntermediateAllocator. Relevant to "x_partitions".
        :param x_partitions: the partitions for input x
        :param t_partitions: the partitions for input t
        :param sticky_addr: intra-partition address for an optional sticky bit (OR of all of the bits that were truncated).
        :param sticky_partition: partition for the optional sticky bit
        :param direction: the direction of the shift. False is right-shift, and True is left-shift.
        """

        if isinstance(x_partitions, np.int64):
            x_partitions = np.array([x_partitions], dtype=int)
        if isinstance(t_partitions, np.int64):
            t_partitions = np.array([t_partitions], dtype=int)

        Nx = len(x_partitions)
        log2_Nx = ceil(log2(Nx))
        Nt = len(t_partitions)
        assert(Nt <= log2_Nx)

        if direction:
            x_partitions = np.flip(x_partitions)

        if isinstance(inter, np.ndarray):
            inter = ParallelArithmetic.IntermediateAllocator(inter, x_partitions)

        local_sticky_bit_addr = None
        if sticky_addr is not None:
            # Stores the current sticky (moved to given output sticky address in end)
            local_sticky_bit_addr = inter.malloc(1, x_partitions[0])
            sim.perform(constants.GateType.INIT0, [], [local_sticky_bit_addr], x_partitions[0])

        for j in range(Nt):

            tj = inter.malloc(1, x_partitions)
            not_tj = inter.malloc(1, x_partitions)
            sim.perform(constants.GateType.INIT1, [], [tj], x_partitions)
            sim.perform(constants.GateType.INIT1, [], [not_tj], x_partitions)
            # Broadcast tj and not tj to all partitions
            sim.perform(constants.GateType.NOT, [t_addr], [not_tj], t_partitions[j], x_partitions[0])
            sim.perform(constants.GateType.NOT, [not_tj], [tj], x_partitions[0])
            for k in range(log2_Nx):
                sim.perform(constants.GateType.NOT, [tj], [not_tj],
                            x_partitions[np.arange(0, Nx - (1 << (log2_Nx - k - 1)), 1 << (log2_Nx - k))],
                            x_partitions[np.arange((1 << (log2_Nx - k - 1)), Nx, 1 << (log2_Nx - k))])
                sim.perform(constants.GateType.NOT, [not_tj], [tj],
                            x_partitions[np.arange((1 << (log2_Nx - k - 1)), Nx, 1 << (log2_Nx - k))])

            if sticky_addr is not None:

                # Compute the OR of the bits that are potentially lost in this step
                or_addr = inter.malloc(1, x_partitions[0])
                ParallelArithmetic.__reduceOR(sim, x_addr, or_addr, inter, x_partitions[:2**j], x_partitions[0])
                # Compute the AND with whether the shift actually occurs
                sim.perform(constants.GateType.NOT, [not_tj], [or_addr], x_partitions[0])  # X-MAGIC
                # Compute the OR with the current sticky bit
                temp_addr = inter.malloc(1, x_partitions[0])
                sim.perform(constants.GateType.INIT1, [], [temp_addr], x_partitions[0])
                sim.perform(constants.GateType.NOR, [local_sticky_bit_addr, or_addr], [temp_addr], x_partitions[0])
                sim.perform(constants.GateType.INIT1, [], [local_sticky_bit_addr], x_partitions[0])
                sim.perform(constants.GateType.NOT, [temp_addr], [local_sticky_bit_addr], x_partitions[0])
                inter.free(or_addr, x_partitions[0])
                inter.free(temp_addr, x_partitions[0])

            # Perform a shift of 2 ** j to the right
            amnt = 2 ** j
            temps_addr = inter.malloc(2, x_partitions)
            sim.perform(constants.GateType.INIT1, [], [temps_addr[0]], x_partitions)
            sim.perform(constants.GateType.NOT, [x_addr], [temps_addr[0]], x_partitions)
            sim.perform(constants.GateType.INIT1, [], [temps_addr[1]], x_partitions[:Nx-amnt])
            sim.perform(constants.GateType.INIT0, [], [temps_addr[1]], x_partitions[Nx - amnt:])
            for lm in range(amnt + 1):
                if len(x_partitions[lm + amnt:Nx:amnt+1]) > 0:
                    sim.perform(constants.GateType.NOT, [temps_addr[0]], [temps_addr[1]], x_partitions[lm + amnt:Nx:amnt+1], x_partitions[lm: Nx - amnt:amnt+1])
            ParallelArithmetic.__mux(sim, tj, temps_addr[1], x_addr, x_addr, inter, x_partitions, nota_addr=not_tj)
            inter.free(temps_addr, x_partitions)

            inter.free(tj, x_partitions)
            inter.free(not_tj, x_partitions)

        if sticky_addr is not None:

            # Update output sticky, while taking OR with previous sticky value
            temps_addr = inter.malloc(2, x_partitions[0])
            sim.perform(constants.GateType.INIT1, [], [temps_addr[0]], x_partitions[0])
            sim.perform(constants.GateType.NOT, [sticky_addr], [temps_addr[0]], sticky_partition, x_partitions[0])
            sim.perform(constants.GateType.INIT1, [], [temps_addr[1]], x_partitions[0])
            sim.perform(constants.GateType.NOT, [temps_addr[0]], [temps_addr[1]], x_partitions[0])
            sim.perform(constants.GateType.INIT1, [], [temps_addr[0]], x_partitions[0])
            sim.perform(constants.GateType.NOR, [local_sticky_bit_addr, temps_addr[1]], [temps_addr[0]], x_partitions[0])
            sim.perform(constants.GateType.INIT1, [], [sticky_addr], sticky_partition)
            sim.perform(constants.GateType.NOT, [temps_addr[0]], [sticky_addr], x_partitions[0], sticky_partition)
            inter.free(temps_addr, x_partitions[0])
            inter.free(local_sticky_bit_addr, x_partitions[0])

    @staticmethod
    def __normalizeShift(sim: simulator.ParallelSimulator, x_addr: int, t_addr: int, inter,
            x_partitions, t_partitions, direction=False):
        """
        Performs the in-place variable normalization operation on the given columns
        :param sim: the simulation environment
        :param x_addr: the intra-partition address of input & output x
        :param t_addr: the intra-partition address of output t
        :param inter: addresses for inter. Either np array or IntermediateAllocator. Relevant to "x_partitions".
        :param x_partitions: the partitions for input x
        :param t_partitions: the partitions for output t
        :param direction: the direction of the shift. False is right-shift, and True is left-shift.
        """

        if isinstance(x_partitions, np.int64):
            x_partitions = np.array([x_partitions], dtype=int)
        if isinstance(t_partitions, np.int64):
            t_partitions = np.array([t_partitions], dtype=int)

        Nx = len(x_partitions)
        log2_Nx = ceil(log2(Nx))
        Nt = len(t_partitions)
        assert(Nt <= log2_Nx)

        if direction:
            x_partitions = np.flip(x_partitions)

        if isinstance(inter, np.ndarray):
            inter = ParallelArithmetic.IntermediateAllocator(inter, x_partitions)

        for j in reversed(range(Nt)):

            tj = inter.malloc(1, x_partitions)
            not_tj = inter.malloc(1, x_partitions)
            sim.perform(constants.GateType.INIT1, [], [tj], x_partitions)
            sim.perform(constants.GateType.INIT1, [], [not_tj], x_partitions)
            ParallelArithmetic.__reduceOR(sim, x_addr, not_tj, inter, x_partitions[:(2 ** j)], x_partitions[0])
            # Broadcast tj and not tj to all partitions
            sim.perform(constants.GateType.NOT, [not_tj], [tj], x_partitions[0])
            for k in range(log2_Nx):
                sim.perform(constants.GateType.NOT, [tj], [not_tj],
                            x_partitions[np.arange(0, Nx - (1 << (log2_Nx - k - 1)), 1 << (log2_Nx - k))],
                            x_partitions[np.arange((1 << (log2_Nx - k - 1)), Nx, 1 << (log2_Nx - k))])
                sim.perform(constants.GateType.NOT, [not_tj], [tj],
                            x_partitions[np.arange((1 << (log2_Nx - k - 1)), Nx, 1 << (log2_Nx - k))])
            # Copy to output t
            sim.perform(constants.GateType.INIT1, [], [t_addr], t_partitions[j])
            sim.perform(constants.GateType.NOT, [not_tj], [t_addr], x_partitions[0], t_partitions[j])

            # Perform a shift of 2 ** j to the right
            amnt = 2 ** j
            temps_addr = inter.malloc(2, x_partitions)
            sim.perform(constants.GateType.INIT1, [], [temps_addr[0]], x_partitions)
            sim.perform(constants.GateType.NOT, [x_addr], [temps_addr[0]], x_partitions)
            sim.perform(constants.GateType.INIT1, [], [temps_addr[1]], x_partitions[:Nx-amnt])
            sim.perform(constants.GateType.INIT0, [], [temps_addr[1]], x_partitions[Nx - amnt:])
            for lm in range(amnt + 1):
                if len(x_partitions[lm + amnt:Nx:amnt+1]) > 0:
                    sim.perform(constants.GateType.NOT, [temps_addr[0]], [temps_addr[1]], x_partitions[lm + amnt:Nx:amnt+1], x_partitions[lm: Nx - amnt:amnt+1])
            ParallelArithmetic.__mux(sim, tj, temps_addr[1], x_addr, x_addr, inter, x_partitions, nota_addr=not_tj)
            inter.free(temps_addr, x_partitions)

            inter.free(tj, x_partitions)
            inter.free(not_tj, x_partitions)

        # If didn't contain any ones, then we define shift amount as zero
        not_lsb_addr = inter.malloc(1, x_partitions[0])
        sim.perform(constants.GateType.INIT1, [], [not_lsb_addr], x_partitions[0])
        sim.perform(constants.GateType.NOT, [x_addr], [not_lsb_addr], x_partitions[0])
        for t in t_partitions:
            sim.perform(constants.GateType.NOT, [not_lsb_addr], [t_addr], x_partitions[0], t)  # X-MAGIC
        inter.free(not_lsb_addr, x_partitions[0])

    @staticmethod
    def __reduceOR(sim: simulator.ParallelSimulator, x_addr: int, z_addr: int, inter, x_partitions, z_partition):
        """
        Performs an OR reduction on the x bits, storing the result in z
        :param sim: the simulation environment
        :param x_addr: the intra-partition address of input x
        :param z_addr: the intra-partition address of output x
        :param inter: addresses for inter. Either np array or IntermediateAllocator. Relevant to "x_partitions".
        :param x_partitions: the partitions for input x
        :param z_partition: the partition for output z
        """

        N = len(x_partitions)
        log2_N = ceil(log2(N))

        if isinstance(inter, np.ndarray):
            inter = ParallelArithmetic.IntermediateAllocator(inter, x_partitions)

        # Compute the NOT of all bits (for De Morgan)
        notx_addr = inter.malloc(1, x_partitions)
        sim.perform(constants.GateType.INIT1, [], [notx_addr], x_partitions)
        sim.perform(constants.GateType.NOT, [x_addr], [notx_addr], x_partitions)

        # Compute the reduction
        temp_addr = inter.malloc(1, x_partitions)
        sim.perform(constants.GateType.INIT1, [], [temp_addr], x_partitions)
        for j in range(log2_N):

            inp = x_partitions[np.flip(N - 1 - np.arange(1 << j, N, 1 << (j + 1)))]
            outp = x_partitions[np.flip(N - 1 - np.arange(0, N - (1 << j), 1 << (j + 1)))]

            sim.perform(constants.GateType.NOT, [notx_addr], [temp_addr], input_partitions=inp)
            sim.perform(constants.GateType.NOT, [temp_addr], [notx_addr], input_partitions=inp, output_partitions=outp)

        inter.free(temp_addr, x_partitions)

        sim.perform(constants.GateType.INIT1, [], [z_addr], z_partition)
        sim.perform(constants.GateType.NOT, [notx_addr], [z_addr], x_partitions[-1], z_partition)
        inter.free(notx_addr, x_partitions)

    @staticmethod
    def __fixedAddBit(sim: simulator.ParallelSimulator, x_addr: int, z_addr: int,
            inter, cin_addr, cin_partition=None, cout_addr=None, cout_partition=None, partitions=None):
        """
        Performs a fixed-point signed addition between a number and a bit (cin_addr)
        :param sim: the simulation environment
        :param x_addr: the intra-partition address of input x (N-bit)
        :param z_addr: the intra-partition address of output z (N-bit)
        :param inter: addresses for inter. Either np array or IntermediateAllocator. Relevant to "partitions".
        :param partitions: the partitions to operate on
        :param cin_addr: the intra-partition address of input cin_addr (1-bit).
        :param cin_partition: the partition address of input cin_addr (1-bit). Lowest partition by default.
        :param cout_addr: the intra-partition address of output cout_addr (1-bit)
        :param cout_partition: the partition address of output cout_addr (1-bit). Highest partition by default.
        """

        if partitions is None:
            partitions = np.arange(sim.num_partitions)
        N = len(partitions)
        log2_N = ceil(log2(N))

        if isinstance(inter, np.ndarray):
            inter = ParallelArithmetic.IntermediateAllocator(inter, partitions)

        # Pre-compute the propagate and not propagate bits
        # Propagate = x_addr
        pbit_addr = inter.malloc(1, partitions)
        notpbit_addr = inter.malloc(1, partitions)
        sim.perform(constants.GateType.INIT1, [], [notpbit_addr], partitions)
        sim.perform(constants.GateType.NOT, [x_addr], [notpbit_addr], partitions)
        sim.perform(constants.GateType.INIT1, [], [pbit_addr], partitions)
        sim.perform(constants.GateType.NOT, [notpbit_addr], [pbit_addr], partitions)

        # Pre-compute the not generate bits
        # Generate = zero
        notgbit_addr = inter.malloc(1, partitions)
        sim.perform(constants.GateType.INIT1, [], [notgbit_addr], partitions)

        # Deal with carry-in: if carry-in and propagate at LSB, then set generate of LSB to one
        notcin_addr = None
        if cin_addr == -1:
            sim.perform(constants.GateType.NOT, [pbit_addr], [notgbit_addr], partitions[0])
        else:
            # Choose default cin_partition if not explicitly chosen
            cin_partition = partitions[0] if cin_partition is None else cin_partition
            # generate = OR(generate, AND(carry-in, propagate))
            # NOT generate = NOT OR(generate, AND(carry-in, propagate)) = AND(NOT generate, NOT AND (carry-in, propagate))
            temp = inter.malloc(1, partitions[0])
            notcin_addr = inter.malloc(1, partitions[0])
            sim.perform(constants.GateType.INIT1, [], [notcin_addr], partitions[0])
            sim.perform(constants.GateType.NOT, [cin_addr], [notcin_addr], cin_partition, partitions[0])
            sim.perform(constants.GateType.INIT1, [], [temp], partitions[0])
            sim.perform(constants.GateType.NOR, [notpbit_addr, notcin_addr], [temp], partitions[0])
            sim.perform(constants.GateType.NOT, [temp], [notgbit_addr], partitions[0])
            inter.free(temp, partitions[0])

        # Perform the prefix operation
        for i in range(log2_N - 1):
            # Perform operation from partitions[np.arange((1 << i), N - (1 << i), 1 << (i + 1))]
            # to partitions[np.arange(1 << (i + 1), N, 1 << (i + 1))]
            inp = partitions[np.arange((1 << i), N - (1 << i), 1 << (i + 1))]
            outp = partitions[np.arange(1 << (i + 1), N, 1 << (i + 1))]

            sim.perform(constants.GateType.NOT, [notgbit_addr], [pbit_addr], input_partitions=inp,
                        output_partitions=outp)

            sim.perform(constants.GateType.NOT, [pbit_addr], [notgbit_addr], input_partitions=outp)

            sim.perform(constants.GateType.INIT1, [], [pbit_addr], input_partitions=outp)
            sim.perform(constants.GateType.NOT, [notpbit_addr], [pbit_addr], input_partitions=outp)

            sim.perform(constants.GateType.NOT, [notpbit_addr], [pbit_addr], input_partitions=inp,
                        output_partitions=outp)

            sim.perform(constants.GateType.INIT1, [], [notpbit_addr], input_partitions=outp)
            sim.perform(constants.GateType.NOT, [pbit_addr], [notpbit_addr], input_partitions=outp)

        for i in range(log2_N):
            # Perform operation from partitions[np.arange(0, N, 1 << (log2_N - i))]
            # to partitions[np.arange((N >> (i + 1)), N + (N >> (i + 1)), 1 << (log2_N - i))]
            inp = partitions[np.arange(0, N - (1 << (log2_N - i - 1)), 1 << (log2_N - i))]
            outp = partitions[np.arange((1 << (log2_N - i - 1)), N, 1 << (log2_N - i))]

            sim.perform(constants.GateType.NOT, [notgbit_addr], [pbit_addr], input_partitions=inp,
                        output_partitions=outp)

            sim.perform(constants.GateType.NOT, [pbit_addr], [notgbit_addr], input_partitions=outp)

            sim.perform(constants.GateType.INIT1, [], [pbit_addr], input_partitions=outp)
            sim.perform(constants.GateType.NOT, [notpbit_addr], [pbit_addr], input_partitions=outp)

            sim.perform(constants.GateType.NOT, [notpbit_addr], [pbit_addr], input_partitions=inp,
                        output_partitions=outp)

            sim.perform(constants.GateType.INIT1, [], [notpbit_addr], input_partitions=outp)
            sim.perform(constants.GateType.NOT, [pbit_addr], [notpbit_addr], input_partitions=outp)

        inter.free(pbit_addr, partitions)
        inter.free(notpbit_addr, partitions)

        carry_loc = inter.malloc(1, partitions)

        # Shift the carries to the right
        sim.perform(constants.GateType.INIT1, [], [carry_loc], partitions)
        if cin_addr == -1:
            sim.perform(constants.GateType.INIT1, [], [carry_loc], partitions[0])
        else:
            sim.perform(constants.GateType.NOT, [notcin_addr], [carry_loc], partitions[0])
            inter.free(notcin_addr, partitions[0])
        sim.perform(constants.GateType.NOT, [notgbit_addr], [carry_loc], partitions[np.arange(0, N - 1, 2)],
                    partitions[np.arange(1, N, 2)])
        sim.perform(constants.GateType.NOT, [notgbit_addr], [carry_loc], partitions[np.arange(1, N - 1, 2)],
                    partitions[np.arange(2, N, 2)])
        if cout_addr is not None:
            # Choose default cout_partition if not explicitly chosen
            cout_partition = partitions[-1] if cout_partition is None else cout_partition
            sim.perform(constants.GateType.INIT1, [], [cout_addr], cout_partition)
            sim.perform(constants.GateType.NOT, [notgbit_addr], [cout_addr], partitions[-1], cout_partition)

        inter.free(notgbit_addr, partitions)

        # Compute the final sum as XOR(x, carry_loc)
        ParallelArithmetic.__xor(sim, x_addr, carry_loc, z_addr, inter, partitions)

        inter.free(carry_loc, partitions)

    @staticmethod
    def __abs(sim: simulator.ParallelSimulator, x_addr: int, z_addr: int, inter, partitions=None):
        """
        Performs a fixed-point signed absolute values operation on x and stores in z
        :param sim: the simulation environment
        :param x_addr: the intra-partition address of input x (N-bit)
        :param z_addr: the intra-partition address of output z (N-bit)
        :param inter: addresses for inter. Either np array or IntermediateAllocator.
        :param partitions: the partitions to operate on
        """

        N = len(partitions)
        log2_N = ceil(log2(N))

        if isinstance(inter, np.ndarray):
            inter = ParallelArithmetic.IntermediateAllocator(inter, partitions)

        # Compute XOR of x in all partitions with the MSB of x
        # Broadcast MSB of x to all partitions
        msb_addr = inter.malloc(1, partitions)
        notmsb_addr = inter.malloc(1, partitions)
        sim.perform(constants.GateType.INIT1, [], [msb_addr], partitions)
        sim.perform(constants.GateType.INIT1, [], [notmsb_addr], partitions)
        sim.perform(constants.GateType.NOT, [x_addr], [notmsb_addr], partitions[-1], partitions[0])
        sim.perform(constants.GateType.NOT, [notmsb_addr], [msb_addr], partitions[0])
        for j in range(log2_N):
            sim.perform(constants.GateType.NOT, [msb_addr], [notmsb_addr],
                        partitions[np.arange(0, N - (1 << (log2_N - j - 1)), 1 << (log2_N - j))],
                        partitions[np.arange((1 << (log2_N - j - 1)), N, 1 << (log2_N - j))])
            sim.perform(constants.GateType.NOT, [notmsb_addr], [msb_addr],
                        partitions[np.arange((1 << (log2_N - j - 1)), N, 1 << (log2_N - j))])

        ParallelArithmetic.__xor(sim, msb_addr, x_addr, z_addr, inter, partitions, nota_addr=notmsb_addr)

        inter.free(notmsb_addr, partitions)

        ParallelArithmetic.__fixedAddBit(sim, z_addr, z_addr, inter, cin_addr=msb_addr, cin_partition=partitions[-1], partitions=partitions)

        inter.free(msb_addr, partitions)
