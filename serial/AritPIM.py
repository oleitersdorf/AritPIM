import numpy as np
from util import constants
from math import ceil, log2
import simulator


class SerialArithmetic:
    """
    The proposed algorithms for bit-serial arithmetic
    """

    class IntermediateAllocator:
        """
        Helper that assists in the allocation of intermediate cells
        """

        def __init__(self, cells: np.ndarray):
            """
            Initializes the allocator
            :param cells: a np list of the available cells
            """

            self.cells = cells
            self.cells_inverse = {cells[i]: i for i in range(len(cells))}
            self.allocated = np.zeros_like(cells, dtype=bool)  # vector containing 1 if allocated, 0 otherwise

        def malloc(self, num_cells: int):
            """
            Allocates num_cells cells
            :param num_cells: the number of cells to allocate
            :return: np array containing the allocated indices, or int if num_cells = 1
            """

            assert(num_cells >= 1)

            allocation = []

            # Search for available cells (first searching between previous allocations, then extending if necessary)
            for i in range(len(self.cells)):
                if not self.allocated[i]:
                    allocation.append(i)
                    # Mark the cell as allocated
                    self.allocated[i] = True
                if len(allocation) == num_cells:
                    break

            # Assert that there were enough cells
            assert(len(allocation) == num_cells)

            # Return the allocated cells
            if num_cells > 1:
                return np.array(self.cells[allocation], dtype=int)
            else:
                return self.cells[allocation[0]]

        def free(self, cells):
            """
            Frees the given cells
            :param cells: np array containing the cells to free, or int (if num_cells was 1)
            """

            if isinstance(cells, np.ndarray):
                self.allocated[np.array([self.cells_inverse[x] for x in cells])] = False
            else:
                self.allocated[self.cells_inverse[cells]] = False

    @staticmethod
    def fixedAddition(sim: simulator.SerialSimulator, x_addr: np.ndarray, y_addr: np.ndarray, z_addr: np.ndarray,
            inter, cin_addr=None, cout_addr=None):
        """
        Performs a fixed-point addition on the given columns. Supports both unsigned and signed numbers.
        Cycles: 1 + N * FA = 18N + 1
        :param sim: the simulation environment
        :param x_addr: the addresses of input x (N-bit)
        :param y_addr: the addresses of input y (N-bit)
        :param z_addr: the addresses for the output z (N-bit)
        :param inter: addresses for inter. Either np array or IntermediateAllocator.
        :param cin_addr: the address for an optional input carry. "-1" designates constant 1 input carry.
        :param cout_addr: the address for an optional output carry
        """

        N = len(x_addr)
        assert(len(y_addr) == N)
        assert(len(z_addr) == N)

        if isinstance(inter, np.ndarray):
            inter = SerialArithmetic.IntermediateAllocator(inter)

        carry_addr = inter.malloc(1)

        # Initialize the input carry
        if cin_addr is None:
            cin_addr = carry_addr
            sim.perform(constants.GateType.INIT0, [], [carry_addr])
        elif cin_addr == -1:
            cin_addr = carry_addr
            sim.perform(constants.GateType.INIT1, [], [carry_addr])

        # Setup the output carry
        if cout_addr is None:
            cout_addr = carry_addr

        # Perform the N iterations of full-adders
        for i in range(N):

            # The input and output carry locations
            in_carry_addr = carry_addr if i > 0 else cin_addr
            out_carry_addr = carry_addr if i < N - 1 else cout_addr

            # Perform the full-adder
            SerialArithmetic.__fullAdder(sim, x_addr[i], y_addr[i], in_carry_addr, z_addr[i], out_carry_addr, inter)

        inter.free(carry_addr)

    @staticmethod
    def fixedSubtraction(sim: simulator.SerialSimulator, x_addr: np.ndarray, y_addr: np.ndarray, z_addr: np.ndarray,
            inter, cout_addr=None):
        """
        Performs a fixed-point subtraction on the given columns. Supports both unsigned and signed numbers.
        Cycles: 1 + N * FS = 20N + 1
        :param sim: the simulation environment
        :param x_addr: the addresses of input x (N-bit)
        :param y_addr: the addresses of input y (N-bit)
        :param z_addr: the addresses for the output z (N-bit)
        :param inter: addresses for inter. Either np array or IntermediateAllocator.
        :param cout_addr: the address for an optional output carry
        """

        N = len(x_addr)
        assert (len(y_addr) == N)
        assert (len(z_addr) == N)

        if isinstance(inter, np.ndarray):
            inter = SerialArithmetic.IntermediateAllocator(inter)

        carry_addr = inter.malloc(1)

        # Initialize the input carry
        sim.perform(constants.GateType.INIT1, [], [carry_addr])

        # Setup the output carry
        if cout_addr is None:
            cout_addr = carry_addr

        # Perform the N iterations of full-subtractors
        for i in range(N):

            # The input and output carry locations
            in_carry_addr = carry_addr
            out_carry_addr = carry_addr if i < N - 1 else cout_addr

            # Perform the full-subtractor
            SerialArithmetic.__fullSubtractor(sim, x_addr[i], y_addr[i], in_carry_addr, z_addr[i], out_carry_addr, inter)

        inter.free(carry_addr)

    @staticmethod
    def fixedMultiplication(sim: simulator.SerialSimulator, x_addr: np.ndarray, y_addr: np.ndarray, z_addr: np.ndarray, inter):
        """
        Performs a fixed-point multiplication on the given columns. Supports only unsigned numbers.
        Note: Can be extended to signed by first performing absolute value on inputs, and then conditionally inverting
            the output if the input signs were different.
        Cycles:
            For N <= 20: 2N + 2N + 3N + (N-1) * (1 + N * (2 + FA)) = 7N + (N-1) * (1 + 20N) = 20N^2 - 12N - 1
            For N > 20:
                1 + 2 * FixedAdd(ceil(N/2)) + FixedMult(ceil(N/2) + 1) + FixedMult(ceil(N/2)) +
                FixedMult(floor(N/2)) + 2 * FixedSubtract(2 * (ceil(N/2) + 1))
                + FixedAdd(2 * (ceil(N/2) + 1)) + FixedAddBit(N - ceil(N/2) - 2)
                = 1 + 2 * (1 + 18 * ceil(N/2)) + FixedMult(ceil(N/2) + 1) + FixedMult(ceil(N/2)) +
                FixedMult(floor(N/2)) + 2 * (1 + 20 * (2 * (ceil(N/2) + 1))) +
                (1 + 18 * (2 * (ceil(N/2) + 1))) + 10 * (N - ceil(N/2) - 2)
                = 142 * ceil(N/2) + 10N + 102 + FixedMult(ceil(N/2) + 1) + FixedMult(ceil(N/2)) +
                FixedMult(floor(N/2))
            Note: for even N such that 20 < N <= 38, we find:
                = 81N + 102 + (20 * (N/2 + 1)^2 - 12*(N/2 + 1) - 1) + 2 * (20 * (N/2)^2 - 12*(N/2) - 1)
                = 15N^2 + 83N + 107
        :param sim: the simulation environment
        :param x_addr: the addresses of input x (N-bit)
        :param y_addr: the addresses of input y (N-bit)
        :param z_addr: the addresses for the output z (2N-bit)
        :param inter: addresses for inter. Either np array or IntermediateAllocator.
        """

        N = len(x_addr)
        assert (len(y_addr) == N)
        assert (len(z_addr) == 2 * N)

        if isinstance(inter, np.ndarray):
            inter = SerialArithmetic.IntermediateAllocator(inter)

        if N <= 20:

            notx_addr = inter.malloc(N)
            noty_bit_addr = inter.malloc(1)
            p_bit_addr = inter.malloc(1)

            # Compute and store not(x) in advance
            for i in range(N):
                sim.perform(constants.GateType.INIT1, [], [notx_addr[i]])
                sim.perform(constants.GateType.NOT, [x_addr[i]], [notx_addr[i]])

            # Iterate over partial products
            for i in range(N):

                # Compute y_i'
                sim.perform(constants.GateType.INIT1, [], [noty_bit_addr])
                sim.perform(constants.GateType.NOT, [y_addr[i]], [noty_bit_addr])

                if i == 0:
                    for j in range(N):
                        sim.perform(constants.GateType.INIT1, [], [z_addr[j]])
                        sim.perform(constants.GateType.NOR, [notx_addr[j], noty_bit_addr], [z_addr[j]])
                    for j in range(N, 2 * N):
                        sim.perform(constants.GateType.INIT0, [], [z_addr[j]])
                else:
                    # Perform partial product computation and addition
                    carry_addr = inter.malloc(1)
                    sim.perform(constants.GateType.INIT0, [], [carry_addr])
                    for j in range(N):

                        in_carry_addr = carry_addr
                        out_carry_addr = carry_addr if j < N - 1 else z_addr[i + N]

                        sim.perform(constants.GateType.INIT1, [], [p_bit_addr])
                        sim.perform(constants.GateType.NOR, [notx_addr[j], noty_bit_addr], [p_bit_addr])

                        SerialArithmetic.__fullAdder(sim, z_addr[i + j], p_bit_addr, in_carry_addr, z_addr[i + j], out_carry_addr, inter)
                    inter.free(carry_addr)

            inter.free(notx_addr)
            inter.free(noty_bit_addr)
            inter.free(p_bit_addr)

        else:

            lower_size = N // 2
            upper_size = N - lower_size
            t1_n = 2 * (upper_size + 1)

            # Zero bit (constant zero used typically for padding)
            zero_bit_addr = inter.malloc(1)
            sim.perform(constants.GateType.INIT0, [], [zero_bit_addr])

            # Compute x0 + x1
            xsum_addr = z_addr[:upper_size + 1]
            SerialArithmetic.fixedAddition(sim,
                np.concatenate((x_addr[:lower_size], np.array([zero_bit_addr] * (upper_size - lower_size), dtype=int))),
                x_addr[lower_size:],
                xsum_addr[:upper_size],
                inter, cout_addr=xsum_addr[upper_size])
            # Compute y0 + y1
            ysum_addr = z_addr[N:N + upper_size + 1]
            SerialArithmetic.fixedAddition(sim,
                np.concatenate((y_addr[:lower_size], np.array([zero_bit_addr] * (upper_size - lower_size), dtype=int))),
                y_addr[lower_size:],
                ysum_addr[:upper_size],
                inter, cout_addr=ysum_addr[upper_size])

            t1tag_addr = inter.malloc(t1_n)

            SerialArithmetic.fixedMultiplication(sim, xsum_addr, ysum_addr, t1tag_addr, inter)

            SerialArithmetic.fixedMultiplication(sim, x_addr[:lower_size], y_addr[:lower_size], z_addr[:2 * lower_size], inter)

            SerialArithmetic.fixedMultiplication(sim, x_addr[lower_size:], y_addr[lower_size:], z_addr[2 * lower_size:], inter)

            # t1 = t1tag - t0 - t2
            SerialArithmetic.fixedSubtraction(sim,
                t1tag_addr,
                np.concatenate((z_addr[:2 * lower_size], np.array([zero_bit_addr] * (t1_n - 2 * lower_size), dtype=int))),
                t1tag_addr,
                inter)
            SerialArithmetic.fixedSubtraction(sim,
                t1tag_addr,
                np.concatenate((z_addr[2 * lower_size:], np.array([zero_bit_addr] * (t1_n - 2 * upper_size), dtype=int))),
                t1tag_addr,
                inter)

            inter.free(zero_bit_addr)

            carry_addr = inter.malloc(1)

            SerialArithmetic.fixedAddition(sim,
                t1tag_addr,
                z_addr[lower_size:lower_size+t1_n],
                z_addr[lower_size:lower_size+t1_n],
                inter, cout_addr=carry_addr)

            inter.free(t1tag_addr)

            SerialArithmetic.__fixedAddBit(sim,
                z_addr[lower_size + t1_n:],
                z_addr[lower_size + t1_n:],
                inter, cin_addr=carry_addr)

            inter.free(carry_addr)

    @staticmethod
    def fixedDivision(sim: simulator.SerialSimulator, z_addr: np.ndarray, d_addr: np.ndarray, q_addr: np.ndarray, r_addr: np.ndarray, inter):
        """
        Performs a fixed-point division on the given columns. Supports only unsigned numbers.
        Cycles:
            1 + FixedSubtract(N + 2) + 2 + (N-1) * (2 + N * XOR + (N+2) * FA + 2) + 3 + N * (4 + FA)
            = 1 + (1 + 20 * (N + 2)) + 2 + (N-1) * (2 + N * 8 + (N+2) * 18 + 2) + 3 + N * (4 + 18)
            = 26N^2 + 56N + 7
        Note: Can be extended to signed by first performing absolute value on inputs, and then conditionally inverting
            the output if the input signs were different.
        :param sim: the simulation environment
        :param z_addr: the addresses of input dividend z (2N-bit)
        :param d_addr: the addresses of input divisor d (N-bit)
        :param q_addr: the addresses for the output quotient q (N-bit)
        :param r_addr: the addresses for the output remainder r (N-bit)
        :param inter: addresses for inter. Either np array or IntermediateAllocator.
        """

        N = len(d_addr)
        assert (len(z_addr) == 2 * N)
        assert (len(q_addr) == N)
        assert (len(r_addr) == N)

        if isinstance(inter, np.ndarray):
            inter = SerialArithmetic.IntermediateAllocator(inter)

        # Zero bit (constant zero used typically for padding)
        zero_bit_addr = inter.malloc(1)
        sim.perform(constants.GateType.INIT0, [], [zero_bit_addr])

        # Effective remainder address (intermediate remainder requires N + 2 bits)
        Neff = N + 2
        reff_ext_addr = inter.malloc(2)
        reff_addr = np.concatenate((r_addr, reff_ext_addr))

        # Unroll iteration N - 1
        # Compute r = (r | zi) - d
        reff_addr = np.roll(reff_addr, 1)
        SerialArithmetic.fixedSubtraction(sim,
            np.concatenate((z_addr[N - 1:2 * N], np.array([zero_bit_addr] * (Neff - N - 1)))),
            np.concatenate((d_addr, np.array([zero_bit_addr] * (Neff - N)))),
            reff_addr, inter)
        # Derive qi from the MSB of r
        sim.perform(constants.GateType.INIT1, [], [q_addr[N - 1]])
        sim.perform(constants.GateType.NOT, [reff_addr[-1]], [q_addr[N - 1]])

        inter.free(zero_bit_addr)

        for i in reversed(range(N - 1)):

            # Compute r = (r | zi) + XOR(d, q[i+1]) + q[i+1]

            rz_addr = np.concatenate((z_addr[i:i + 1], reff_addr[:Neff - 1]))
            out_addr = np.roll(reff_addr, 1)

            # Precompute NOT q[i+1]
            notq_bit_addr = inter.malloc(1)
            sim.perform(constants.GateType.INIT1, [], [notq_bit_addr])
            sim.perform(constants.GateType.NOT, [q_addr[i + 1]], [notq_bit_addr])

            # Initialize the carry in
            carry_addr = inter.malloc(1)

            # Perform the full-adder iterations
            for j in range(Neff):

                in_carry_addr = carry_addr if j > 0 else q_addr[i + 1]
                out_carry_addr = carry_addr

                if j < N:

                    dq_xor_addr = inter.malloc(1)
                    SerialArithmetic.__xor(sim, d_addr[j], q_addr[i + 1], dq_xor_addr, inter, notb_addr=notq_bit_addr)
                    SerialArithmetic.__fullAdder(sim, rz_addr[j], dq_xor_addr, in_carry_addr, out_addr[j], out_carry_addr, inter)
                    inter.free(dq_xor_addr)

                else:
                    SerialArithmetic.__fullAdder(sim, rz_addr[j], q_addr[i + 1], in_carry_addr, out_addr[j], out_carry_addr, inter)

            inter.free(notq_bit_addr)
            inter.free(carry_addr)

            reff_addr = np.roll(reff_addr, 1)

            # Derive qi from the MSB of r
            sim.perform(constants.GateType.INIT1, [], [q_addr[i]])
            sim.perform(constants.GateType.NOT, [reff_addr[-1]], [q_addr[i]])

        # Compute r = r + AND(d, r_{-1})

        # Precompute NOT r[-1]
        notr_bit_addr = inter.malloc(1)
        sim.perform(constants.GateType.INIT1, [], [notr_bit_addr])
        sim.perform(constants.GateType.NOT, [reff_addr[-1]], [notr_bit_addr])
        # Initialize carry
        carry_addr = inter.malloc(1)
        sim.perform(constants.GateType.INIT0, [], [carry_addr])
        for j in range(N):

            temps_addr = inter.malloc(2)

            # Compute t[1] = AND(r[-1], d[j])
            sim.perform(constants.GateType.INIT1, [], [temps_addr[0]])
            sim.perform(constants.GateType.NOT, [d_addr[j]], [temps_addr[0]])
            sim.perform(constants.GateType.INIT1, [], [temps_addr[1]])
            sim.perform(constants.GateType.NOR, [notr_bit_addr, temps_addr[0]], [temps_addr[1]])

            SerialArithmetic.__fullAdder(sim, reff_addr[j], temps_addr[1], carry_addr, r_addr[j], carry_addr, inter)

            inter.free(temps_addr)

        inter.free(notr_bit_addr)
        inter.free(carry_addr)

        inter.free(reff_ext_addr)

    @staticmethod
    def floatingAdditionUnsignedIEEE(sim: simulator.SerialSimulator, x_addr: np.ndarray, y_addr: np.ndarray, z_addr: np.ndarray, inter):
        """
        Performs a floating-point addition on the given columns. Supports only unsigned numbers.
        Note: Assumes stored (exponent, mantissa), with sizes chosen according to the IEEE standard for
            15-bit, 31-bit, or 63-bit numbers (no sign).
        :param sim: the simulation environment
        :param x_addr: the addresses of floating-point input x
        :param y_addr: the addresses of floating-point input y
        :param z_addr: the addresses of floating-point output z
        :param inter: addresses for inter. Either np array or IntermediateAllocator.
        """

        N = len(x_addr)
        assert (len(y_addr) == N)
        assert (len(z_addr) == N)

        Ns, Ne, Nm = constants.getIEEE754Split(N + 1)

        SerialArithmetic.floatingAdditionUnsigned(sim,
            x_addr[:Ne], x_addr[Ne:],
            y_addr[:Ne], y_addr[Ne:],
            z_addr[:Ne], z_addr[Ne:], inter)

    @staticmethod
    def floatingAdditionUnsigned(sim: simulator.SerialSimulator,
            xe_addr: np.ndarray, xm_addr: np.ndarray, ye_addr: np.ndarray, ym_addr: np.ndarray,
            ze_addr: np.ndarray, zm_addr: np.ndarray, inter):
        """
        Performs a floating-point addition on the given columns. Supports only unsigned numbers.
        Cycles:
            2 * ReduceOr(Ne) + FixedSubtract(Ne) + 4 + Ne * MUX + 2 * (Nm + 1) * MUX + Abs(Ne + 1)
            + ReduceOr(Ne + 1 - ceil(log(Nm + 2))) + Nm + 1 + 2 + VariableShift(Nm + 2, ceil(log(Nm + 2)), with sticky)
            + FixedAdd(Nm + 1) + VariableShift(Nm + 2, 1, with sticky) + 6 + FixedAddBit(Nm + 1) + OR + FA
            + FixedAddBit(Ne - 1)
            = 2 * (3 + Ne) + (1 + 20 * Ne) + 4 + Ne * 6 + 2 * (Nm + 1) * 6 + (18 * (Ne + 1) - 15)
            + (3 + Ne + 1 - ceil(log(Nm + 2))) + Nm + 1 + 2 + ceil(log(Nm + 2)) * (6 * (Nm + 2) + 10) - 4 * (2^ceil(log(Nm + 2)) - 1)
            + (1 + 18 * (Nm + 1)) + 1 * (6 * (Nm + 2)) + 10) - 4 * (2^1 - 1) + 6 + 10 * (Nm + 1) + 4 + 18 + 10 * (Ne - 1)
            = (12Nm + 46Ne + 26)
            + (Nm + Ne + 11 + (6Nm + 21) * ceil(log(Nm + 2)) - 2 ^ (ceil(log(Nm + 2))+2))
            + 34Nm + 10Ne + 65

            = 47 * Nm + 57 * Ne + 102 + (6Nm + 21) * ceil(log(Nm + 2)) - 4 * 2^ceil(log(Nm + 2))
        :param sim: the simulation environment
        :param xe_addr: the addresses of input xe (Ne-bit)
        :param xm_addr: the addresses of input xm (Nm-bit)
        :param ye_addr: the addresses of input ye (Ne-bit)
        :param ym_addr: the addresses of input yn (Nm-bit)
        :param ze_addr: the addresses of output ze (Ne-bit)
        :param zm_addr: the addresses of output zn (Nm-bit)
        :param inter: addresses for inter. Either np array or IntermediateAllocator.
        """

        Ne = len(xe_addr)
        Nm = len(xm_addr)
        assert (len(ye_addr) == Ne)
        assert (len(ym_addr) == Nm)
        assert (len(ze_addr) == Ne)
        assert (len(zm_addr) == Nm)

        if isinstance(inter, np.ndarray):
            inter = SerialArithmetic.IntermediateAllocator(inter)

        # Setup the hidden bits to be 1 if the exponent is non-zero
        xhidden_addr = inter.malloc(1)
        yhidden_addr = inter.malloc(1)
        zhidden_addr = inter.malloc(1)
        SerialArithmetic.__reduceOR(sim, xe_addr, xhidden_addr, inter)
        SerialArithmetic.__reduceOR(sim, ye_addr, yhidden_addr, inter)
        xm_addr = np.concatenate((xm_addr, np.array([xhidden_addr])))
        ym_addr = np.concatenate((ym_addr, np.array([yhidden_addr])))
        zm_addr = np.concatenate((zm_addr, np.array([zhidden_addr])))
        Nm = Nm + 1

        # Compute deltaE and swap using fixed-point subtraction
        deltaE_addr = inter.malloc(Ne + 1)
        swap_addr = inter.malloc(1)
        notswap_addr = inter.malloc(1)
        SerialArithmetic.fixedSubtraction(sim, xe_addr, ye_addr, deltaE_addr[:Ne], inter, cout_addr=notswap_addr)
        sim.perform(constants.GateType.INIT1, [], [deltaE_addr[Ne]])
        sim.perform(constants.GateType.NOT, [notswap_addr], [deltaE_addr[Ne]])
        sim.perform(constants.GateType.INIT1, [], [swap_addr])
        sim.perform(constants.GateType.NOT, [notswap_addr], [swap_addr])

        # Perform conditional swap

        # ze = mux_swap(ye, xe)
        for i in range(Ne):
            SerialArithmetic.__mux(sim, swap_addr, ye_addr[i], xe_addr[i], ze_addr[i], inter, nota_addr=notswap_addr)

        # xmt = mux_swap(ym, xm)
        xmt_addr = zm_addr
        for i in range(Nm):
            SerialArithmetic.__mux(sim, swap_addr, ym_addr[i], xm_addr[i], xmt_addr[i], inter, nota_addr=notswap_addr)

        # ymt = mux_swap(xm, ym)
        ymt_addr = inter.malloc(Nm)
        for i in range(Nm):
            SerialArithmetic.__mux(sim, swap_addr, xm_addr[i], ym_addr[i], ymt_addr[i], inter, nota_addr=notswap_addr)

        inter.free(swap_addr)
        inter.free(notswap_addr)

        # Compute absDeltaE = abs(deltaE)
        SerialArithmetic.__abs(sim, deltaE_addr, deltaE_addr, inter)

        # Perform variable shift

        # We optimize the variable shift to use ceil(log2(Nm + 1)) bits instead of Nt by computing the OR of the top bits
        or_addr = inter.malloc(1)
        if Ne > ceil(log2(Nm + 1)):
            # Compute the OR of the top bits
            SerialArithmetic.__reduceOR(sim, deltaE_addr[ceil(log2(Nm + 1)):], or_addr, inter)
            # If the OR is one, then zero the mantissa
            for i in range(Nm):
                sim.perform(constants.GateType.NOT, [or_addr], [ymt_addr[i]])  # X-MAGIC
        inter.free(or_addr)
        sticky_addr = inter.malloc(1)
        guard_addr = inter.malloc(1)
        sim.perform(constants.GateType.INIT0, [], [sticky_addr])
        sim.perform(constants.GateType.INIT0, [], [guard_addr])
        SerialArithmetic.__variableShift(sim,
            np.concatenate((np.array([guard_addr]), ymt_addr)), deltaE_addr[:ceil(log2(Nm + 1))], inter, sticky_addr=sticky_addr)

        inter.free(deltaE_addr)

        # Perform mantissa addition
        mantissa_carry_addr = inter.malloc(1)
        SerialArithmetic.fixedAddition(sim, xmt_addr, ymt_addr, zm_addr, inter, cout_addr=mantissa_carry_addr)

        inter.free(ymt_addr)

        # Perform right-shift normalization
        # SerialArithmetic.__fixedAddBit(sim, ze_addr, ze_addr, inter, cin_addr=mantissa_carry_addr)  # performed as part of overflow addition
        SerialArithmetic.__variableShift(sim,
            np.concatenate((np.array([guard_addr]), zm_addr)), np.array([mantissa_carry_addr]), inter, sticky_addr=sticky_addr)

        # Perform the round-to-nearest-tie-to-even
        # should_round_addr = AND(guard, OR(sticky_addr, zm[0])) = NOR(NOT guard, NOR(sticky_addr, zm[0]))
        should_round_addr = inter.malloc(1)
        temps_addr = inter.malloc(2)
        sim.perform(constants.GateType.INIT1, [], [temps_addr[0]])
        sim.perform(constants.GateType.NOR, [sticky_addr, zm_addr[0]], [temps_addr[0]])
        sim.perform(constants.GateType.INIT1, [], [temps_addr[1]])
        sim.perform(constants.GateType.NOT, [guard_addr], [temps_addr[1]])
        sim.perform(constants.GateType.INIT1, [], [should_round_addr])
        sim.perform(constants.GateType.NOR, [temps_addr[0], temps_addr[1]], [should_round_addr])
        inter.free(temps_addr)

        inter.free(guard_addr)
        inter.free(sticky_addr)

        # Add the rounding correction bit to the mantissa
        # Store the carry-out (whether the rounding caused an overflow since the mantissa was all 1) in overflow_addr
        overflow_addr = inter.malloc(1)
        SerialArithmetic.__fixedAddBit(sim, zm_addr, zm_addr, inter, should_round_addr, overflow_addr)
        inter.free(should_round_addr)
        # If such overflow occurred, increment the exponent
        # Perform the addition with the addition of mantissa_carry_addr
        SerialArithmetic.__or(sim, zhidden_addr, overflow_addr, zhidden_addr, inter)
        temp_carry_addr = inter.malloc(1)
        SerialArithmetic.__fullAdder(sim, mantissa_carry_addr, overflow_addr, ze_addr[0], ze_addr[0], temp_carry_addr, inter)
        SerialArithmetic.__fixedAddBit(sim, ze_addr[1:], ze_addr[1:], inter, temp_carry_addr)
        inter.free(temp_carry_addr)

        inter.free(mantissa_carry_addr)

        inter.free(overflow_addr)

        inter.free(xhidden_addr)
        inter.free(yhidden_addr)
        inter.free(zhidden_addr)

    @staticmethod
    def floatingAdditionSignedIEEE(sim: simulator.SerialSimulator, x_addr: np.ndarray, y_addr: np.ndarray, z_addr: np.ndarray, inter):
        """
        Performs a floating-point addition on the given columns. Supports only signed numbers.
        Note: Assumes stored (sign, exponent, mantissa), with sizes chosen according to the IEEE standard for
            16-bit, 32-bit, or 64-bit numbers.
        :param sim: the simulation environment
        :param x_addr: the addresses of floating-point input x
        :param y_addr: the addresses of floating-point input y
        :param z_addr: the addresses of floating-point output z
        :param inter: addresses for inter. Either np array or IntermediateAllocator.
        """

        N = len(x_addr)
        assert (len(y_addr) == N)
        assert (len(z_addr) == N)

        Ns, Ne, Nm = constants.getIEEE754Split(N)

        SerialArithmetic.floatingAdditionSigned(sim,
            x_addr[:Ns], x_addr[Ns:Ns+Ne], x_addr[Ns+Ne:],
            y_addr[:Ns], y_addr[Ns:Ns+Ne], y_addr[Ns+Ne:],
            z_addr[:Ns], z_addr[Ns:Ns+Ne], z_addr[Ns+Ne:], inter)

    @staticmethod
    def floatingSubtractionSignedIEEE(sim: simulator.SerialSimulator, x_addr: np.ndarray, y_addr: np.ndarray, z_addr: np.ndarray, inter):
        """
        Performs a floating-point subtraction on the given columns. Supports only signed numbers.
        Note: Assumes stored (sign, exponent, mantissa), with sizes chosen according to the IEEE standard for
            16-bit, 32-bit, or 64-bit numbers.
        :param sim: the simulation environment
        :param x_addr: the addresses of floating-point input x
        :param y_addr: the addresses of floating-point input y
        :param z_addr: the addresses of floating-point output z
        :param inter: addresses for inter. Either np array or IntermediateAllocator.
        """

        N = len(x_addr)
        assert (len(y_addr) == N)
        assert (len(z_addr) == N)

        Ns, Ne, Nm = constants.getIEEE754Split(N)

        if isinstance(inter, np.ndarray):
            inter = SerialArithmetic.IntermediateAllocator(inter)

        # Invert the sign bit of y
        notys_addr = inter.malloc(1)
        sim.perform(constants.GateType.INIT1, [], [notys_addr])
        sim.perform(constants.GateType.NOT, y_addr[:Ns], [notys_addr])

        SerialArithmetic.floatingAdditionSigned(sim,
                                                x_addr[:Ns], x_addr[Ns:Ns + Ne], x_addr[Ns + Ne:],
                                                notys_addr, y_addr[Ns:Ns + Ne], y_addr[Ns + Ne:],
                                                z_addr[:Ns], z_addr[Ns:Ns + Ne], z_addr[Ns + Ne:], inter)

        inter.free(notys_addr)

    @staticmethod
    def floatingAdditionSigned(sim: simulator.SerialSimulator,
            xs_addr: np.ndarray, xe_addr: np.ndarray, xm_addr: np.ndarray, ys_addr: np.ndarray, ye_addr: np.ndarray, ym_addr: np.ndarray,
            zs_addr: np.ndarray, ze_addr: np.ndarray, zm_addr: np.ndarray, inter):
        """
        Performs a floating-point addition on the given columns. Supports only signed numbers.
        Cycles:
            2 * ReduceOr(Ne) + 1 + FixedSubtract(Ne) + 4 + Ne * MUX + 1 + 2 * (Nm + 1) * MUX + Abs(Ne + 1)
            + ReduceOr(Ne + 1 - ceil(log(Nm + 3))) + Nm + 1 + 3 + VariableShift(Nm + 3, ceil(log(Nm + 3)), with sticky)
            + XOR + (Nm + 1) * (XOR + FA) + 6 + Abs(Nm + 5) + 1 + ID + VariableShift(Nm + 4, 1, with sticky) +
            + NormalizeShift(Nm + 3, ceil(log(Nm + 3))) + (Ne + 1 - ceil(log2(Nm + 3))) + FixedSubtract(Ne + 1)
            + OR + 6 + FixedAddBit(Nm + 1) + OR + FA + FixedAddBit(Ne) + 4 + 2 * XOR + MUX + 2 + OR + (Nm + 1 + Ne + 1 + 1)
            = 2 * (3 + Ne) + 1 + (1 + 20 * Ne) + 4 + Ne * 6 + 1 + 2 * (Nm + 1) * 6 + (18 * (Ne + 1) - 15)
            + (3 + Ne + 1 - ceil(log(Nm + 3))) + Nm + 1 + 3 + ceil(log(Nm + 3)) * (6 * (Nm + 3) + 10) - 4 * (2^ceil(log(Nm + 3)) - 1)
            + 12 + (26 * (Nm + 1)) + 6 + (18 * (Nm + 5) - 15) + 1 + 4 + 1 * (6 * (Nm + 4) + 10) - 4 * (2^1 - 1)
            + ceil(log(Nm + 3)) * (6 * (Nm + 3) + 4) - 4 * 2^ceil(log(Nm + 3)) + 6 + (Ne + 1 - ceil(log(Nm + 3))) + (1 + 20 * (Ne + 1))
            + 4 + 6 + 10 * (Nm + 1) + 4 + 18 + 10 * Ne + 4 + 2*8 + 6 + 2 + 4 + (Nm + Ne + 1 + 1)
            = 12Nm + 46Ne + 28
            + (Nm + Ne + 12 + (6Nm + 27) * ceil(log(Nm + 3)) - 2^(ceil(log(Nm + 3))+2))
            + 50Nm + 154
            + 21Ne + 28 + (6Nm + 21) * ceil(log(Nm + 3)) - 4*2^(ceil(log(Nm + 3)))
            + 11Nm + 11Ne + 77

            = 74Nm + 79Ne + 299 + (12Nm + 48) * ceil(log(Nm + 3)) - 8*2^(ceil(log(Nm + 3))))
        :param sim: the simulation environment
        :param xs_addr: the addresses of input xs (1-bit)
        :param xe_addr: the addresses of input xe (Ne-bit)
        :param xm_addr: the addresses of input xm (Nm-bit)
        :param ys_addr: the addresses of input ys (1-bit)
        :param ye_addr: the addresses of input ye (Ne-bit)
        :param ym_addr: the addresses of input yn (Nm-bit)
        :param zs_addr: the addresses of output zs (1-bit)
        :param ze_addr: the addresses of output ze (Ne-bit)
        :param zm_addr: the addresses of output zn (Nm-bit)
        :param inter: addresses for inter. Either np array or IntermediateAllocator.
        """

        Ne = len(xe_addr)
        Nm = len(xm_addr)
        assert (len(ye_addr) == Ne)
        assert (len(ym_addr) == Nm)
        assert (len(ze_addr) == Ne)
        assert (len(zm_addr) == Nm)

        if isinstance(inter, np.ndarray):
            inter = SerialArithmetic.IntermediateAllocator(inter)

        # Setup the hidden bits to be 1 if the exponent is non-zero
        xhidden_addr = inter.malloc(1)
        yhidden_addr = inter.malloc(1)
        zhidden_addr = inter.malloc(1)
        SerialArithmetic.__reduceOR(sim, xe_addr, xhidden_addr, inter)
        SerialArithmetic.__reduceOR(sim, ye_addr, yhidden_addr, inter)
        xm_addr = np.concatenate((xm_addr, np.array([xhidden_addr])))
        ym_addr = np.concatenate((ym_addr, np.array([yhidden_addr])))
        zm_addr = np.concatenate((zm_addr, np.array([zhidden_addr])))
        Nm = Nm + 1

        # Zero bit (constant zero used typically for padding)
        zero_bit_addr = inter.malloc(1)
        sim.perform(constants.GateType.INIT0, [], [zero_bit_addr])

        # MSB of ze
        zemsb_addr = inter.malloc(1)
        ze_addr = np.concatenate((ze_addr, np.array([zemsb_addr])))

        # Compute deltaE and swap using fixed-point subtraction
        deltaE_addr = inter.malloc(Ne + 1)
        swap_addr = inter.malloc(1)
        notswap_addr = inter.malloc(1)
        SerialArithmetic.fixedSubtraction(sim, xe_addr, ye_addr, deltaE_addr[:Ne], inter, cout_addr=notswap_addr)
        sim.perform(constants.GateType.INIT1, [], [deltaE_addr[Ne]])
        sim.perform(constants.GateType.NOT, [notswap_addr], [deltaE_addr[Ne]])
        sim.perform(constants.GateType.INIT1, [], [swap_addr])
        sim.perform(constants.GateType.NOT, [notswap_addr], [swap_addr])

        # Perform conditional swap

        # ze = mux_swap(ye, xe)
        for i in range(Ne):
            SerialArithmetic.__mux(sim, swap_addr, ye_addr[i], xe_addr[i], ze_addr[i], inter, nota_addr=notswap_addr)
        sim.perform(constants.GateType.INIT0, [], [ze_addr[Ne]])

        # xmt = mux_swap(ym, xm)
        xmt_addr = zm_addr
        for i in range(Nm):
            SerialArithmetic.__mux(sim, swap_addr, ym_addr[i], xm_addr[i], xmt_addr[i], inter, nota_addr=notswap_addr)

        # ymt = mux_swap(xm, ym)
        ymt_addr = inter.malloc(Nm)
        for i in range(Nm):
            SerialArithmetic.__mux(sim, swap_addr, xm_addr[i], ym_addr[i], ymt_addr[i], inter, nota_addr=notswap_addr)

        # Compute absDeltaE = abs(deltaE)
        SerialArithmetic.__abs(sim, deltaE_addr, deltaE_addr, inter)

        # Perform variable shift

        # We optimize the variable shift to use ceil(log2(Nm + 2)) bits instead of Nt by computing the OR of the top bits
        or_addr = inter.malloc(1)
        if Ne > ceil(log2(Nm + 2)):
            # Compute the OR of the top bits
            SerialArithmetic.__reduceOR(sim, deltaE_addr[ceil(log2(Nm + 2)):], or_addr, inter)
            # If the OR is one, then zero the mantissa
            for i in range(Nm):
                sim.perform(constants.GateType.NOT, [or_addr], [ymt_addr[i]])
        inter.free(or_addr)
        guard_addr = inter.malloc(1)
        round_addr = inter.malloc(1)
        sticky_addr = inter.malloc(1)
        sim.perform(constants.GateType.INIT0, [], [sticky_addr])
        sim.perform(constants.GateType.INIT0, [], [guard_addr])
        sim.perform(constants.GateType.INIT0, [], [round_addr])
        SerialArithmetic.__variableShift(sim, np.concatenate((np.array([round_addr, guard_addr]), ymt_addr)),
            deltaE_addr[:ceil(log2(Nm + 2))], inter, sticky_addr=sticky_addr)

        inter.free(deltaE_addr)

        # Perform XOR on the signs of x and y
        sdiff_addr = inter.malloc(1)
        notsdiff_addr = inter.malloc(1)
        SerialArithmetic.__xor(sim, xs_addr.item(), ys_addr.item(), sdiff_addr, inter, notz_addr=notsdiff_addr)

        # Perform mantissa addition
        # Compute zm = (x_m' if (x_s == y_s) else -x_m') + y_m' = (x_m' XOR sdiff) + y_m + sdiff
        mantissa_carry_addr = inter.malloc(1)
        temp_carry_addr = inter.malloc(1)
        for j in range(Nm):

            in_carry_addr = temp_carry_addr if j > 0 else sdiff_addr
            out_carry_addr = temp_carry_addr if j < Nm - 1 else mantissa_carry_addr

            xor_bit_addr = inter.malloc(1)

            SerialArithmetic.__xor(sim, xmt_addr[j], sdiff_addr, xor_bit_addr, inter, notb_addr=notsdiff_addr)

            SerialArithmetic.__fullAdder(sim, ymt_addr[j], xor_bit_addr, in_carry_addr, zm_addr[j], out_carry_addr, inter)

            inter.free(xor_bit_addr)

        inter.free(temp_carry_addr)

        inter.free(ymt_addr)

        # If sdiff and not mantissa_carry, then negative_m (if negative_m = 1, then zm is negative);
        # thus, negative_M = sdiff AND (NOT mantissa_carry)) = NOT(notsdiff OR mantissa_carry) = NOR(notsdiff, mantissa_carry)
        negativeM_addr = inter.malloc(1)
        notnegativeM_addr = inter.malloc(1)
        sim.perform(constants.GateType.INIT1, [], [negativeM_addr])
        sim.perform(constants.GateType.NOR, [notsdiff_addr, mantissa_carry_addr], [negativeM_addr])
        sim.perform(constants.GateType.INIT1, [], [notnegativeM_addr])
        sim.perform(constants.GateType.NOT, [negativeM_addr], [notnegativeM_addr])

        # If negative, then set s = -s. Specifically, set s = (s XOR negative) and add with carry-in of negative
        # (implemented using the absolute value routine)
        negativeM_copy_addr = inter.malloc(1)
        sim.perform(constants.GateType.INIT1, [], [negativeM_copy_addr])
        sim.perform(constants.GateType.NOT, [notnegativeM_addr], [negativeM_copy_addr])
        SerialArithmetic.__abs(sim,
            np.concatenate((np.array([sticky_addr, round_addr, guard_addr]), zm_addr, np.array([negativeM_copy_addr]))),
            np.concatenate((np.array([sticky_addr, round_addr, guard_addr]), zm_addr, np.array([negativeM_copy_addr]))), inter)
        inter.free(negativeM_copy_addr)

        # if diff_signs, then mantissa_carry = False
        sim.perform(constants.GateType.NOT, [sdiff_addr], [mantissa_carry_addr])  # X-MAGIC

        # Perform right-shift normalization
        # SerialArithmetic.__fixedAddBit(sim, ze_addr, ze_addr, inter, cin_addr=mantissa_carry_addr)  # performed as part of overflow addition
        mantissa_carry_copy_addr = inter.malloc(1)
        SerialArithmetic.__id(sim, mantissa_carry_addr, mantissa_carry_copy_addr, inter)
        SerialArithmetic.__variableShift(sim,
            np.concatenate((np.array([round_addr, guard_addr]), zm_addr, np.array([mantissa_carry_copy_addr]))),
            np.array([mantissa_carry_addr]), inter, sticky_addr=sticky_addr)
        inter.free(mantissa_carry_copy_addr)

        # Perform left-shift normalization
        left_shift_addr = inter.malloc(Ne + 1)
        SerialArithmetic.__normalizeShift(sim,
            np.concatenate((np.array([round_addr, guard_addr]), zm_addr)),
            left_shift_addr[:ceil(log2(Nm + 2))], inter, direction=True)
        for col in left_shift_addr[ceil(log2(Nm + 2)):]:
            sim.perform(constants.GateType.INIT0, [], [col])
        # Subtract from exponent
        SerialArithmetic.fixedSubtraction(sim, ze_addr, left_shift_addr, ze_addr, inter)
        inter.free(left_shift_addr)

        # Perform the round-to-nearest-tie-to-even
        # sticky_addr = OR(round_addr, sticky_addr)
        SerialArithmetic.__or(sim, round_addr, sticky_addr, sticky_addr, inter)
        # should_round_addr = AND(guard, OR(sticky_addr, zm[0])) = NOR(NOT guard, NOR(sticky_addr, zm[0]))
        should_round_addr = inter.malloc(1)
        temps_addr = inter.malloc(2)
        sim.perform(constants.GateType.INIT1, [], [temps_addr[0]])
        sim.perform(constants.GateType.NOR, [sticky_addr, zm_addr[0]], [temps_addr[0]])
        sim.perform(constants.GateType.INIT1, [], [temps_addr[1]])
        sim.perform(constants.GateType.NOT, [guard_addr], [temps_addr[1]])
        sim.perform(constants.GateType.INIT1, [], [should_round_addr])
        sim.perform(constants.GateType.NOR, [temps_addr[0], temps_addr[1]], [should_round_addr])
        inter.free(temps_addr)

        inter.free(guard_addr)
        inter.free(round_addr)
        inter.free(sticky_addr)

        # Add the rounding correction bit to the mantissa
        # Store the carry-out (whether the rounding caused an overflow since the mantissa was all 1) in overflow_addr
        overflow_addr = inter.malloc(1)
        SerialArithmetic.__fixedAddBit(sim, zm_addr, zm_addr, inter, should_round_addr, overflow_addr)
        inter.free(should_round_addr)
        # If such overflow occurred, increment the exponent
        # Perform the addition with the addition of mantissa_carry_addr
        SerialArithmetic.__or(sim, zhidden_addr, overflow_addr, zhidden_addr, inter)
        temp_carry_addr = inter.malloc(1)
        SerialArithmetic.__fullAdder(sim, mantissa_carry_addr, overflow_addr, ze_addr[0], ze_addr[0], temp_carry_addr, inter)
        SerialArithmetic.__fixedAddBit(sim, ze_addr[1:], ze_addr[1:], inter, temp_carry_addr)
        inter.free(temp_carry_addr)

        inter.free(mantissa_carry_addr)

        inter.free(overflow_addr)

        # Computing the final sign

        # Idea: Control flow (before conversion to mux - for reference)
        # if xs == ys:
        #     zs = xs
        # else:
        #     if xs AND (NOT ys):
        #         zs = negativeM XOR swap
        #     else:
        #         zs = not negativeM XOR swap

        # Data flow. Observations:
        # 1. AND(xs, NOT ys) = NOR(NOT xs, ys)
        # 2. The top else evaluates to:
        # notNegativeM XOR swap XOR NOR(NOT xs, ys)
        # 3. Overall, we find:
        # zs = xs if XNOR(xs, ys) else (notNegativeM XOR swap XOR NOR(NOT xs, ys))  (implemented using mux)

        # Data flow. Implementation:
        xor_addr = inter.malloc(1)

        not_xs_addr = inter.malloc(1)
        sim.perform(constants.GateType.INIT1, [], [not_xs_addr])
        sim.perform(constants.GateType.NOT, [xs_addr], [not_xs_addr])

        sim.perform(constants.GateType.INIT1, [], [xor_addr])
        sim.perform(constants.GateType.NOR, [ys_addr.item(), not_xs_addr], [xor_addr])

        inter.free(not_xs_addr)

        SerialArithmetic.__xor(sim, xor_addr, swap_addr, xor_addr, inter, notb_addr=notswap_addr)
        SerialArithmetic.__xor(sim, xor_addr, notnegativeM_addr, xor_addr, inter, notb_addr=negativeM_addr)
        SerialArithmetic.__mux(sim, notsdiff_addr, xs_addr.item(), xor_addr, zs_addr.item(), inter, nota_addr=sdiff_addr)

        inter.free(xor_addr)

        inter.free(swap_addr)
        inter.free(notswap_addr)
        inter.free(sdiff_addr)
        inter.free(notsdiff_addr)
        inter.free(negativeM_addr)
        inter.free(notnegativeM_addr)

        # Set the output to zero if the zhidden is zero or exponent is negative
        # should_zero = OR(NOT zhidden, ze[-1])
        should_zero_addr = inter.malloc(1)
        sim.perform(constants.GateType.INIT1, [], [should_zero_addr])
        sim.perform(constants.GateType.NOT, [zhidden_addr], [should_zero_addr])
        SerialArithmetic.__or(sim, should_zero_addr, ze_addr[-1], should_zero_addr, inter)
        for z in np.concatenate((zs_addr, ze_addr, zm_addr)):
            sim.perform(constants.GateType.NOT, [should_zero_addr], [z])  # X-MAGIC
        inter.free(should_zero_addr)

        inter.free(zero_bit_addr)
        inter.free(zemsb_addr)

        inter.free(xhidden_addr)
        inter.free(yhidden_addr)
        inter.free(zhidden_addr)

    @staticmethod
    def floatingMultiplicationIEEE(sim: simulator.SerialSimulator, x_addr: np.ndarray, y_addr: np.ndarray, z_addr: np.ndarray, inter):
        """
        Performs a floating-point multiplication on the given columns. Supports only signed numbers.
        Note: Assumes stored (sign, exponent, mantissa), with sizes chosen according to the IEEE standard for
            16-bit, 32-bit, or 64-bit numbers.
        :param sim: the simulation environment
        :param x_addr: the addresses of floating-point input x
        :param y_addr: the addresses of floating-point input y
        :param z_addr: the addresses of floating-point output z
        :param inter: addresses for inter. Either np array or IntermediateAllocator.
        """

        N = len(x_addr)
        assert (len(y_addr) == N)
        assert (len(z_addr) == N)

        Ns, Ne, Nm = constants.getIEEE754Split(N)

        SerialArithmetic.floatingMultiplication(sim,
            x_addr[:Ns], x_addr[Ns:Ns+Ne], x_addr[Ns+Ne:],
            y_addr[:Ns], y_addr[Ns:Ns+Ne], y_addr[Ns+Ne:],
            z_addr[:Ns], z_addr[Ns:Ns+Ne], z_addr[Ns+Ne:], inter)

    @staticmethod
    def floatingMultiplication(sim: simulator.SerialSimulator,
            xs_addr: np.ndarray, xe_addr: np.ndarray, xm_addr: np.ndarray, ys_addr: np.ndarray, ye_addr: np.ndarray, ym_addr: np.ndarray,
            zs_addr: np.ndarray, ze_addr: np.ndarray, zm_addr: np.ndarray, inter):
        """
        Performs a floating-point multiplication on the given columns. Supports only signed numbers.
        :param sim: the simulation environment
        :param xs_addr: the addresses of input xs (1-bit)
        :param xe_addr: the addresses of input xe (Ne-bit)
        :param xm_addr: the addresses of input xm (Nm-bit)
        :param ys_addr: the addresses of input ys (1-bit)
        :param ye_addr: the addresses of input ye (Ne-bit)
        :param ym_addr: the addresses of input yn (Nm-bit)
        :param zs_addr: the addresses of output zs (1-bit)
        :param ze_addr: the addresses of output ze (Ne-bit)
        :param zm_addr: the addresses of output zn (Nm-bit)
        :param inter: addresses for inter. Either np array or IntermediateAllocator.
        """

        Ne = len(xe_addr)
        Nm = len(xm_addr)
        assert (len(ye_addr) == Ne)
        assert (len(ym_addr) == Nm)
        assert (len(ze_addr) == Ne)
        assert (len(zm_addr) == Nm)

        if isinstance(inter, np.ndarray):
            inter = SerialArithmetic.IntermediateAllocator(inter)

        # Setup the hidden bits to be 1 if the exponent is non-zero
        xhidden_addr = inter.malloc(1)
        yhidden_addr = inter.malloc(1)
        zhidden_addr = inter.malloc(1)
        SerialArithmetic.__reduceOR(sim, xe_addr, xhidden_addr, inter)
        SerialArithmetic.__reduceOR(sim, ye_addr, yhidden_addr, inter)
        xm_addr = np.concatenate((xm_addr, np.array([xhidden_addr])))
        ym_addr = np.concatenate((ym_addr, np.array([yhidden_addr])))
        zm_addr = np.concatenate((zm_addr, np.array([zhidden_addr])))
        Nm = Nm + 1

        # Zero bit (constant zero used typically for padding)
        zero_bit_addr = inter.malloc(1)
        sim.perform(constants.GateType.INIT0, [], [zero_bit_addr])

        # MSB of ze
        zemsb_addr = inter.malloc(1)
        ze_addr = np.concatenate((ze_addr, np.array([zemsb_addr])))

        # Compute the product of the mantissas
        mantissa_carry_addr = inter.malloc(1)
        lower_mult_addr = inter.malloc(Nm - 2)
        guard_addr = inter.malloc(1)
        SerialArithmetic.fixedMultiplication(sim, xm_addr, ym_addr,
            np.concatenate((lower_mult_addr, np.array([guard_addr]), zm_addr, np.array([mantissa_carry_addr]))), inter)
        sticky_addr = inter.malloc(1)
        SerialArithmetic.__reduceOR(sim, lower_mult_addr, sticky_addr, inter)
        inter.free(lower_mult_addr)

        # Write -(1 << Ne - 1) to ze_addr
        sim.perform(constants.GateType.INIT1, [], [ze_addr[0]])
        sim.perform(constants.GateType.INIT1, [], [ze_addr[-2]])
        sim.perform(constants.GateType.INIT1, [], [ze_addr[-1]])
        for z in ze_addr[1:-2]:
            sim.perform(constants.GateType.INIT0, [], [z])

        # Increment exponent
        # performed as part of exponent addition below
        # SerialArithmetic.__fixedAddBit(sim, ze_addr, ze_addr, inter, cin_addr=mantissa_carry_addr)

        # Perform right-shift normalization
        mantissa_carry_copy_addr = inter.malloc(1)
        SerialArithmetic.__id(sim, mantissa_carry_addr, mantissa_carry_copy_addr, inter)
        SerialArithmetic.__variableShift(sim,
            np.concatenate((np.array([guard_addr]), zm_addr, np.array([mantissa_carry_copy_addr]))), np.array([mantissa_carry_addr]),
            inter, sticky_addr=sticky_addr)
        inter.free(mantissa_carry_copy_addr)

        # Perform the round-to-nearest-tie-to-even
        # should_round_addr = AND(guard, OR(sticky_addr, zm[0])) = NOR(NOT guard, NOR(sticky_addr, zm[0]))
        should_round_addr = inter.malloc(1)
        temps_addr = inter.malloc(2)
        sim.perform(constants.GateType.INIT1, [], [temps_addr[0]])
        sim.perform(constants.GateType.NOR, [sticky_addr, zm_addr[0]], [temps_addr[0]])
        sim.perform(constants.GateType.INIT1, [], [temps_addr[1]])
        sim.perform(constants.GateType.NOT, [guard_addr], [temps_addr[1]])
        sim.perform(constants.GateType.INIT1, [], [should_round_addr])
        sim.perform(constants.GateType.NOR, [temps_addr[0], temps_addr[1]], [should_round_addr])
        inter.free(temps_addr)

        inter.free(guard_addr)
        inter.free(sticky_addr)

        overflow_addr = inter.malloc(1)
        SerialArithmetic.__fixedAddBit(sim, zm_addr, zm_addr, inter, should_round_addr, overflow_addr)
        inter.free(should_round_addr)
        # In case the rounding causes an overflow (the mantissa was all ones), increment the exponent
        SerialArithmetic.__or(sim, zhidden_addr, overflow_addr, zhidden_addr, inter)
        # SerialArithmetic.__fixedAddBit(sim, ze_addr, ze_addr, inter, overflow_addr)  # performed as part of next addition

        # Compute the new exponent
        SerialArithmetic.fixedAddition(sim, np.concatenate((xe_addr, np.array([zero_bit_addr]))), ze_addr, ze_addr, inter, cin_addr=overflow_addr)
        SerialArithmetic.fixedAddition(sim, np.concatenate((ye_addr, np.array([zero_bit_addr]))), ze_addr, ze_addr, inter, cin_addr=mantissa_carry_addr)

        inter.free(mantissa_carry_addr)

        inter.free(overflow_addr)

        # Compute the XOR of the signs
        SerialArithmetic.__xor(sim, xs_addr.item(), ys_addr.item(), zs_addr.item(), inter)

        # Set the output to zero if the zhidden is zero or exponent is negative
        # should_zero = OR(NOT zhidden, ze[-1])
        should_zero_addr = inter.malloc(1)
        sim.perform(constants.GateType.INIT1, [], [should_zero_addr])
        sim.perform(constants.GateType.NOT, [zhidden_addr], [should_zero_addr])
        SerialArithmetic.__or(sim, should_zero_addr, ze_addr[-1], should_zero_addr, inter)
        for z in np.concatenate((zs_addr, ze_addr, zm_addr)):
            sim.perform(constants.GateType.NOT, [should_zero_addr], [z])  # X-MAGIC
        inter.free(should_zero_addr)

        inter.free(zero_bit_addr)
        inter.free(zemsb_addr)

        inter.free(xhidden_addr)
        inter.free(yhidden_addr)
        inter.free(zhidden_addr)

    @staticmethod
    def floatingDivisionIEEE(sim: simulator.SerialSimulator, x_addr: np.ndarray, y_addr: np.ndarray, z_addr: np.ndarray, inter):
        """
        Performs a floating-point division on the given columns. Supports only signed numbers.
        Note: Assumes stored (sign, exponent, mantissa), with sizes chosen according to the IEEE standard for
            16-bit, 32-bit, or 64-bit numbers.
        :param sim: the simulation environment
        :param x_addr: the addresses of floating-point input x
        :param y_addr: the addresses of floating-point input y
        :param z_addr: the addresses of floating-point output z
        :param inter: addresses for inter. Either np array or IntermediateAllocator.
        """

        N = len(x_addr)
        assert (len(y_addr) == N)
        assert (len(z_addr) == N)

        Ns, Ne, Nm = constants.getIEEE754Split(N)

        SerialArithmetic.floatingDivision(sim,
            x_addr[:Ns], x_addr[Ns:Ns+Ne], x_addr[Ns+Ne:],
            y_addr[:Ns], y_addr[Ns:Ns+Ne], y_addr[Ns+Ne:],
            z_addr[:Ns], z_addr[Ns:Ns+Ne], z_addr[Ns+Ne:], inter)

    @staticmethod
    def floatingDivision(sim: simulator.SerialSimulator,
            xs_addr: np.ndarray, xe_addr: np.ndarray, xm_addr: np.ndarray, ys_addr: np.ndarray, ye_addr: np.ndarray, ym_addr: np.ndarray,
            zs_addr: np.ndarray, ze_addr: np.ndarray, zm_addr: np.ndarray, inter):
        """
        Performs a floating-point division on the given columns. Supports only signed numbers.
        :param sim: the simulation environment
        :param xs_addr: the addresses of input xs (1-bit)
        :param xe_addr: the addresses of input xe (Ne-bit)
        :param xm_addr: the addresses of input xm (Nm-bit)
        :param ys_addr: the addresses of input ys (1-bit)
        :param ye_addr: the addresses of input ye (Ne-bit)
        :param ym_addr: the addresses of input yn (Nm-bit)
        :param zs_addr: the addresses of output zs (1-bit)
        :param ze_addr: the addresses of output ze (Ne-bit)
        :param zm_addr: the addresses of output zn (Nm-bit)
        :param inter: addresses for inter. Either np array or IntermediateAllocator.
        """

        Ne = len(xe_addr)
        Nm = len(xm_addr)
        assert (len(ye_addr) == Ne)
        assert (len(ym_addr) == Nm)
        assert (len(ze_addr) == Ne)
        assert (len(zm_addr) == Nm)

        if isinstance(inter, np.ndarray):
            inter = SerialArithmetic.IntermediateAllocator(inter)

        # Setup the hidden bits to be 1 if the exponent is non-zero
        xhidden_addr = inter.malloc(1)
        yhidden_addr = inter.malloc(1)
        zhidden_addr = inter.malloc(1)
        SerialArithmetic.__reduceOR(sim, xe_addr, xhidden_addr, inter)
        SerialArithmetic.__reduceOR(sim, ye_addr, yhidden_addr, inter)
        xm_addr = np.concatenate((xm_addr, np.array([xhidden_addr])))
        ym_addr = np.concatenate((ym_addr, np.array([yhidden_addr])))
        zm_addr = np.concatenate((zm_addr, np.array([zhidden_addr])))
        Nm = Nm + 1

        # Zero bit (constant zero used typically for padding)
        zero_bit_addr = inter.malloc(1)
        sim.perform(constants.GateType.INIT0, [], [zero_bit_addr])

        # MSB of ze
        zemsb_addr = inter.malloc(1)
        ze_addr = np.concatenate((ze_addr, np.array([zemsb_addr])))

        # Compute the quotient of the mantissas
        guard_addr = inter.malloc(1)
        round_addr = inter.malloc(1)
        r_addr = inter.malloc(Nm + 2)
        SerialArithmetic.fixedDivision(sim,
            np.concatenate((np.array([zero_bit_addr] * (Nm + 1)), xm_addr, np.array([zero_bit_addr] * 3))),
            np.concatenate((ym_addr, np.array([zero_bit_addr] * 2))),
            np.concatenate((np.array([round_addr, guard_addr]), zm_addr)), r_addr, inter)
        sticky_addr = inter.malloc(1)
        SerialArithmetic.__reduceOR(sim, r_addr, sticky_addr, inter)
        inter.free(r_addr)

        # Initialize the exponent to bias
        for z in ze_addr[:Ne-1]:
            sim.perform(constants.GateType.INIT1, [], [z])
        for z in ze_addr[Ne-1:]:
            sim.perform(constants.GateType.INIT0, [], [z])

        # Perform left-shift normalization
        norm_addr = inter.malloc(1)
        sim.perform(constants.GateType.INIT1, [], [norm_addr])
        sim.perform(constants.GateType.NOT, [zm_addr[-1]], [norm_addr])
        sim.perform(constants.GateType.INIT1, [], [ze_addr[0]])
        # Subtract (whether normalization occurred) from the exponent
        # Set ze[0] to NOT norm to effectively perform the subtraction
        sim.perform(constants.GateType.NOT, [norm_addr], [ze_addr[0]])
        SerialArithmetic.__variableShift(sim,
            np.concatenate((np.array([sticky_addr, round_addr, guard_addr]), zm_addr)), np.array([norm_addr]), inter, direction=True)
        inter.free(norm_addr)

        # Perform the round-to-nearest-tie-to-even
        # sticky_addr = OR(round_addr, sticky_addr)
        SerialArithmetic.__or(sim, round_addr, sticky_addr, sticky_addr, inter)
        # should_round_addr = AND(guard, OR(sticky_addr, zm[0])) = NOR(NOT guard, NOR(sticky_addr, zm[0]))
        should_round_addr = inter.malloc(1)
        temps_addr = inter.malloc(2)
        sim.perform(constants.GateType.INIT1, [], [temps_addr[0]])
        sim.perform(constants.GateType.NOR, [sticky_addr, zm_addr[0]], [temps_addr[0]])
        sim.perform(constants.GateType.INIT1, [], [temps_addr[1]])
        sim.perform(constants.GateType.NOT, [guard_addr], [temps_addr[1]])
        sim.perform(constants.GateType.INIT1, [], [should_round_addr])
        sim.perform(constants.GateType.NOR, [temps_addr[0], temps_addr[1]], [should_round_addr])
        inter.free(temps_addr)

        inter.free(guard_addr)
        inter.free(round_addr)
        inter.free(sticky_addr)

        overflow_addr = inter.malloc(1)
        SerialArithmetic.__fixedAddBit(sim, zm_addr, zm_addr, inter, should_round_addr, overflow_addr)
        inter.free(should_round_addr)
        # In case the rounding causes an overflow (the mantissa was all ones), increment the exponent
        SerialArithmetic.__or(sim, zhidden_addr, overflow_addr, zhidden_addr, inter)
        # SerialArithmetic.__fixedAddBit(sim, ze_addr, ze_addr, inter, overflow_addr)  # performed as part of next addition

        # Compute the new exponent
        SerialArithmetic.fixedAddition(sim, ze_addr, np.concatenate((xe_addr, np.array([zero_bit_addr]))), ze_addr, inter, cin_addr=overflow_addr)
        SerialArithmetic.fixedSubtraction(sim, ze_addr, np.concatenate((ye_addr, np.array([zero_bit_addr]))), ze_addr, inter)

        inter.free(overflow_addr)

        # Compute the XOR of the signs
        SerialArithmetic.__xor(sim, xs_addr.item(), ys_addr.item(), zs_addr.item(), inter)

        # Zero in case negative exponent or x was zero
        should_zero_addr = inter.malloc(1)
        sim.perform(constants.GateType.INIT1, [], [should_zero_addr])
        sim.perform(constants.GateType.NOT, [xhidden_addr], [should_zero_addr])
        SerialArithmetic.__or(sim, should_zero_addr, ze_addr[Ne], should_zero_addr, inter)
        for z in np.concatenate((zs_addr, ze_addr[:Ne], zm_addr)):
            sim.perform(constants.GateType.NOT, [should_zero_addr], [z])  # X-MAGIC
        inter.free(should_zero_addr)

        inter.free(zero_bit_addr)
        inter.free(zemsb_addr)

        inter.free(xhidden_addr)
        inter.free(yhidden_addr)
        inter.free(zhidden_addr)

    @staticmethod
    def __id(sim: simulator.SerialSimulator, a_addr: int, z_addr: int, inter, notz_addr=None):
        """
        Performs z = ID(a) on the given columns
        :param sim: the simulation environment
        :param a_addr: the index of input a
        :param z_addr: the index of the output
        :param notz_addr: the index of the optional output which stores the not of z
        :param inter: addresses for inter. Either np array or IntermediateAllocator.
        """

        if isinstance(inter, np.ndarray):
            inter = SerialArithmetic.IntermediateAllocator(inter)

        custom_not_out_addr = notz_addr is None
        if custom_not_out_addr:
            notz_addr = inter.malloc(1)

        sim.perform(constants.GateType.INIT1, [], [notz_addr])
        sim.perform(constants.GateType.NOT, [a_addr], [notz_addr])

        sim.perform(constants.GateType.INIT1, [], [z_addr])
        sim.perform(constants.GateType.NOT, [notz_addr], [z_addr])

        if custom_not_out_addr:
            inter.free(notz_addr)

    @staticmethod
    def __or(sim: simulator.SerialSimulator, a_addr: int, b_addr: int, z_addr: int, inter,
            nota_addr=None, notb_addr=None, notz_addr=None):
        """
        Performs z = OR(a,b) on the given columns
        :param sim: the simulation environment
        :param a_addr: the index of input a
        :param b_addr: the index of input b
        :param z_addr: the index of the output
        :param nota_addr: the index of the optional input which stores the not of a
        :param notb_addr: the index of the optional input which stores the not of b
        :param notz_addr: the index of the optional output which stores the not of z
        :param inter: addresses for inter. Either np array or IntermediateAllocator.
        """

        if isinstance(inter, np.ndarray):
            inter = SerialArithmetic.IntermediateAllocator(inter)

        custom_nor_out_addr = notz_addr is None
        if custom_nor_out_addr:
            notz_addr = inter.malloc(1)

        sim.perform(constants.GateType.INIT1, [], [notz_addr])
        sim.perform(constants.GateType.NOR, [a_addr, b_addr], [notz_addr])

        sim.perform(constants.GateType.INIT1, [], [z_addr])
        sim.perform(constants.GateType.NOT, [notz_addr], [z_addr])

        if custom_nor_out_addr:
            inter.free(notz_addr)

    @staticmethod
    def __xor(sim: simulator.SerialSimulator, a_addr: int, b_addr: int, z_addr: int, inter,
            nota_addr=None, notb_addr=None, notz_addr=None):
        """
        Performs z = XOR(a,b) on the given columns
        :param sim: the simulation environment
        :param a_addr: the index of input a
        :param b_addr: the index of input b
        :param z_addr: the index of the output
        :param nota_addr: the index of the optional input which stores the not of a
        :param notb_addr: the index of the optional input which stores the not of b
        :param notz_addr: the index of the optional output which stores the not of z
        :param inter: addresses for inter. Either np array or IntermediateAllocator.
        """

        if isinstance(inter, np.ndarray):
            inter = SerialArithmetic.IntermediateAllocator(inter)

        computed_not_a = nota_addr is None
        if computed_not_a:
            nota_addr = inter.malloc(1)
            sim.perform(constants.GateType.INIT1, [], [nota_addr])
            sim.perform(constants.GateType.NOT, [a_addr], [nota_addr])

        computed_not_b = notb_addr is None
        if computed_not_b:
            notb_addr = inter.malloc(1)
            sim.perform(constants.GateType.INIT1, [], [notb_addr])
            sim.perform(constants.GateType.NOT, [b_addr], [notb_addr])

        temps_addr = inter.malloc(2)

        sim.perform(constants.GateType.INIT1, [], [temps_addr[0]])
        sim.perform(constants.GateType.NOR, [a_addr, b_addr], [temps_addr[0]])
        sim.perform(constants.GateType.INIT1, [], [temps_addr[1]])
        sim.perform(constants.GateType.NOR, [nota_addr, notb_addr], [temps_addr[1]])
        sim.perform(constants.GateType.INIT1, [], [z_addr])
        sim.perform(constants.GateType.NOR, [temps_addr[0], temps_addr[1]], [z_addr])

        inter.free(temps_addr)

        if notz_addr is not None:
            sim.perform(constants.GateType.INIT1, [], [notz_addr])
            sim.perform(constants.GateType.NOT, [z_addr], [notz_addr])

        if computed_not_a:
            inter.free(nota_addr)
        if computed_not_b:
            inter.free(notb_addr)

    @staticmethod
    def __mux(sim: simulator.SerialSimulator, a_addr: int, b_addr: int, c_addr: int, z_addr: int, inter,
            nota_addr=None):
        """
        Performs a 1-bit mux_a(b,c) on the given columns
        :param sim: the simulation environment
        :param a_addr: the index of input a (the condition)
        :param b_addr: the index of input b (if a if true)
        :param c_addr: the index of input c (if a is false)
        :param z_addr: the index of the output
        :param inter: addresses for inter. Either np array or IntermediateAllocator.
        :param nota_addr: the index of the optional input which stores the not of a
        """

        if isinstance(inter, np.ndarray):
            inter = SerialArithmetic.IntermediateAllocator(inter)

        computed_not_a = nota_addr is None
        if computed_not_a:
            nota_addr = inter.malloc(1)
            sim.perform(constants.GateType.INIT1, [], [nota_addr])
            sim.perform(constants.GateType.NOT, [a_addr], [nota_addr])

        temps_addr = inter.malloc(2)

        sim.perform(constants.GateType.INIT1, [], [temps_addr[0]])
        sim.perform(constants.GateType.NOR, [b_addr, nota_addr], [temps_addr[0]])
        sim.perform(constants.GateType.INIT1, [], [temps_addr[1]])
        sim.perform(constants.GateType.NOR, [c_addr, a_addr], [temps_addr[1]])
        sim.perform(constants.GateType.INIT1, [], [z_addr])
        sim.perform(constants.GateType.NOR, [temps_addr[0], temps_addr[1]], [z_addr])

        inter.free(temps_addr)

        if computed_not_a:
            inter.free(nota_addr)

    @staticmethod
    def __halfAdder(sim: simulator.SerialSimulator, a_addr: int, b_addr: int, s_addr: int, cout_addr: int, inter):
        """
        Performs a half-adder on the given columns
        :param sim: the simulation environment
        :param a_addr: the index of input a
        :param b_addr: the index of input b
        :param s_addr: the index of the output sum
        :param cout_addr: the index of the output carry
        :param inter: addresses for inter. Either np array or IntermediateAllocator.
        """

        if isinstance(inter, np.ndarray):
            inter = SerialArithmetic.IntermediateAllocator(inter)

        temps_addr = inter.malloc(3)

        sim.perform(constants.GateType.INIT1, [], [temps_addr[0]])
        sim.perform(constants.GateType.NOR, [a_addr, b_addr], [temps_addr[0]])

        sim.perform(constants.GateType.INIT1, [], [temps_addr[1]])
        sim.perform(constants.GateType.NOT, [a_addr], [temps_addr[1]])

        sim.perform(constants.GateType.INIT1, [], [temps_addr[2]])
        sim.perform(constants.GateType.NOT, [b_addr], [temps_addr[2]])

        sim.perform(constants.GateType.INIT1, [], [cout_addr])
        sim.perform(constants.GateType.NOR, [temps_addr[1], temps_addr[2]], [cout_addr])

        sim.perform(constants.GateType.INIT1, [], [s_addr])
        sim.perform(constants.GateType.NOR, [temps_addr[0], cout_addr], [s_addr])

        inter.free(temps_addr)

    @staticmethod
    def __fullAdder(sim: simulator.SerialSimulator, a_addr: int, b_addr: int, c_addr: int, s_addr: int,
            cout_addr: int, inter):
        """
        Performs a full-adder on the given columns
        :param sim: the simulation environment
        :param a_addr: the index of input a
        :param b_addr: the index of input b
        :param c_addr: the index of input c
        :param s_addr: the index of the output sum
        :param cout_addr: the index of the output carry
        :param inter: addresses for inter. Either np array or IntermediateAllocator.
        """

        if isinstance(inter, np.ndarray):
            inter = SerialArithmetic.IntermediateAllocator(inter)

        temps_addr = inter.malloc(4)

        sim.perform(constants.GateType.INIT1, [], [temps_addr[0]])
        sim.perform(constants.GateType.NOR, [a_addr, b_addr], [temps_addr[0]])

        sim.perform(constants.GateType.INIT1, [], [temps_addr[1]])
        sim.perform(constants.GateType.NOR, [a_addr, temps_addr[0]], [temps_addr[1]])

        sim.perform(constants.GateType.INIT1, [], [temps_addr[2]])
        sim.perform(constants.GateType.NOR, [b_addr, temps_addr[0]], [temps_addr[2]])

        sim.perform(constants.GateType.INIT1, [], [temps_addr[3]])
        sim.perform(constants.GateType.NOR, [temps_addr[1], temps_addr[2]], [temps_addr[3]])

        sim.perform(constants.GateType.INIT1, [], [temps_addr[1]])
        sim.perform(constants.GateType.NOR, [temps_addr[3], c_addr], [temps_addr[1]])

        sim.perform(constants.GateType.INIT1, [], [temps_addr[2]])
        sim.perform(constants.GateType.NOR, [temps_addr[1], temps_addr[3]], [temps_addr[2]])

        sim.perform(constants.GateType.INIT1, [], [temps_addr[3]])
        sim.perform(constants.GateType.NOR, [temps_addr[1], c_addr], [temps_addr[3]])

        sim.perform(constants.GateType.INIT1, [], [s_addr])
        sim.perform(constants.GateType.NOR, [temps_addr[3], temps_addr[2]], [s_addr])

        sim.perform(constants.GateType.INIT1, [], [cout_addr])
        sim.perform(constants.GateType.NOR, [temps_addr[0], temps_addr[1]], [cout_addr])

        inter.free(temps_addr)

    @staticmethod
    def __fullSubtractor(sim: simulator.SerialSimulator, a_addr: int, b_addr: int, c_addr: int, s_addr: int,
            cout_addr: int, inter):
        """
        Performs a full-subtractor on the given columns
        :param sim: the simulation environment
        :param a_addr: the index of input a
        :param b_addr: the index of input b
        :param c_addr: the index of input c
        :param s_addr: the index of the output sum
        :param cout_addr: the index of the output carry
        :param inter: addresses for inter. Either np array or IntermediateAllocator.
        """

        if isinstance(inter, np.ndarray):
            inter = SerialArithmetic.IntermediateAllocator(inter)

        notb_addr = inter.malloc(1)

        sim.perform(constants.GateType.INIT1, [], [notb_addr])
        sim.perform(constants.GateType.NOT, [b_addr], [notb_addr])

        SerialArithmetic.__fullAdder(sim, a_addr, notb_addr, c_addr, s_addr, cout_addr, inter)

        inter.free(notb_addr)

    @staticmethod
    def __fixedAddBit(sim: simulator.SerialSimulator, x_addr: np.ndarray, z_addr: np.ndarray, inter,
            cin_addr, cout_addr=None):
        """
        Adds a single bit to the given number using half-adders
        Cycles: 1 + N * HA = 10N + 1
        :param sim: the simulation environment
        :param x_addr: the addresses of input x (N-bit)
        :param z_addr: the addresses for the output z (N-bit)
        :param inter: addresses for inter. Either np array or IntermediateAllocator.
        :param cin_addr: the address for the input carry. "-1" designates constant 1 input carry.
        :param cout_addr: the address for an optional output carry
        """

        N = len(x_addr)
        assert (len(z_addr) == N)

        if isinstance(inter, np.ndarray):
            inter = SerialArithmetic.IntermediateAllocator(inter)

        carry_addr = inter.malloc(1)

        if cin_addr == -1:
            cin_addr = carry_addr
            sim.perform(constants.GateType.INIT1, [], [carry_addr])

        # Initialize the carry out
        if cout_addr is None:
            cout_addr = carry_addr

        for i in range(N):

            in_carry_addr = carry_addr if i > 0 else cin_addr
            out_carry_addr = carry_addr if i < N - 1 else cout_addr

            SerialArithmetic.__halfAdder(sim, x_addr[i], in_carry_addr, z_addr[i], out_carry_addr, inter)

        inter.free(carry_addr)

    @staticmethod
    def __abs(sim: simulator.SerialSimulator, x_addr: np.ndarray, z_addr: np.ndarray, inter):
        """
        Computes the absolute value of the given fixed-point number.
        Cycles: 3 + (N-1) * (XOR(given not b) + HA) = 3 + (N-1) * (18) = 18N - 15
        :param sim: the simulation environment
        :param x_addr: the addresses of input x (N-bit)
        :param z_addr: the addresses for the output z (N-bit)
        :param inter: addresses for inter. Either np array or IntermediateAllocator.
        """

        N = len(x_addr)
        assert (len(z_addr) == N)

        if isinstance(inter, np.ndarray):
            inter = SerialArithmetic.IntermediateAllocator(inter)

        not_msb_addr = inter.malloc(1)
        sim.perform(constants.GateType.INIT1, [], [not_msb_addr])
        sim.perform(constants.GateType.NOT, [x_addr[-1]], [not_msb_addr])

        carry_addr = inter.malloc(1)

        for i in range(N - 1):

            in_carry_addr = carry_addr if i > 0 else x_addr[-1]
            out_carry_addr = carry_addr

            xor_addr = inter.malloc(1)

            SerialArithmetic.__xor(sim, x_addr[i], x_addr[-1], xor_addr, inter, notb_addr=not_msb_addr)

            SerialArithmetic.__halfAdder(sim, xor_addr, in_carry_addr, z_addr[i], out_carry_addr, inter)

            inter.free(xor_addr)

        sim.perform(constants.GateType.INIT0, [], [z_addr[-1]])

        inter.free(not_msb_addr)
        inter.free(carry_addr)

    @staticmethod
    def __reduceOR(sim: simulator.SerialSimulator, x_addr: np.ndarray, z_addr: int, inter, notz_addr=None):
        """
        Performs an OR reduction on the x bits, storing the result in z.
        Cycles: 3 + N
        :param sim: the simulation environment
        :param x_addr: the addresses of the input columns
        :param z_addr: the address of the output column
        :param inter: addresses for inter. Either np array or IntermediateAllocator.
        :param notz_addr: the index of the optional output which stores the not of z
        """

        if isinstance(inter, np.ndarray):
            inter = SerialArithmetic.IntermediateAllocator(inter)

        # Performed using De Morgan

        custom_nor_out_addr = notz_addr is None
        if custom_nor_out_addr:
            notz_addr = inter.malloc(1)

        sim.perform(constants.GateType.INIT1, [], [notz_addr])
        for x in x_addr:
            sim.perform(constants.GateType.NOT, [x], [notz_addr])

        sim.perform(constants.GateType.INIT1, [], [z_addr])
        sim.perform(constants.GateType.NOT, [notz_addr], [z_addr])

        if custom_nor_out_addr:
            inter.free(notz_addr)

    @staticmethod
    def __variableShift(sim: simulator.SerialSimulator, x_addr: np.ndarray, t_addr: np.ndarray, inter,
            sticky_addr=None, direction=False):
        """
        Performs the in-place variable shift operation on the given columns.
        Cycles:
            Without sticky:
                Sum j from 0 to Nt - 1 of (2 + (Nx - 2^j) * MUX + (2^j))
                = Sum j from 0 to Nt - 1 of (2 + (Nx - 2^j) * 6 + (2^j))
                = Nt * (6 * Nx + 2) - 5 * 2^Nt + 5
            With sticky:
                Sum j from 0 to Nt - 1 of (2 + ReduceOr(2^j) + 1 + OR + (Nx - 2^j) * MUX + (2^j))
                = Sum j from 0 to Nt - 1 of (2 + (3 + 2^j) + 1 + 4 + (Nx - 2^j) * 6 + (2^j))
                = Nt * (6 * Nx + 10) - 4 * (2^Nt - 1)
        :param sim: the simulation environment
        :param x_addr: the addresses of input & output x (Nx-bit)
        :param t_addr: the addresses of input t (Nt-bit)
        :param inter: addresses for inter. Either np array or IntermediateAllocator.
        :param sticky_addr: an optional column for a sticky bit (OR of all of the bits that were truncated).
        :param direction: the direction of the shift. False is right-shift, and True is left-shift.
        """

        Nx = len(x_addr)
        log2_Nx = ceil(log2(Nx))
        Nt = len(t_addr)
        assert (Nt <= log2_Nx)

        if direction:
            x_addr = np.flip(x_addr)

        if isinstance(inter, np.ndarray):
            inter = SerialArithmetic.IntermediateAllocator(inter)

        for j in range(Nt):

            not_tj = inter.malloc(1)
            sim.perform(constants.GateType.INIT1, [], [not_tj])
            sim.perform(constants.GateType.NOT, [t_addr[j]], [not_tj])

            if sticky_addr is not None:

                # Compute the OR of the bits that are potentially lost in this step
                or_addr = inter.malloc(1)
                SerialArithmetic.__reduceOR(sim, x_addr[:2 ** j], or_addr, inter)
                # Compute the AND with whether the shift actually occurs
                sim.perform(constants.GateType.NOT, [not_tj], [or_addr])  # X-MAGIC
                # Compute the OR with the current sticky bit
                SerialArithmetic.__or(sim, sticky_addr, or_addr, sticky_addr, inter)
                inter.free(or_addr)

            for i in range(Nx - 2 ** j):
                SerialArithmetic.__mux(sim, t_addr[j], x_addr[i + (2 ** j)], x_addr[i], x_addr[i], inter, nota_addr=not_tj)

            for i in range(max(Nx - 2 ** j, 0), Nx):
                sim.perform(constants.GateType.NOT, [t_addr[j]], [x_addr[i]])  # X-MAGIC

            inter.free(not_tj)

    @staticmethod
    def __normalizeShift(sim: simulator.SerialSimulator, x_addr: np.ndarray, t_addr: np.ndarray, inter, direction=False):
        """
        Performs the in-place normalize shift operation on the given columns
        Cycles:
            (Sum j from 0 to Nt - 1 of (ReduceOr(2^j) + (Nx - 2^j) * MUX + (2^j))) + 2 + Nt
            = (Sum j from 0 to Nt - 1 of ((3 + 2^j) + (Nx - 2^j) * 6 + (2^j))) + 2 + Nt
            = Nt * (6 * Nx + 4) - 4 * 2^Nt + 6
        :param sim: the simulation environment
        :param x_addr: the addresses of input & output x (Nx-bit)
        :param t_addr: the addresses of output t (Nt-bit)
        :param inter: addresses for inter. Either np array or IntermediateAllocator.
        :param direction: the direction of the shift. False is right-shift, and True is left-shift.
        """

        Nx = len(x_addr)
        Nt = len(t_addr)

        if direction:
            x_addr = np.flip(x_addr)

        if isinstance(inter, np.ndarray):
            inter = SerialArithmetic.IntermediateAllocator(inter)

        for j in reversed(range(Nt)):

            not_tj = inter.malloc(1)

            SerialArithmetic.__reduceOR(sim, x_addr[:(2 ** j)], not_tj, inter, notz_addr=t_addr[j])

            for i in range(Nx - 2 ** j):
                SerialArithmetic.__mux(sim, t_addr[j], x_addr[i + (2 ** j)], x_addr[i], x_addr[i], inter, nota_addr=not_tj)

            for i in range(max(Nx - 2 ** j, 0), Nx):
                sim.perform(constants.GateType.NOT, [t_addr[j]], [x_addr[i]])  # X-MAGIC

            inter.free(not_tj)

        # If didn't contain any ones, then we define shift amount as zero
        not_lsb_addr = inter.malloc(1)
        sim.perform(constants.GateType.INIT1, [], [not_lsb_addr])
        sim.perform(constants.GateType.NOT, [x_addr[0]], [not_lsb_addr])
        for t in t_addr:
            sim.perform(constants.GateType.NOT, [not_lsb_addr], [t])  # X-MAGIC
        inter.free(not_lsb_addr)
