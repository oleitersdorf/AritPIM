import os
os.chdir('./parallel')
import sys
sys.path.insert(0, '..')
sys.path.insert(0, '.')

import numpy as np
import unittest
from util import representation, IO
import simulator
import AritPIM


class TestParallel(unittest.TestCase):
    """
    Tests the proposed bit-parallel algorithms to verify the correctness and measure performance.
    """

    def test_fixedAddition(self):
        """
        Tests the fixed-point addition algorithm (signed).
        """

        # Parameters
        n = 1 << 20
        N = 32
        num_cols = 1024

        # Address allocation
        x_addr = 0
        y_addr = 1
        z_addr = 2
        inter_addr = np.arange(3, num_cols // N)
        partitions = np.arange(N)

        # Define the simulator
        sim = simulator.ParallelSimulator(n, num_cols, N)

        # Sample the inputs at random
        x = np.random.randint(low=-(1 << (N - 1)), high=(1 << (N - 1)), size=(1, n), dtype=np.longlong)
        y = np.random.randint(low=-(1 << (N - 1)), high=(1 << (N - 1)), size=(1, n), dtype=np.longlong)

        # Write the inputs to the memory
        sim.write(x_addr, representation.signedToBinaryFixed(x, N))
        sim.write(y_addr, representation.signedToBinaryFixed(y, N))

        # Perform the addition algorithm
        AritPIM.ParallelArithmetic.fixedAddition(sim, x_addr, y_addr, z_addr, inter_addr, partitions)

        # Read the outputs from the memory
        z = representation.binaryToSignedFixed(sim.read(z_addr))

        # Verify correctness
        expected = x + y
        # Generate mask to avoid cases where an overflow occurred
        mask = np.logical_and(expected >= -(1 << (N - 1)), expected < (1 << (N - 1)))
        self.assertTrue(((z == expected)[mask]).all())

        # Print and save results
        header = f'Parallel Fixed Addition: {N}-bit with {sim.latency} cycles, {sim.energy} gates, and {(sim.maxUsed + 1) * N} cells total.'
        print(header)
        IO.parallelSaveLog(header, 'Supports both unsigned and signed numbers.',
            {'x': (x_addr, partitions), 'y': (y_addr, partitions)},
            {'z': (z_addr, partitions)}, sim.getLog(), '../output/parallel_fixed_addition.txt')

    def test_fixedSubtraction(self):
        """
        Tests the fixed-point subtraction algorithm (signed).
        """

        # Parameters
        n = 1 << 20
        N = 32
        num_cols = 1024

        # Address allocation
        x_addr = 0
        y_addr = 1
        z_addr = 2
        inter_addr = np.arange(3, num_cols // N)
        partitions = np.arange(N)

        # Define the simulator
        sim = simulator.ParallelSimulator(n, num_cols, N)

        # Sample the inputs at random
        x = np.random.randint(low=-(1 << (N - 1)), high=(1 << (N - 1)), size=(1, n), dtype=np.longlong)
        y = np.random.randint(low=-(1 << (N - 1)), high=(1 << (N - 1)), size=(1, n), dtype=np.longlong)

        # Write the inputs to the memory
        sim.write(x_addr, representation.signedToBinaryFixed(x, N))
        sim.write(y_addr, representation.signedToBinaryFixed(y, N))

        # Perform the subtraction algorithm
        AritPIM.ParallelArithmetic.fixedSubtraction(sim, x_addr, y_addr, z_addr, inter_addr, partitions)

        # Read the outputs from the memory
        z = representation.binaryToSignedFixed(sim.read(z_addr))

        # Verify correctness
        expected = x - y
        # Generate mask to avoid cases where an overflow occurred
        mask = np.logical_and(expected >= -(1 << (N - 1)), expected < (1 << (N - 1)))
        self.assertTrue(((z == expected)[mask]).all())

        # Print and save results
        header = f'Parallel Fixed Subtraction: {N}-bit with {sim.latency} cycles, {sim.energy} gates, and {(sim.maxUsed + 1) * N} cells total.'
        print(header)
        IO.parallelSaveLog(header, 'Supports both unsigned and signed numbers.',
            {'x': (x_addr, partitions), 'y': (y_addr, partitions)},
            {'z': (z_addr, partitions)}, sim.getLog(), '../output/parallel_fixed_subtraction.txt')

    def test_fixedMultiplication(self):
        """
        Tests the fixed-point multiplication algorithm (unsigned).
        """

        # Parameters
        # Note: N <= 32 even though the algorithm supports larger N
        # as numpy's long does not support results larger than 64-bit
        n = 1 << 20
        N = 32
        num_cols = 1024

        # Address allocation
        x_addr = 0
        y_addr = 1
        z_addr = 2
        w_addr = 3
        inter_addr = np.arange(4, num_cols // N)
        partitions = np.arange(N)

        # Define the simulator
        sim = simulator.ParallelSimulator(n, num_cols, N)

        # Sample the inputs at random
        x = np.random.randint(low=0, high=(1 << N), size=(1, n), dtype=np.ulonglong)
        y = np.random.randint(low=0, high=(1 << N), size=(1, n), dtype=np.ulonglong)

        # Write the inputs to the memory
        sim.write(x_addr, representation.unsignedToBinaryFixed(x, N))
        sim.write(y_addr, representation.unsignedToBinaryFixed(y, N))

        # Perform the multiplication algorithm
        AritPIM.ParallelArithmetic.fixedMultiplication(sim, x_addr, y_addr, z_addr, w_addr, inter_addr, partitions)

        # Read the outputs from the memory
        z = representation.binaryToUnsignedFixed(sim.read(z_addr))
        w = representation.binaryToUnsignedFixed(sim.read(w_addr))

        # Verify correctness
        expected = x * y
        self.assertTrue(((z + (w << N)) == expected).all())

        # Print and save results
        header = f'Parallel Fixed Multiplication: {N}-bit with {sim.latency} cycles, {sim.energy} gates, and {(sim.maxUsed + 1) * N} cells total.'
        print(header)
        IO.parallelSaveLog(header, 'Supports only unsigned numbers; can be extended to signed numbers.',
            {'x': (x_addr, partitions), 'y': (y_addr, partitions)},
            {'z': (z_addr, partitions), 'w': (w_addr, partitions)}, sim.getLog(), '../output/parallel_fixed_multiplication.txt')

    def test_fixedDivision(self):
        """
        Tests the fixed-point division algorithm (unsigned).
        """

        # Parameters
        # Note: N <= 32 even though the algorithm supports larger N
        # as numpy's long does not support results larger than 64-bit
        n = 1 << 20
        N = 32
        num_cols = 1024

        # Address allocation
        w_addr = 0
        z_addr = 1
        d_addr = 2
        q_addr = 3
        r_addr = 4
        inter_addr = np.arange(5, num_cols // N)
        partitions = np.arange(N)

        # Define the simulator
        sim = simulator.ParallelSimulator(n, num_cols, N)

        # Sample the inputs at random
        z = np.random.randint(low=0, high=(1 << (2 * N)), size=(1, n), dtype=np.ulonglong)
        d = np.random.randint(low=1, high=(1 << N), size=(1, n), dtype=np.ulonglong)

        # Write the inputs to the memory
        sim.write(w_addr, representation.unsignedToBinaryFixed(z >> N, N))
        sim.write(z_addr, representation.unsignedToBinaryFixed(z % (1 << N), N))
        sim.write(d_addr, representation.unsignedToBinaryFixed(d, N))

        # Perform the division algorithm
        AritPIM.ParallelArithmetic.fixedDivision(sim, w_addr, z_addr, d_addr, q_addr, r_addr, inter_addr, partitions)

        # Read the outputs from the memory
        q = representation.binaryToUnsignedFixed(sim.read(q_addr))
        r = representation.binaryToUnsignedFixed(sim.read(r_addr))

        # Verify correctness
        expected_q = z // d
        expected_r = z % d
        # Generate mask to avoid cases where an overflow occurred
        mask = (z >> N) < d
        self.assertTrue((q == expected_q)[mask].all())
        self.assertTrue((r == expected_r)[mask].all())

        # Print and save results
        header = f'Parallel Fixed Division: {N}-bit with {sim.latency} cycles, {sim.energy} gates, and {(sim.maxUsed + 1) * N} cells total.'
        print(header)
        IO.parallelSaveLog(header, 'Supports only unsigned numbers; can be extended to signed numbers.',
            {'w': (w_addr, partitions), 'z': (z_addr, partitions), 'd': (d_addr, partitions)},
            {'q': (q_addr, partitions), 'r': (r_addr, partitions)}, sim.getLog(), '../output/parallel_fixed_division.txt')

    def test_floatingAdditionUnsigned(self):
        """
        Tests the floating-point addition algorithm (unsigned).
        """

        # Parameters
        n = 1 << 20
        Ne = 8
        Nm = 23
        N = Ne + Nm
        num_cols = 1024

        # Address allocation
        x_addr = 0
        y_addr = 1
        z_addr = 2
        inter_addr = np.arange(3, num_cols // N)
        partitions = np.arange(N)
        e_partitions = np.arange(0, Ne)
        m_partitions = np.arange(Ne, N)

        # Define the simulator
        sim = simulator.ParallelSimulator(n, num_cols, N)

        # Sample the inputs at random
        xe = np.random.randint(low=0, high=1 << Ne, size=(1, n), dtype=np.longlong)
        xm = np.random.randint(low=1 << Nm, high=1 << (Nm + 1), size=(1, n), dtype=np.longlong)
        ye = np.random.randint(low=0, high=1 << Ne, size=(1, n), dtype=np.longlong)
        ym = np.random.randint(low=1 << Nm, high=1 << (Nm + 1), size=(1, n), dtype=np.longlong)

        # Avoid subnormals
        xm = np.where(xe == 0, 0, xm)
        ym = np.where(ye == 0, 0, ym)

        # Write the inputs to the memory
        sim.write(x_addr, representation.unsignedToBinaryFixed(xe, Ne), e_partitions)
        sim.write(x_addr, representation.unsignedToBinaryFixed(xm, Nm), m_partitions)
        sim.write(y_addr, representation.unsignedToBinaryFixed(ye, Ne), e_partitions)
        sim.write(y_addr, representation.unsignedToBinaryFixed(ym, Nm), m_partitions)

        # Perform the addition algorithm
        AritPIM.ParallelArithmetic.floatingAdditionUnsigned(sim, Ne, Nm, x_addr, y_addr, z_addr, inter_addr, partitions)

        # Read the outputs from the memory
        ze = representation.binaryToUnsignedFixed(sim.read(z_addr, e_partitions)).astype(np.longlong)
        zm = representation.binaryToUnsignedFixed(sim.read(z_addr, m_partitions)).astype(np.longlong)
        zm += (ze != 0) << Nm
        z = representation.composeUnsignedFloat(ze, zm).astype(np.float32)

        # Verify correctness
        np.seterr(over='ignore')
        x = representation.composeUnsignedFloat(xe, xm)
        y = representation.composeUnsignedFloat(ye, ym)
        expected = (x + y).astype(np.float32)
        mask = np.logical_not(np.isinf(expected))

        self.assertTrue((z == expected)[mask].all())

        # Print and save results
        header = f'Parallel Floating Addition Unsigned: {Ne}-bit exponent, and {Nm}-bit mantissa with ' \
                 f'{sim.latency} cycles, {sim.energy} gates, and {(sim.maxUsed + 1) * N} cells total.'
        print(header)
        IO.parallelSaveLog(header, 'Supports only unsigned numbers.',
                           {'xe': (x_addr, e_partitions), 'xm': (x_addr, m_partitions),
                            'ye': (y_addr, e_partitions), 'ym': (y_addr, m_partitions)},
                           {'ze': (z_addr, e_partitions), 'zm': (z_addr, m_partitions)},
                           sim.getLog(), '../output/parallel_floating_addition_unsigned.txt')

    def test_floatingAdditionSigned(self):
        """
        Tests the floating-point addition algorithm (signed).
        """

        # Parameters
        n = 1 << 20
        Ns = 1
        Ne = 8
        Nm = 23
        N = Ns + Ne + Nm
        num_cols = 1024

        # Address allocation
        x_addr = 0
        y_addr = 1
        z_addr = 2
        inter_addr = np.arange(3, num_cols // N)
        partitions = np.arange(N)
        s_partitions = np.arange(0, Ns)
        e_partitions = np.arange(Ns, Ns + Ne)
        m_partitions = np.arange(Ns + Ne, N)

        # Define the simulator
        sim = simulator.ParallelSimulator(n, num_cols, N)

        # Sample the inputs at random
        xe = np.random.randint(low=0, high=1 << Ne, size=(1, n), dtype=np.longlong)
        xm = np.random.randint(low=1 << Nm, high=1 << (Nm + 1), size=(1, n), dtype=np.longlong)
        xs = np.random.randint(low=0, high=2, size=(1, n), dtype=np.longlong)
        ye = np.random.randint(low=0, high=1 << Ne, size=(1, n), dtype=np.longlong)
        ym = np.random.randint(low=1 << Nm, high=1 << (Nm + 1), size=(1, n), dtype=np.longlong)
        ys = np.random.randint(low=0, high=2, size=(1, n), dtype=np.longlong)

        # Avoid subnormals
        xm = np.where(xe == 0, 0, xm)
        xs = np.where(xe == 0, 0, xs)
        ym = np.where(ye == 0, 0, ym)
        ys = np.where(ys == 0, 0, ys)

        # Write the inputs to the memory
        sim.write(x_addr, representation.unsignedToBinaryFixed(xs, Ns), s_partitions)
        sim.write(x_addr, representation.unsignedToBinaryFixed(xe, Ne), e_partitions)
        sim.write(x_addr, representation.unsignedToBinaryFixed(xm, Nm), m_partitions)
        sim.write(y_addr, representation.unsignedToBinaryFixed(ys, Ns), s_partitions)
        sim.write(y_addr, representation.unsignedToBinaryFixed(ye, Ne), e_partitions)
        sim.write(y_addr, representation.unsignedToBinaryFixed(ym, Nm), m_partitions)

        # Perform the addition algorithm
        AritPIM.ParallelArithmetic.floatingAdditionSigned(sim, Ne, Nm, x_addr, y_addr, z_addr, inter_addr, partitions)

        # Read the outputs from the memory
        zs = sim.read(z_addr, s_partitions)
        ze = representation.binaryToUnsignedFixed(sim.read(z_addr, e_partitions)).astype(np.longlong)
        zm = representation.binaryToUnsignedFixed(sim.read(z_addr, m_partitions)).astype(np.longlong)
        zm += (ze != 0) << Nm
        z = representation.composeSignedFloat(zs, ze, zm).astype(np.float32)

        # Verify correctness
        np.seterr(over='ignore')
        x = representation.composeSignedFloat(xs, xe, xm)
        y = representation.composeSignedFloat(ys, ye, ym)
        expected = (x + y).astype(np.float32)
        mask = np.logical_and(np.logical_not(np.isinf(expected)),
                          np.logical_or(expected == 0, np.abs(expected) >= np.finfo(np.float32).tiny))

        self.assertTrue((z == expected)[mask].all())

        # Print and save results
        header = f'Parallel Floating Addition Signed: {Ns}-bit sign, {Ne}-bit exponent, and {Nm}-bit mantissa with ' \
                 f'{sim.latency} cycles, {sim.energy} gates, and {(sim.maxUsed + 1) * N} cells total.'
        print(header)
        IO.parallelSaveLog(header, 'Supports only signed numbers.',
            {'xs': (x_addr, s_partitions), 'xe': (x_addr, e_partitions), 'xm': (x_addr, m_partitions),
             'ys': (y_addr, s_partitions), 'ye': (y_addr, e_partitions), 'ym': (y_addr, m_partitions)},
            {'zs': (z_addr, s_partitions), 'ze': (z_addr, e_partitions), 'zm': (z_addr, m_partitions)},
            sim.getLog(), '../output/parallel_floating_addition_signed.txt')

    def test_floatingMultiplication(self):
        """
        Tests the floating-point multiplication algorithm (signed).
        """

        # Parameters
        n = 1 << 20
        Ns = 1
        Ne = 8
        Nm = 23
        N = Ns + Ne + Nm
        num_cols = 1024

        # Address allocation
        x_addr = 0
        y_addr = 1
        z_addr = 2
        inter_addr = np.arange(3, num_cols // N)
        partitions = np.arange(N)
        s_partitions = np.arange(0, Ns)
        e_partitions = np.arange(Ns, Ns + Ne)
        m_partitions = np.arange(Ns + Ne, N)

        # Define the simulator
        sim = simulator.ParallelSimulator(n, num_cols, N)

        # Sample the inputs at random
        xe = np.random.randint(low=0, high=1 << Ne, size=(1, n), dtype=np.longlong)
        xm = np.random.randint(low=1 << Nm, high=1 << (Nm + 1), size=(1, n), dtype=np.longlong)
        xs = np.random.randint(low=0, high=2, size=(1, n), dtype=np.longlong)
        ye = np.random.randint(low=0, high=1 << Ne, size=(1, n), dtype=np.longlong)
        ym = np.random.randint(low=1 << Nm, high=1 << (Nm + 1), size=(1, n), dtype=np.longlong)
        ys = np.random.randint(low=0, high=2, size=(1, n), dtype=np.longlong)

        # Avoid subnormals
        xm = np.where(xe == 0, 0, xm)
        ym = np.where(ye == 0, 0, ym)

        # Write the inputs to the memory
        sim.write(x_addr, representation.unsignedToBinaryFixed(xs, Ns), s_partitions)
        sim.write(x_addr, representation.unsignedToBinaryFixed(xe, Ne), e_partitions)
        sim.write(x_addr, representation.unsignedToBinaryFixed(xm, Nm), m_partitions)
        sim.write(y_addr, representation.unsignedToBinaryFixed(ys, Ns), s_partitions)
        sim.write(y_addr, representation.unsignedToBinaryFixed(ye, Ne), e_partitions)
        sim.write(y_addr, representation.unsignedToBinaryFixed(ym, Nm), m_partitions)

        # Perform the multiplication algorithm
        AritPIM.ParallelArithmetic.floatingMultiplication(sim, Ne, Nm, x_addr, y_addr, z_addr, inter_addr, partitions)

        # Read the outputs from the memory
        zs = sim.read(z_addr, s_partitions)
        ze = representation.binaryToUnsignedFixed(sim.read(z_addr, e_partitions)).astype(np.longlong)
        zm = representation.binaryToUnsignedFixed(sim.read(z_addr, m_partitions)).astype(np.longlong)
        zm += (ze != 0) << Nm
        z = representation.composeSignedFloat(zs, ze, zm).astype(np.float32)

        # Verify correctness
        np.seterr(over='ignore')
        x = representation.composeSignedFloat(xs, xe, xm)
        y = representation.composeSignedFloat(ys, ye, ym)
        expected = (x * y).astype(np.float32)
        mask = np.logical_and(np.logical_not(np.isinf(expected)),
                              np.logical_or(expected == 0, np.abs(expected) >= np.finfo(np.float32).tiny))

        self.assertTrue((z == expected)[mask].all())

        # Print and save results
        header = f'Parallel Floating Multiplication: {Ns}-bit sign, {Ne}-bit exponent, and {Nm}-bit mantissa with ' \
                 f'{sim.latency} cycles, {sim.energy} gates, and {(sim.maxUsed + 1) * N} cells total.'
        print(header)
        IO.parallelSaveLog(header, 'Supports only signed numbers.',
            {'xs': (x_addr, s_partitions), 'xe': (x_addr, e_partitions), 'xm': (x_addr, m_partitions),
             'ys': (y_addr, s_partitions), 'ye': (y_addr, e_partitions), 'ym': (y_addr, m_partitions)},
            {'zs': (z_addr, s_partitions), 'ze': (z_addr, e_partitions), 'zm': (z_addr, m_partitions)},
            sim.getLog(), '../output/parallel_floating_multiplication.txt')

    def test_floatingDivision(self):
        """
        Tests the floating-point division algorithm (signed).
        """

        # Parameters
        n = 1 << 20
        Ns = 1
        Ne = 8
        Nm = 23
        N = Ns + Ne + Nm
        num_cols = 1024

        # Address allocation
        x_addr = 0
        y_addr = 1
        z_addr = 2
        inter_addr = np.arange(3, num_cols // N)
        partitions = np.arange(N)
        s_partitions = np.arange(0, Ns)
        e_partitions = np.arange(Ns, Ns + Ne)
        m_partitions = np.arange(Ns + Ne, N)

        # Define the simulator
        sim = simulator.ParallelSimulator(n, num_cols, N)

        # Sample the inputs at random
        xe = np.random.randint(low=0, high=1 << Ne, size=(1, n), dtype=np.longlong)
        xm = np.random.randint(low=1 << Nm, high=1 << (Nm + 1), size=(1, n), dtype=np.longlong)
        xs = np.random.randint(low=0, high=2, size=(1, n), dtype=np.longlong)
        ye = np.random.randint(low=1, high=1 << Ne, size=(1, n), dtype=np.longlong)
        ym = np.random.randint(low=1 << Nm, high=1 << (Nm + 1), size=(1, n), dtype=np.longlong)
        ys = np.random.randint(low=0, high=2, size=(1, n), dtype=np.longlong)

        # Avoid subnormals
        xm = np.where(xe == 0, 0, xm)
        ym = np.where(ye == 0, 0, ym)

        # Write the inputs to the memory
        sim.write(x_addr, representation.unsignedToBinaryFixed(xs, Ns), s_partitions)
        sim.write(x_addr, representation.unsignedToBinaryFixed(xe, Ne), e_partitions)
        sim.write(x_addr, representation.unsignedToBinaryFixed(xm, Nm), m_partitions)
        sim.write(y_addr, representation.unsignedToBinaryFixed(ys, Ns), s_partitions)
        sim.write(y_addr, representation.unsignedToBinaryFixed(ye, Ne), e_partitions)
        sim.write(y_addr, representation.unsignedToBinaryFixed(ym, Nm), m_partitions)

        # Perform the division algorithm
        AritPIM.ParallelArithmetic.floatingDivision(sim, Ne, Nm, x_addr, y_addr, z_addr, inter_addr, partitions)

        # Read the outputs from the memory
        zs = sim.read(z_addr, s_partitions)
        ze = representation.binaryToUnsignedFixed(sim.read(z_addr, e_partitions)).astype(np.longlong)
        zm = representation.binaryToUnsignedFixed(sim.read(z_addr, m_partitions)).astype(np.longlong)
        zm += (ze != 0) << Nm
        z = representation.composeSignedFloat(zs, ze, zm).astype(np.float32)

        # Verify correctness
        np.seterr(over='ignore', divide='ignore')
        x = representation.composeSignedFloat(xs, xe, xm)
        y = representation.composeSignedFloat(ys, ye, ym)
        expected = (x / y).astype(np.float32)
        mask = np.logical_and(np.logical_not(np.isinf(expected)),
                              np.logical_or(expected == 0, np.abs(expected) >= np.finfo(np.float32).tiny))

        self.assertTrue((z == expected)[mask].all())

        # Print and save results
        header = f'Parallel Floating Division: {Ns}-bit sign, {Ne}-bit exponent, and {Nm}-bit mantissa with ' \
                 f'{sim.latency} cycles, {sim.energy} gates, and {(sim.maxUsed + 1) * N} cells total.'
        print(header)
        IO.parallelSaveLog(header, 'Supports only signed numbers.',
            {'xs': (x_addr, s_partitions), 'xe': (x_addr, e_partitions), 'xm': (x_addr, m_partitions),
             'ys': (y_addr, s_partitions), 'ye': (y_addr, e_partitions), 'ym': (y_addr, m_partitions)},
            {'zs': (z_addr, s_partitions), 'ze': (z_addr, e_partitions), 'zm': (z_addr, m_partitions)},
            sim.getLog(), '../output/parallel_floating_division.txt')

    def test_fixedSystem(self):
        """
        System test for single-row fixed-point arithmetic
        """

        # Parameters
        n = 1 << 20
        N = 32
        num_cols = 1024

        # Address allocation
        x0_addr = 0
        x1_addr = 1
        x2_addr = 2
        x3_addr = 3
        y0_addr = 4
        y1_addr = 5
        y2_addr = 6
        y3_addr = 7
        z_addr = 8
        w_addr = 9
        inter_addr = np.arange(10, num_cols // N)
        partitions = np.arange(N)

        # Define the simulator
        sim = simulator.ParallelSimulator(n, num_cols, N)

        # Sample the inputs at random
        x0 = np.random.randint(low=0, high=(1 << N), size=(1, n), dtype=np.longlong)
        x1 = np.random.randint(low=0, high=(1 << N), size=(1, n), dtype=np.longlong)
        x2 = np.random.randint(low=0, high=(1 << N), size=(1, n), dtype=np.longlong)
        x3 = np.random.randint(low=0, high=(1 << N), size=(1, n), dtype=np.longlong)
        y0 = np.random.randint(low=1, high=(1 << N), size=(1, n), dtype=np.longlong)
        y1 = np.random.randint(low=1, high=(1 << N), size=(1, n), dtype=np.longlong)
        y2 = np.random.randint(low=0, high=(1 << N), size=(1, n), dtype=np.longlong)
        y3 = np.random.randint(low=0, high=(1 << N), size=(1, n), dtype=np.longlong)

        # Write the inputs to the memory
        sim.write(x0_addr, representation.unsignedToBinaryFixed(x0, N), partitions)
        sim.write(x1_addr, representation.unsignedToBinaryFixed(x1, N), partitions)
        sim.write(x2_addr, representation.unsignedToBinaryFixed(x2, N), partitions)
        sim.write(x3_addr, representation.unsignedToBinaryFixed(x3, N), partitions)
        sim.write(y0_addr, representation.unsignedToBinaryFixed(y0, N), partitions)
        sim.write(y1_addr, representation.unsignedToBinaryFixed(y1, N), partitions)
        sim.write(y2_addr, representation.unsignedToBinaryFixed(y2, N), partitions)
        sim.write(y3_addr, representation.unsignedToBinaryFixed(y3, N), partitions)

        # Test 1: Dot Product

        tempz = inter_addr[0]
        tempw = inter_addr[1]
        AritPIM.ParallelArithmetic.fixedMultiplication(sim, x0_addr, y0_addr, z_addr, w_addr, inter_addr[2:], partitions)
        AritPIM.ParallelArithmetic.fixedMultiplication(sim, x1_addr, y1_addr, tempz, tempw, inter_addr[2:], partitions)
        AritPIM.ParallelArithmetic.fixedAddition(sim, z_addr, tempz, z_addr, inter_addr[2:], partitions)
        AritPIM.ParallelArithmetic.fixedMultiplication(sim, x2_addr, y2_addr, tempz, tempw, inter_addr[2:], partitions)
        AritPIM.ParallelArithmetic.fixedAddition(sim, z_addr, tempz, z_addr, inter_addr[2:], partitions)
        AritPIM.ParallelArithmetic.fixedMultiplication(sim, x3_addr, y3_addr, tempz, tempw, inter_addr[2:], partitions)
        AritPIM.ParallelArithmetic.fixedAddition(sim, z_addr, tempz, z_addr, inter_addr[2:], partitions)

        # Read the outputs from the memory
        z = representation.binaryToUnsignedFixed(sim.read(z_addr, partitions))

        # Verify correctness
        np.seterr(over='ignore')
        expected = (((x0 * y0) % (1 << N)) + ((x1 * y1) % (1 << N)) + ((x2 * y2) % (1 << N)) + ((x3 * y3) % (1 << N))) % (1 << N)
        # Generate mask to avoid cases where an overflow occurred
        mask = np.bitwise_and(expected >= 0, expected < (1 << (2 * N)))
        self.assertTrue(((z == expected)[mask]).all())

        # Test 2: Variant of Dot Product with Division and Subtraction

        temp = inter_addr[0]
        rem = inter_addr[1]
        AritPIM.ParallelArithmetic.fixedDivision(sim, x1_addr, x0_addr, y0_addr, z_addr, rem, inter_addr[2:], partitions)
        AritPIM.ParallelArithmetic.fixedDivision(sim, x3_addr, x2_addr, y1_addr, temp, rem, inter_addr[2:], partitions)
        AritPIM.ParallelArithmetic.fixedSubtraction(sim, z_addr, temp, z_addr, inter_addr[2:], partitions)

        # Read the outputs from the memory
        z = representation.binaryToSignedFixed(sim.read(z_addr, partitions))

        # Verify correctness
        np.seterr(over='ignore', divide='ignore')
        expected = ((x0.astype(np.uint64) + (x1.astype(np.uint64) << N)) // y0.astype(np.uint64))\
            - ((x2.astype(np.uint64) + (x3.astype(np.uint64) << N)) // y1.astype(np.uint64))
        # Generate mask to avoid cases where an overflow occurred
        mask = np.bitwise_and(np.bitwise_and(x1 < y0, x3 < y1), np.bitwise_and(expected >= -(1 << (N - 1)), expected < (1 << (N - 1))))
        self.assertTrue(((z == expected)[mask]).all())

    def test_floatingSystem(self):
        """
        System test for single-row floating-point arithmetic
        """

        # Parameters
        n = 1 << 20
        N = 32
        num_cols = 1024

        # Address allocation
        x0_addr = 0
        x1_addr = 1
        x2_addr = 2
        x3_addr = 3
        y0_addr = 4
        y1_addr = 5
        y2_addr = 6
        y3_addr = 7
        z_addr = 8
        inter_addr = np.arange(9, num_cols // N)
        partitions = np.arange(N)

        # Define the simulator
        sim = simulator.ParallelSimulator(n, num_cols, N)

        # Sample the inputs at random
        x0 = np.random.random((1, n)).astype(np.float32)
        x1 = np.random.random((1, n)).astype(np.float32)
        x2 = np.random.random((1, n)).astype(np.float32)
        x3 = np.random.random((1, n)).astype(np.float32)
        y0 = np.random.random((1, n)).astype(np.float32)
        y1 = np.random.random((1, n)).astype(np.float32)
        y2 = np.random.random((1, n)).astype(np.float32)
        y3 = np.random.random((1, n)).astype(np.float32)

        # Write the inputs to the memory
        sim.write(x0_addr, representation.signedFloatToBinary(x0), partitions)
        sim.write(x1_addr, representation.signedFloatToBinary(x1), partitions)
        sim.write(x2_addr, representation.signedFloatToBinary(x2), partitions)
        sim.write(x3_addr, representation.signedFloatToBinary(x3), partitions)
        sim.write(y0_addr, representation.signedFloatToBinary(y0), partitions)
        sim.write(y1_addr, representation.signedFloatToBinary(y1), partitions)
        sim.write(y2_addr, representation.signedFloatToBinary(y2), partitions)
        sim.write(y3_addr, representation.signedFloatToBinary(y3), partitions)

        # Test 1: Dot Product

        temp = inter_addr[0]
        temp2 = inter_addr[1]
        AritPIM.ParallelArithmetic.floatingMultiplicationIEEE(sim, x0_addr, y0_addr, temp2, inter_addr[2:], partitions)
        AritPIM.ParallelArithmetic.floatingMultiplicationIEEE(sim, x1_addr, y1_addr, temp, inter_addr[2:], partitions)
        AritPIM.ParallelArithmetic.floatingAdditionSignedIEEE(sim, temp2, temp, z_addr, inter_addr[2:], partitions)
        AritPIM.ParallelArithmetic.floatingMultiplicationIEEE(sim, x2_addr, y2_addr, temp, inter_addr[2:], partitions)
        AritPIM.ParallelArithmetic.floatingAdditionSignedIEEE(sim, z_addr, temp, temp2, inter_addr[2:], partitions)
        AritPIM.ParallelArithmetic.floatingMultiplicationIEEE(sim, x3_addr, y3_addr, temp, inter_addr[2:], partitions)
        AritPIM.ParallelArithmetic.floatingAdditionSignedIEEE(sim, temp2, temp, z_addr, inter_addr[2:], partitions)

        # Read the outputs from the memory
        z = representation.binaryToSignedFloat(sim.read(z_addr, partitions)).astype(np.float32)

        # Verify correctness
        np.seterr(over='ignore')
        expected = (x0 * y0) + (x1 * y1) + (x2 * y2) + (x3 * y3)
        # Generate mask to avoid cases where an overflow occurred
        mask = np.logical_and(np.logical_not(np.isinf(expected)), np.logical_or(expected == 0, np.abs(expected) >= np.finfo(np.float32).tiny))
        self.assertTrue(((z == expected)[mask]).all())

        # Test 2: Variant of Dot Product with Division and Subtraction

        temp = inter_addr[0]
        temp2 = inter_addr[1]
        AritPIM.ParallelArithmetic.floatingDivisionIEEE(sim, x0_addr, y0_addr, temp2, inter_addr[2:], partitions)
        AritPIM.ParallelArithmetic.floatingDivisionIEEE(sim, x1_addr, y1_addr, temp, inter_addr[2:], partitions)
        AritPIM.ParallelArithmetic.floatingSubtractionSignedIEEE(sim, temp2, temp, z_addr, inter_addr[2:], partitions)
        AritPIM.ParallelArithmetic.floatingDivisionIEEE(sim, x2_addr, y2_addr, temp, inter_addr[2:], partitions)
        AritPIM.ParallelArithmetic.floatingSubtractionSignedIEEE(sim, z_addr, temp, temp2, inter_addr[2:], partitions)
        AritPIM.ParallelArithmetic.floatingDivisionIEEE(sim, x3_addr, y3_addr, temp, inter_addr[2:], partitions)
        AritPIM.ParallelArithmetic.floatingSubtractionSignedIEEE(sim, temp2, temp, z_addr, inter_addr[2:], partitions)

        # Read the outputs from the memory
        z = representation.binaryToSignedFloat(sim.read(z_addr, partitions)).astype(np.float32)

        # Verify correctness
        np.seterr(over='ignore', divide='ignore')
        expected = (x0 / y0) - (x1 / y1) - (x2 / y2) - (x3 / y3)
        # Generate mask to avoid cases where an overflow occurred
        mask = np.logical_and(np.logical_not(np.isinf(expected)),
                              np.logical_or(expected == 0, np.abs(expected) >= np.finfo(np.float32).tiny))
        self.assertTrue(((z == expected)[mask]).all())


if __name__ == '__main__':
    unittest.main()
