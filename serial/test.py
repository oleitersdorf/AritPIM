import os
os.chdir('./serial')
import sys
sys.path.insert(0, '..')
sys.path.insert(0, '.')

import numpy as np
import unittest
from util import representation, IO
import simulator
import AritPIM


class TestSerial(unittest.TestCase):
    """
    Tests the proposed bit-serial algorithms to verify the correctness and measure performance.
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
        x_addr = np.arange(0, N)
        y_addr = np.arange(N, 2 * N)
        z_addr = np.arange(2 * N, 3 * N)
        inter_addr = np.arange(3 * N, num_cols)

        # Define the simulator
        sim = simulator.SerialSimulator(n, num_cols)

        # Sample the inputs at random
        x = np.random.randint(low=-(1 << (N - 1)), high=(1 << (N - 1)), size=(1, n), dtype=np.longlong)
        y = np.random.randint(low=-(1 << (N - 1)), high=(1 << (N - 1)), size=(1, n), dtype=np.longlong)

        # Write the inputs to the memory
        sim.memory[x_addr] = representation.signedToBinaryFixed(x, N)
        sim.memory[y_addr] = representation.signedToBinaryFixed(y, N)

        # Perform the addition algorithm
        AritPIM.SerialArithmetic.fixedAddition(sim, x_addr, y_addr, z_addr, inter_addr)

        # Read the outputs from the memory
        z = representation.binaryToSignedFixed(sim.memory[z_addr])

        # Verify correctness
        expected = x + y
        # Generate mask to avoid cases where an overflow occurred
        mask = np.logical_and(expected >= -(1 << (N - 1)), expected < (1 << (N - 1)))
        self.assertTrue(((z == expected)[mask]).all())

        # Print and save results
        header = f'Serial Fixed Addition: {N}-bit with {sim.latency} cycles and {sim.maxUsed + 1} cells total.'
        print(header)
        IO.serialSaveLog(header, 'Supports both unsigned and signed numbers.',
            {'x': x_addr, 'y': y_addr}, {'z': z_addr}, sim.getLog(), '../output/serial_fixed_addition.txt')

    def test_fixedSubtraction(self):
        """
        Tests the fixed-point subtraction algorithm (signed).
        """

        # Parameters
        n = 1 << 20
        N = 32
        num_cols = 1024

        # Address allocation
        x_addr = np.arange(0, N)
        y_addr = np.arange(N, 2 * N)
        z_addr = np.arange(2 * N, 3 * N)
        inter_addr = np.arange(3 * N, num_cols)

        # Define the simulator
        sim = simulator.SerialSimulator(n, num_cols)

        # Sample the inputs at random
        x = np.random.randint(low=-(1 << (N - 1)), high=(1 << (N - 1)), size=(1, n), dtype=np.longlong)
        y = np.random.randint(low=-(1 << (N - 1)), high=(1 << (N - 1)), size=(1, n), dtype=np.longlong)

        # Write the inputs to the memory
        sim.memory[x_addr] = representation.signedToBinaryFixed(x, N)
        sim.memory[y_addr] = representation.signedToBinaryFixed(y, N)

        # Perform the subtraction algorithm
        AritPIM.SerialArithmetic.fixedSubtraction(sim, x_addr, y_addr, z_addr, inter_addr)

        # Read the outputs from the memory
        z = representation.binaryToSignedFixed(sim.memory[z_addr])

        # Verify correctness
        expected = x - y
        # Generate mask to avoid cases where an overflow occurred
        mask = np.logical_and(expected >= -(1 << (N - 1)), expected < (1 << (N - 1)))
        self.assertTrue(((z == expected)[mask]).all())

        # Print and save results
        header = f'Serial Fixed Subtraction: {N}-bit with {sim.latency} cycles and {sim.maxUsed + 1} cells total.'
        print(header)
        IO.serialSaveLog(header, 'Supports both unsigned and signed numbers.',
            {'x': x_addr, 'y': y_addr}, {'z': z_addr}, sim.getLog(), '../output/serial_fixed_subtraction.txt')

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
        x_addr = np.arange(0, N)
        y_addr = np.arange(N, 2 * N)
        z_addr = np.arange(2 * N, 4 * N)
        inter_addr = np.arange(4 * N, num_cols)

        # Define the simulator
        sim = simulator.SerialSimulator(n, num_cols)

        # Sample the inputs at random
        x = np.random.randint(low=0, high=(1 << N), size=(1, n), dtype=np.ulonglong)
        y = np.random.randint(low=0, high=(1 << N), size=(1, n), dtype=np.ulonglong)

        # Write the inputs to the memory
        sim.memory[x_addr] = representation.unsignedToBinaryFixed(x, N)
        sim.memory[y_addr] = representation.unsignedToBinaryFixed(y, N)

        # Perform the multiplication algorithm
        AritPIM.SerialArithmetic.fixedMultiplication(sim, x_addr, y_addr, z_addr, inter_addr)

        # Read the outputs from the memory
        z = representation.binaryToUnsignedFixed(sim.memory[z_addr])

        # Verify correctness
        expected = x * y
        self.assertTrue((z == expected).all())

        # Print and save results
        header = f'Serial Fixed Multiplication: {N}-bit with {sim.latency} cycles and {sim.maxUsed + 1} cells total.'
        print(header)
        IO.serialSaveLog(header, 'Supports only unsigned numbers; can be extended to signed numbers.',
            {'x': x_addr, 'y': y_addr}, {'z': z_addr}, sim.getLog(), '../output/serial_fixed_multiplication.txt')

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
        z_addr = np.arange(0, 2 * N)
        d_addr = np.arange(2 * N, 3 * N)
        q_addr = np.arange(3 * N, 4 * N)
        r_addr = np.arange(4 * N, 5 * N)
        inter_addr = np.arange(5 * N, num_cols)

        # Define the simulator
        sim = simulator.SerialSimulator(n, num_cols)

        # Sample the inputs at random
        z = np.random.randint(low=0, high=(1 << (2 * N)), size=(1, n), dtype=np.ulonglong)
        d = np.random.randint(low=1, high=(1 << N), size=(1, n), dtype=np.ulonglong)

        # Write the inputs to the memory
        sim.memory[z_addr] = representation.unsignedToBinaryFixed(z, 2 * N)
        sim.memory[d_addr] = representation.unsignedToBinaryFixed(d, N)

        # Perform the division algorithm
        AritPIM.SerialArithmetic.fixedDivision(sim, z_addr, d_addr, q_addr, r_addr, inter_addr)

        # Read the outputs from the memory
        q = representation.binaryToUnsignedFixed(sim.memory[q_addr])
        r = representation.binaryToUnsignedFixed(sim.memory[r_addr])

        # Verify correctness
        expected_q = z // d
        expected_r = z % d
        # Generate mask to avoid cases where an overflow occurred
        mask = (z >> N) < d
        self.assertTrue((q == expected_q)[mask].all())
        self.assertTrue((r == expected_r)[mask].all())

        # Print and save results
        header = f'Serial Fixed Division: {N}-bit with {sim.latency} cycles and {sim.maxUsed + 1} cells total.'
        print(header)
        IO.serialSaveLog(header, 'Supports only unsigned numbers; can be extended to signed numbers.',
            {'z': z_addr, 'd': d_addr}, {'q': q_addr, 'r': r_addr}, sim.getLog(), '../output/serial_fixed_division.txt')

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
        xe_addr = np.arange(0, Ne)
        xm_addr = np.arange(Ne, N)
        ye_addr = np.arange(N, N + Ne)
        ym_addr = np.arange(N + Ne, 2 * N)
        ze_addr = np.arange(2 * N, 2 * N + Ne)
        zm_addr = np.arange(2 * N + Ne, 3 * N)
        inter_addr = np.arange(3 * N, num_cols)

        # Define the simulator
        sim = simulator.SerialSimulator(n, num_cols)

        # Sample the inputs at random
        xe = np.random.randint(low=0, high=1 << Ne, size=(1, n), dtype=np.longlong)
        xm = np.random.randint(low=1 << Nm, high=1 << (Nm + 1), size=(1, n), dtype=np.longlong)
        ye = np.random.randint(low=0, high=1 << Ne, size=(1, n), dtype=np.longlong)
        ym = np.random.randint(low=1 << Nm, high=1 << (Nm + 1), size=(1, n), dtype=np.longlong)

        # Avoid subnormals
        xm = np.where(xe == 0, 0, xm)
        ym = np.where(ye == 0, 0, ym)

        # Write the inputs to the memory
        sim.memory[xe_addr] = representation.unsignedToBinaryFixed(xe, Ne)
        sim.memory[xm_addr] = representation.unsignedToBinaryFixed(np.where(xe == 0, 0, xm - (1 << Nm)), Nm)
        sim.memory[ye_addr] = representation.unsignedToBinaryFixed(ye, Ne)
        sim.memory[ym_addr] = representation.unsignedToBinaryFixed(np.where(ye == 0, 0, ym - (1 << Nm)), Nm)

        # Perform the addition algorithm
        AritPIM.SerialArithmetic.floatingAdditionUnsigned(sim, xe_addr, xm_addr, ye_addr, ym_addr, ze_addr, zm_addr, inter_addr)

        # Read the outputs from the memory
        ze = representation.binaryToUnsignedFixed(sim.memory[ze_addr]).astype(np.longlong)
        zm = representation.binaryToUnsignedFixed(sim.memory[zm_addr]).astype(np.longlong)
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
        header = f'Serial Floating Addition Unsigned: {Ne}-bit exponent and {Nm}-bit mantissa with {sim.latency} cycles and {sim.maxUsed + 1} cells total.'
        print(header)
        IO.serialSaveLog(header, 'Supports only unsigned numbers.',
            {'xe': xe_addr, 'xm': xm_addr, 'ye': ye_addr, 'ym': ym_addr},
            {'ze': ze_addr, 'zm': zm_addr}, sim.getLog(), '../output/serial_floating_addition_unsigned.txt')

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
        xs_addr = np.arange(0, Ns)
        xe_addr = np.arange(Ns, Ns + Ne)
        xm_addr = np.arange(Ns + Ne, N)
        ys_addr = np.arange(N, N + Ns)
        ye_addr = np.arange(N + Ns, N + Ns + Ne)
        ym_addr = np.arange(N + Ns + Ne, 2 * N)
        zs_addr = np.arange(2 * N, 2 * N + Ns)
        ze_addr = np.arange(2 * N + Ns, 2 * N + Ns + Ne)
        zm_addr = np.arange(2 * N + Ns + Ne, 3 * N)
        inter_addr = np.arange(3 * N, num_cols)

        # Define the simulator
        sim = simulator.SerialSimulator(n, num_cols)

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
        ys = np.where(ye == 0, 0, ys)

        # Write the inputs to the memory
        sim.memory[xs_addr] = xs
        sim.memory[xe_addr] = representation.unsignedToBinaryFixed(xe, Ne)
        sim.memory[xm_addr] = representation.unsignedToBinaryFixed(np.where(xe == 0, 0, xm - (1 << Nm)), Nm)
        sim.memory[ys_addr] = ys
        sim.memory[ye_addr] = representation.unsignedToBinaryFixed(ye, Ne)
        sim.memory[ym_addr] = representation.unsignedToBinaryFixed(np.where(ye == 0, 0, ym - (1 << Nm)), Nm)

        # Perform the addition algorithm
        AritPIM.SerialArithmetic.floatingAdditionSigned(sim, xs_addr, xe_addr, xm_addr, ys_addr, ye_addr, ym_addr,
            zs_addr, ze_addr, zm_addr, inter_addr)

        # Read the outputs from the memory
        zs = sim.memory[zs_addr]
        ze = representation.binaryToUnsignedFixed(sim.memory[ze_addr]).astype(np.longlong)
        zm = representation.binaryToUnsignedFixed(sim.memory[zm_addr]).astype(np.longlong)
        zm += (ze != 0) << Nm
        z = representation.composeSignedFloat(zs, ze, zm).astype(np.float32)

        # Verify correctness
        np.seterr(over='ignore')
        x = representation.composeSignedFloat(xs, xe, xm)
        y = representation.composeSignedFloat(ys, ye, ym)
        expected = (x + y).astype(np.float32)
        mask = np.logical_and(np.logical_not(np.isinf(expected)), np.logical_or(expected == 0, np.abs(expected) >= np.finfo(np.float32).tiny))

        self.assertTrue((z == expected)[mask].all())

        # Print and save results
        header = f'Serial Floating Addition Signed: {Ns}-bit sign, {Ne}-bit exponent, and {Nm}-bit mantissa with {sim.latency} cycles and {sim.maxUsed + 1} cells total.'
        print(header)
        IO.serialSaveLog(header, 'Supports only signed numbers.',
            {'xs': xs_addr, 'xe': xe_addr, 'xm': xm_addr, 'ys': ys_addr, 'ye': ye_addr, 'ym': ym_addr},
            {'zs': zs_addr, 'ze': ze_addr, 'zm': zm_addr}, sim.getLog(), '../output/serial_floating_addition_signed.txt')

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
        xs_addr = np.arange(0, Ns)
        xe_addr = np.arange(Ns, Ns + Ne)
        xm_addr = np.arange(Ns + Ne, N)
        ys_addr = np.arange(N, N + Ns)
        ye_addr = np.arange(N + Ns, N + Ns + Ne)
        ym_addr = np.arange(N + Ns + Ne, 2 * N)
        zs_addr = np.arange(2 * N, 2 * N + Ns)
        ze_addr = np.arange(2 * N + Ns, 2 * N + Ns + Ne)
        zm_addr = np.arange(2 * N + Ns + Ne, 3 * N)
        inter_addr = np.arange(3 * N, num_cols)

        # Define the simulator
        sim = simulator.SerialSimulator(n, num_cols)

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
        sim.memory[xs_addr] = xs
        sim.memory[xe_addr] = representation.unsignedToBinaryFixed(xe, Ne)
        sim.memory[xm_addr] = representation.unsignedToBinaryFixed(np.where(xe == 0, 0, xm - (1 << Nm)), Nm)
        sim.memory[ys_addr] = ys
        sim.memory[ye_addr] = representation.unsignedToBinaryFixed(ye, Ne)
        sim.memory[ym_addr] = representation.unsignedToBinaryFixed(np.where(ye == 0, 0, ym - (1 << Nm)), Nm)

        # Perform the multiplication algorithm
        AritPIM.SerialArithmetic.floatingMultiplication(sim, xs_addr, xe_addr, xm_addr, ys_addr, ye_addr, ym_addr,
            zs_addr, ze_addr, zm_addr, inter_addr)

        # Read the outputs from the memory
        zs = sim.memory[zs_addr]
        ze = representation.binaryToUnsignedFixed(sim.memory[ze_addr]).astype(np.longlong)
        zm = representation.binaryToUnsignedFixed(sim.memory[zm_addr]).astype(np.longlong)
        zm += (ze != 0) << Nm
        z = representation.composeSignedFloat(zs, ze, zm).astype(np.float32)

        # Verify correctness
        np.seterr(over='ignore')
        x = representation.composeSignedFloat(xs, xe, xm)
        y = representation.composeSignedFloat(ys, ye, ym)
        expected = (x * y).astype(np.float32)
        mask = np.logical_and(np.logical_not(np.isinf(expected)), np.logical_or(expected == 0, np.abs(expected) >= np.finfo(np.float32).tiny))

        self.assertTrue((z == expected)[mask].all())

        # Print and save results
        header = f'Serial Floating Multiplication: {Ns}-bit sign, {Ne}-bit exponent, and {Nm}-bit mantissa with {sim.latency} cycles and {sim.maxUsed + 1} cells total.'
        print(header)
        IO.serialSaveLog(header, 'Supports only signed numbers.',
            {'xs': xs_addr, 'xe': xe_addr, 'xm': xm_addr, 'ys': ys_addr, 'ye': ye_addr, 'ym': ym_addr},
            {'zs': zs_addr, 'ze': ze_addr, 'zm': zm_addr}, sim.getLog(), '../output/serial_floating_multiplication.txt')

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
        xs_addr = np.arange(0, Ns)
        xe_addr = np.arange(Ns, Ns + Ne)
        xm_addr = np.arange(Ns + Ne, N)
        ys_addr = np.arange(N, N + Ns)
        ye_addr = np.arange(N + Ns, N + Ns + Ne)
        ym_addr = np.arange(N + Ns + Ne, 2 * N)
        zs_addr = np.arange(2 * N, 2 * N + Ns)
        ze_addr = np.arange(2 * N + Ns, 2 * N + Ns + Ne)
        zm_addr = np.arange(2 * N + Ns + Ne, 3 * N)
        inter_addr = np.arange(3 * N, num_cols)

        # Define the simulator
        sim = simulator.SerialSimulator(n, num_cols)

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
        sim.memory[xs_addr] = xs
        sim.memory[xe_addr] = representation.unsignedToBinaryFixed(xe, Ne)
        sim.memory[xm_addr] = representation.unsignedToBinaryFixed(np.where(xe == 0, 0, xm - (1 << Nm)), Nm)
        sim.memory[ys_addr] = ys
        sim.memory[ye_addr] = representation.unsignedToBinaryFixed(ye, Ne)
        sim.memory[ym_addr] = representation.unsignedToBinaryFixed(np.where(ye == 0, 0, ym - (1 << Nm)), Nm)

        # Perform the division algorithm
        AritPIM.SerialArithmetic.floatingDivision(sim, xs_addr, xe_addr, xm_addr, ys_addr, ye_addr, ym_addr,
            zs_addr, ze_addr, zm_addr, inter_addr)

        # Read the outputs from the memory
        zs = sim.memory[zs_addr]
        ze = representation.binaryToUnsignedFixed(sim.memory[ze_addr]).astype(np.longlong)
        zm = representation.binaryToUnsignedFixed(sim.memory[zm_addr]).astype(np.longlong)
        zm += (ze != 0) << Nm
        z = representation.composeSignedFloat(zs, ze, zm).astype(np.float32)

        # Verify correctness
        np.seterr(over='ignore', divide='ignore')
        x = representation.composeSignedFloat(xs, xe, xm)
        y = representation.composeSignedFloat(ys, ye, ym)
        expected = (x / y).astype(np.float32)
        mask = np.logical_and(np.logical_not(np.isinf(expected)), np.logical_or(expected == 0, np.abs(expected) >= np.finfo(np.float32).tiny))

        self.assertTrue((z == expected)[mask].all())

        # Print and save results
        header = f'Serial Floating Division: {Ns}-bit sign, {Ne}-bit exponent, and {Nm}-bit mantissa with {sim.latency} cycles and {sim.maxUsed + 1} cells total.'
        print(header)
        IO.serialSaveLog(header, 'Supports only signed numbers.',
            {'xs': xs_addr, 'xe': xe_addr, 'xm': xm_addr, 'ys': ys_addr, 'ye': ye_addr, 'ym': ym_addr},
            {'zs': zs_addr, 'ze': ze_addr, 'zm': zm_addr}, sim.getLog(), '../output/serial_floating_division.txt')

    def test_fixedSystem(self):
        """
        System test for single-row fixed-point arithmetic
        """

        # Parameters
        n = 1 << 20
        N = 32
        num_cols = 1024

        # Address allocation
        x0_addr = np.arange(0, N)
        x1_addr = np.arange(N, 2 * N)
        x2_addr = np.arange(2 * N, 3 * N)
        x3_addr = np.arange(3 * N, 4 * N)
        y0_addr = np.arange(4 * N, 5 * N)
        y1_addr = np.arange(5 * N, 6 * N)
        y2_addr = np.arange(6 * N, 7 * N)
        y3_addr = np.arange(7 * N, 8 * N)
        z_addr = np.arange(8 * N, 10 * N)
        inter_addr = np.arange(10 * N, num_cols)

        # Define the simulator
        sim = simulator.SerialSimulator(n, num_cols)

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
        sim.memory[x0_addr] = representation.unsignedToBinaryFixed(x0, N)
        sim.memory[x1_addr] = representation.unsignedToBinaryFixed(x1, N)
        sim.memory[x2_addr] = representation.unsignedToBinaryFixed(x2, N)
        sim.memory[x3_addr] = representation.unsignedToBinaryFixed(x3, N)
        sim.memory[y0_addr] = representation.unsignedToBinaryFixed(y0, N)
        sim.memory[y1_addr] = representation.unsignedToBinaryFixed(y1, N)
        sim.memory[y2_addr] = representation.unsignedToBinaryFixed(y2, N)
        sim.memory[y3_addr] = representation.unsignedToBinaryFixed(y3, N)

        # Test 1: Dot Product

        temp = inter_addr[:2 * N]
        AritPIM.SerialArithmetic.fixedMultiplication(sim, x0_addr, y0_addr, z_addr, inter_addr[2 * N:])
        AritPIM.SerialArithmetic.fixedMultiplication(sim, x1_addr, y1_addr, temp, inter_addr[2 * N:])
        AritPIM.SerialArithmetic.fixedAddition(sim, z_addr, temp, z_addr, inter_addr[2 * N:])
        AritPIM.SerialArithmetic.fixedMultiplication(sim, x2_addr, y2_addr, temp, inter_addr[2 * N:])
        AritPIM.SerialArithmetic.fixedAddition(sim, z_addr, temp, z_addr, inter_addr[2 * N:])
        AritPIM.SerialArithmetic.fixedMultiplication(sim, x3_addr, y3_addr, temp, inter_addr[2 * N:])
        AritPIM.SerialArithmetic.fixedAddition(sim, z_addr, temp, z_addr, inter_addr[2 * N:])

        # Read the outputs from the memory
        z = representation.binaryToUnsignedFixed(sim.memory[z_addr])

        # Verify correctness
        np.seterr(over='ignore')
        expected = (x0 * y0) + (x1 * y1) + (x2 * y2) + (x3 * y3)
        # Generate mask to avoid cases where an overflow occurred
        mask = np.bitwise_and(expected >= 0, expected < (1 << (2 * N)))
        self.assertTrue(((z == expected)[mask]).all())

        # Test 2: Variant of Dot Product with Division and Subtraction

        temp = inter_addr[:N]
        rem = inter_addr[N:2 * N]
        AritPIM.SerialArithmetic.fixedDivision(sim, np.concatenate((x0_addr, x1_addr)), y0_addr, z_addr[:N], rem, inter_addr[2 * N:])
        AritPIM.SerialArithmetic.fixedDivision(sim, np.concatenate((x2_addr, x3_addr)), y1_addr, temp, rem, inter_addr[2 * N:])
        AritPIM.SerialArithmetic.fixedSubtraction(sim, z_addr[:N], temp, z_addr[:N], inter_addr[2 * N:])
        # Read the outputs from the memory
        z = representation.binaryToSignedFixed(sim.memory[z_addr[:N]])

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
        x0_addr = np.arange(0, N)
        x1_addr = np.arange(N, 2 * N)
        x2_addr = np.arange(2 * N, 3 * N)
        x3_addr = np.arange(3 * N, 4 * N)
        y0_addr = np.arange(4 * N, 5 * N)
        y1_addr = np.arange(5 * N, 6 * N)
        y2_addr = np.arange(6 * N, 7 * N)
        y3_addr = np.arange(7 * N, 8 * N)
        z_addr = np.arange(8 * N, 9 * N)
        inter_addr = np.arange(9 * N, num_cols)

        # Define the simulator
        sim = simulator.SerialSimulator(n, num_cols)

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
        sim.memory[x0_addr] = representation.signedFloatToBinary(x0)
        sim.memory[x1_addr] = representation.signedFloatToBinary(x1)
        sim.memory[x2_addr] = representation.signedFloatToBinary(x2)
        sim.memory[x3_addr] = representation.signedFloatToBinary(x3)
        sim.memory[y0_addr] = representation.signedFloatToBinary(y0)
        sim.memory[y1_addr] = representation.signedFloatToBinary(y1)
        sim.memory[y2_addr] = representation.signedFloatToBinary(y2)
        sim.memory[y3_addr] = representation.signedFloatToBinary(y3)

        # Test 1: Dot Product

        temp = inter_addr[:N]
        temp2 = inter_addr[N:2*N]
        AritPIM.SerialArithmetic.floatingMultiplicationIEEE(sim, x0_addr, y0_addr, temp2, inter_addr[2*N:])
        AritPIM.SerialArithmetic.floatingMultiplicationIEEE(sim, x1_addr, y1_addr, temp, inter_addr[2*N:])
        AritPIM.SerialArithmetic.floatingAdditionSignedIEEE(sim, temp2, temp, z_addr, inter_addr[2*N:])
        AritPIM.SerialArithmetic.floatingMultiplicationIEEE(sim, x2_addr, y2_addr, temp, inter_addr[2*N:])
        AritPIM.SerialArithmetic.floatingAdditionSignedIEEE(sim, z_addr, temp, temp2, inter_addr[2*N:])
        AritPIM.SerialArithmetic.floatingMultiplicationIEEE(sim, x3_addr, y3_addr, temp, inter_addr[2*N:])
        AritPIM.SerialArithmetic.floatingAdditionSignedIEEE(sim, temp2, temp, z_addr, inter_addr[2*N:])

        # Read the outputs from the memory
        z = representation.binaryToSignedFloat(sim.memory[z_addr]).astype(np.float32)

        # Verify correctness
        np.seterr(over='ignore')
        expected = (x0 * y0) + (x1 * y1) + (x2 * y2) + (x3 * y3)
        # Generate mask to avoid cases where an overflow occurred
        mask = np.logical_and(np.logical_not(np.isinf(expected)), np.logical_or(expected == 0, np.abs(expected) >= np.finfo(np.float32).tiny))
        self.assertTrue(((z == expected)[mask]).all())

        # Test 2: Variant of Dot Product with Division and Subtraction

        temp = inter_addr[:N]
        temp2 = inter_addr[N:2 * N]
        AritPIM.SerialArithmetic.floatingDivisionIEEE(sim, x0_addr, y0_addr, temp2, inter_addr[2 * N:])
        AritPIM.SerialArithmetic.floatingDivisionIEEE(sim, x1_addr, y1_addr, temp, inter_addr[2 * N:])
        AritPIM.SerialArithmetic.floatingSubtractionSignedIEEE(sim, temp2, temp, z_addr, inter_addr[2 * N:])
        AritPIM.SerialArithmetic.floatingDivisionIEEE(sim, x2_addr, y2_addr, temp, inter_addr[2 * N:])
        AritPIM.SerialArithmetic.floatingSubtractionSignedIEEE(sim, z_addr, temp, temp2, inter_addr[2 * N:])
        AritPIM.SerialArithmetic.floatingDivisionIEEE(sim, x3_addr, y3_addr, temp, inter_addr[2 * N:])
        AritPIM.SerialArithmetic.floatingSubtractionSignedIEEE(sim, temp2, temp, z_addr, inter_addr[2 * N:])

        # Read the outputs from the memory
        z = representation.binaryToSignedFloat(sim.memory[z_addr]).astype(np.float32)

        # Verify correctness
        np.seterr(over='ignore', divide='ignore')
        expected = (x0 / y0) - (x1 / y1) - (x2 / y2) - (x3 / y3)
        # Generate mask to avoid cases where an overflow occurred
        mask = np.logical_and(np.logical_not(np.isinf(expected)),
                              np.logical_or(expected == 0, np.abs(expected) >= np.finfo(np.float32).tiny))
        self.assertTrue(((z == expected)[mask]).all())


if __name__ == '__main__':
    unittest.main()

