from enum import Enum


class GateType(Enum):
    """
    Represents a type of logic gate.
    """

    NOT = 0
    NOR = 1
    INIT0 = 2
    INIT1 = 3


def getIEEE754Split(N: int):
    """
    Computes the split of N into Ns, Ne, Nm according to the IEEE 754 standard for floating-point numbers.
    Supports 16-bit, 32-bit, and 64-bit.
    :param N: the total size of the floating point number
    :return: Ns (number of sign bits), Ne (number of exponent bits), Nm (number of mantissa bits)
    """

    if N == 16:
        return 1, 5, 10
    elif N == 32:
        return 1, 8, 23
    elif N == 64:
        return 1, 11, 52
    else:
        assert False
