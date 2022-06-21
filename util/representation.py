import numpy as np


def unsignedToBinaryFixed(x, N) -> np.ndarray:
    """
    Converts x to a binary representation with N bits
    :param x: tensor of unsigned integers of dimension 1xn
    :param N: the representation size (e.g., 32)
    :return: tensor of booleans of dimension Nxn. Index 0 stores the LSB.
    """

    n = x.shape[1]

    y = np.zeros((N, n), dtype=np.bool_)

    for i in range(N):
        y[i] = x & (1 << i)

    return y


def binaryToUnsignedFixed(y) -> np.ndarray:
    """
    Converts y from a binary representation with N bits to an integer representation
    :param y: tensor of booleans of dimension Nxn. Index 0 stores the LSB.
    :return: tensor of unsigned integers of dimension 1xn
    """

    N, n = y.shape

    x = np.zeros((1, n), dtype=np.ulonglong)

    for i in range(N):
        x += y[i].astype(np.ulonglong) << i

    return x


def signedToBinaryFixed(x, N) -> np.ndarray:
    """
    Converts x to a binary representation with N bits
    :param x: tensor of signed integers of dimension 1xn
    :param N: the representation size (e.g., 32)
    :return: tensor of booleans of dimension Nxn. Index 0 stores the LSB.
    """

    n = x.shape[1]

    y = np.zeros((N, n), dtype=np.bool_)
    x_abs = np.where(x >= 0, x, (1 << N) + x)  # unsigned number with the same binary representation

    for i in range(N):
        y[i] = x_abs & (1 << i)

    return y


def binaryToSignedFixed(y) -> np.ndarray:
    """
    Converts y from a binary representation with N bits to an integer representation
    :param y: tensor of booleans of dimension Nxn. Index 0 stores the LSB.
    :return: tensor of signed integers of dimension 1xn
    """

    N, n = y.shape

    # Generate unsigned number with the same binary representation
    x_abs = np.zeros((1, n), dtype=np.longlong)

    for i in range(N):
        x_abs += y[i].astype(np.longlong) << i

    return np.where(x_abs >= (1 << (N-1)), x_abs - (1 << N), x_abs)


def composeUnsignedFloat(exponent, significand):
    """
    Converts the given exponent and significand to numpy float (unsigned)
    :param exponent: integer between 0 and 255
    :param significand: 23-bit integer
    :return the numpy float representing the value
    """
    return np.ldexp((significand / (2 ** 23)), (exponent - 127))


def binaryToUnsignedFloat(y):
    """
    Converts the given binary representation to numpy float (unsigned)
    :param y: tensor of booleans of dimension 31xn.
    :return tensor of numpy floats of dimension 1xn.
    """
    exp = binaryToUnsignedFixed(y[:8]).astype(int)
    sig = binaryToUnsignedFixed(y[8:]).astype(int) + np.where(exp == 0, 0, (2 ** 23))
    return composeUnsignedFloat(exp + 1, sig / 2)


def decomposeUnsignedFloat(x):
    """
    Converts the given numpy float (unsigned) to exponent, and significand
    :param x: the given numpy float array
    :return exponent (integer between 0 and 255), significand (23-bit integer)
    """
    x = x.astype(np.float32)
    significand, exponent = np.frexp(np.abs(x))
    return np.where(significand == 0, 0, exponent + 127 - 1), (significand * (2 ** 24)).astype(int)


def unsignedFloatToBinary(x):
    """
    Converts the given numpy float (unsigned) to binary representation
    :param x: the given numpy float array
    :return tensor of booleans of dimension 32xn.
    """
    exp, sig = decomposeUnsignedFloat(x)
    return np.concatenate((unsignedToBinaryFixed(exp, 8), unsignedToBinaryFixed(sig, 23)))


def composeSignedFloat(sign, exponent, significand):
    """
    Converts the given sign, exponent, and significand to numpy float (signed)
    :param sign: boolean (true if negative)
    :param exponent: integer between 0 and 255
    :param significand: 23-bit integer
    :return the numpy float representing the value
    """
    return ((-1) ** sign) * composeUnsignedFloat(exponent, significand)


def binaryToSignedFloat(y):
    """
    Converts the given binary representation to numpy float (signed)
    :param y: tensor of booleans of dimension 32xn.
    :return tensor of numpy floats of dimension 1xn.
    """
    sign = y[0]
    exp = binaryToUnsignedFixed(y[1:1+8]).astype(int)
    sig = binaryToUnsignedFixed(y[1+8:]).astype(int) + np.where(exp == 0, 0, (2 ** 23))
    return composeSignedFloat(sign, exp + 1, sig / 2)


def decomposeSignedFloat(x):
    """
    Converts the given numpy float (signed) to sign, exponent, and significand
    :param x: the given numpy float array
    :return sign (true if negative), exponent (integer between 0 and 255), significand (23-bit integer)
    """
    x = x.astype(np.float32)
    sign = x < 0
    x = np.abs(x)
    exp, sig = decomposeUnsignedFloat(x)
    return sign, exp, sig


def signedFloatToBinary(x):
    """
    Converts the given numpy float (signed) to binary representation
    :param x: the given numpy float array
    :return tensor of booleans of dimension 32xn.
    """
    sign, exp, sig = decomposeSignedFloat(x)
    return np.concatenate((unsignedToBinaryFixed(sign, 1), unsignedToBinaryFixed(exp, 8), unsignedToBinaryFixed(sig, 23)))
