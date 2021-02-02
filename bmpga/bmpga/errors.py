# coding=utf-8
"""Provides all the custom error classes we use in bmpga"""


class ServerNotFoundError(Exception):
    """Base exception raised when a QuenchClient cannot reach a server"""
    pass


class InvalidURIError(Exception):
    """Exception raised when a URI is invalid"""
    pass


class ParentsNotSameError(Exception):
    """Error Raised when parents selected for mating are not the same."""
    pass


class CantIdentifyHeadTail(Exception):
    """Error raised when the head and tail can't be automatically determined"""
    pass


class ClientError(Exception):
    """Error raised when the client crashes"""
    pass


# Potential errors
"""Errors/exceptions to be thrown by potentials"""


class MinimizeError(Exception):
    """Generic error thrown by failed minimizers, other errors should inherit from this"""
    pass


class DFTBError(MinimizeError):
    """Error thrown when DFTB code fails"""
    pass


class DFTExitedUnexpectedlyError(MinimizeError):
    """Error raised when the DFT code unexpectedly exits"""
    pass


class OptimizationNotConvergedError(MinimizeError):
    """
    Error to be raised whenever a minimization does not converge
    Should be handled by the quench client
    """
    pass
