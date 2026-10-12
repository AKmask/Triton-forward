"""Triton kernels for 3-D VTI interior/PML propagation and sparse operations.

All flattened field pointers follow [z, y, x] order, with x varying fastest.
"""
