"""Triton kernels for 2-D VTI propagation, source injection, and sampling.

Kernel pointers address flattened [z, x] arrays, with x varying fastest.
"""
