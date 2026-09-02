"""Line-HTR training package.

Submodules are imported directly. This init stays empty so
``pipeline.training.infer`` / ``serve`` can load in the Azure backend image
without pulling curriculum (which needs ``pipeline.dataset``).
"""
