"""Job-file contract shared by the pdf2fb2 orchestrator and its model workers.

The orchestrator and each model worker live in separate virtual environments
(PaddlePaddle and PyTorch cannot share one). They communicate only through JSON
job files described here, so this package must stay dependency-free.
"""
