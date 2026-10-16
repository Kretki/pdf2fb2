"""PDF -> FB2 converter.

Pipeline (orchestrated in ``pipeline.conversion_pipeline``):
  1. Render PDF pages (PyMuPDF), in this venv.
  2. Segment pages with PP-DocLayoutV3 -> ``workers/layout_worker`` (PaddlePaddle venv).
  3. Read text/title/table/footnote segments with Qianfan-OCR -> ``workers/ocr_worker`` (PyTorch venv).
  4. Detect reader marks with picture segments blanked -> ``workers/readermarks_worker``; segment and
     OCR the cleaned pages again and merge them with the first-pass pictures (``pipeline.mark_rescan``).
  5. Assemble an FB2 book (sections, tables, images, footnotes, cover), in this venv.
  6. Draw segmentation boxes per page and run sanity checks.

Workers are separate processes with their own virtual environments and talk to
this package only through JSON job files (``worker_protocol``).
"""
