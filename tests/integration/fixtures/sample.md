# Engram test document

This document is used for integration testing of the Engram DoclingClient.

## Section One

This is the first section. It contains enough text to produce at least one chunk
when processed by the hybrid chunker. Docling should split this into meaningful
semantic units rather than arbitrary token windows.

## Section Two

This is the second section. Integration tests verify that both sections appear
in the chunked output, confirming that section boundaries are preserved across
the full parse → chunk → result pipeline.

## Section Three

Additional content to ensure the document is substantial enough to require
multiple chunks in Docling's output. Each chunk should carry non-empty text.
