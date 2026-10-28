# Tests

## Purpose

This directory contains automated checks for the cleaned pipeline.

## Test organization

`unit/` contains fast tests built from temporary synthetic inputs. Future
integration tests should live in `integration/` and state their external model
or dataset requirements explicitly.

## How to run the tests

Run the complete suite from the repository root:

    PYTHONPATH=src python -m unittest discover -s tests -v

The unit suite uses temporary synthetic files and mocked external programs. It
does not download checkpoints, separate real datasets, or run the complete
MusicFM model. Real-model smoke tests and full output-equivalence checks are
tracked separately in the component documentation.
