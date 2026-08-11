# Execution Backend Support

This directory contains low-level programs and helpers used by the remote/local execution backend, including the OpenMP thread-sweep driver.

Files here should document the compile command, runtime environment variables, expected stdout/stderr format, and how the harness invokes them.

These components implement execution mechanics only; experiment interpretation belongs in `experiments/`.
