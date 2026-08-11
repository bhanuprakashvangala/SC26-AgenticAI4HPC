# Sanitizer and Race-Detection Records

This directory stores the independent parallel-correctness cross-checks produced with Archer/ThreadSanitizer or equivalent released tooling.

Each record should identify the candidate, tool/runtime configuration, raw verdict, and whether the reported race is attributable to user code.

Injected controls used to validate detector sensitivity should be clearly separated from generated-program results.
