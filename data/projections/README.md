# Serial-Projection Records

This directory stores the executable serial-projection experiment outputs.

For each accepted OpenMP program, the artifact should preserve:

- the original candidate identifier and source path;
- the mechanically projected source with OpenMP directives removed;
- the exact verification command/protocol used on the projection;
- the resulting correctness verdict;
- timing measurements for the projected program when collected.

The key requirement is that projection correctness is based on rerunning the verifier, not on a static prediction.
