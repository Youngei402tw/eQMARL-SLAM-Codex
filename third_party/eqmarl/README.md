# Pinned eQMARL source extracts

Original authors: Alexander DeRieux and Walid Saad, Virginia Tech.
Source: <https://github.com/news-vt/eqmarl/tree/6b1b2e3817f661da81aad5e7188a362380dbdd1d>.
License: CC BY 4.0, retained in [LICENSE.md](LICENSE.md).

The source extracts in this directory are unchanged. `maa2c.py` was originally
`eqmarl/algorithms/maa2c.py`. They are documentation/reference inputs, not
imports in the new training package. The new `oracle.py`, `quantum.py`, and
faithful learner arithmetic adapt their gate/encoding/readout/objective
definitions to NumPy and Torch/PennyLane. Changes include float64 precision,
explicit dimensions, new validation/resource reporting, and an explicitly
empty CZ ring for one-wire diagnostic partitions. Corrected heads/learner terms
and recurrent models are new, separately named paths.
