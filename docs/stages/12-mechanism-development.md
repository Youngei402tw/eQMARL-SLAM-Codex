# Stage 12 — Equal-budget mechanism development and freeze

Status: running from exact Stage 11 completion commit `7442b2354fc5f630c3c0753697f60078682f7b10`.

The first registered phase gives each of five critics the same two learning rates, two seeds, 5,000 training primitives, and 30 greedy validation scenarios in the low-complementarity condition. Mean validation return across the two seeds selects one rate per method, with the lower rate winning exact ties. Every result, including failures, remains in the archive.

After rate selection, a fixed pilot matrix will run every method in additive, low, and high complementarity with three paired seeds and 10,000 primitives. Pilot outcomes may describe variance and feasibility but cannot change the primary contrast, methods, conditions, confirmatory seeds, or checkpoint rule. The mechanism test catalog is not available to either phase.

Stage gate: pending tuning, pilots, complete retention, budget reconciliation, and publication of the locked Stage 13 source/config commit before any mechanism-test access.
