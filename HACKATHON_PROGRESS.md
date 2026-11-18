# Hackathon Progress

## Member 3: HMM + Viterbi

### Completed
- Checkpoint 1: Added dependency-free HMM data structures and the initial module contract in `route_reconstruction/types.py`.
- Checkpoint 2: Added configurable Gaussian distance likelihoods and support for calibrated candidate probabilities.
- Added ignore rules for local hackathon archives, raw trip data files, and macOS metadata.

### Pending
- Emission and transition probability models.
- Initial probabilities, log-space decoding, Viterbi and backtracking.
- Gap/disconnection handling, validation, tests, integration documentation.

### Interface status
- The repository currently contains no Member 1 candidate-generation or Member 4 API code. The structures are a provisional contract and will be documented for integration review.

### Validation
- Checkpoint 1 import/structure smoke check: passed using the public data structures.
- Checkpoint 2 numerical smoke checks: passed (Gaussian reference value, distance monotonicity, classifier-probability selection, and invalid inputs).
