# PathSnap-ML-Hackathon

ML-based GPS map matching using road candidates, an HMM, and Viterbi route reconstruction.

## Member 3: HMM + Viterbi

The current implementation lives in [`route_reconstruction/`](route_reconstruction/README.md). It contains the typed integration contract, geometric or classifier-scored emissions, directed network transitions, log-space Viterbi decoding, and recovery at gaps or infeasible transitions.

The repository currently has no committed Member 1 candidate-generation or Member 4 API implementation. The module interface is provisional; see the HMM guide before integrating it.

Run the synthetic automated tests with:

```bash
python3 -m unittest discover -s tests -v
```
