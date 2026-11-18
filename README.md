# PathSnap-ML-Hackathon

ML-based GPS map matching using road candidates, an HMM, and Viterbi route reconstruction.

## Member 3: HMM + Viterbi

The current implementation lives in [`route_reconstruction/`](route_reconstruction/README.md). It contains the typed integration contract, geometric or classifier-scored emissions, directed network transitions, log-space Viterbi decoding, and recovery at gaps or infeasible transitions. `build_map_matching_input` adapts the current `data_processing/` outputs without changing those modules.

Preprocessing and cross-validation model training code are present. A saved final classifier artifact, unlabeled inference feature builder, and Member 4 API/UI implementation are still needed for full application integration. See the HMM guide for the current adapter contract and limitations.

Run the synthetic automated tests with:

```bash
python3 -m unittest discover -s tests -v
```
