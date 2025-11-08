# Bottle Capacity Estimator

## Working principle
Segment bottle → rotate vertical via PCA → crop → integrate cross-sections assuming solid of revolution → convert px→mm with ArUco or Gemini scale.

## Expected outcome
mL capacity and mm height, plus outline overlay image and JSON dump.

## Challenges
Reflective edges, thin handles, severe perspective skew, missing scale cues, and poor lighting.

## Papers referred
Classical PCA-based orientation; radial integration for solids of revolution; ArUco detection (Garrido-Jurado et al.).

## Observations
If ArUco is present, accuracy improves markedly. Gemini can be helpful but uncertain; treat as heuristic unless a known-size object is visible.
