# Project basis and honest evaluation

This implementation follows the uploaded KolamNet report:
- A cell is encoded by a 4-bit side mask (N=1, E=2, S=4, W=8).
- The eight permitted masks are 0, 3, 5, 6, 9, 10, 12 and 15.
- The top-left quadrant is evolved and mirrored horizontally and vertically.
- Task A recreates a target using 0.7 similarity + 0.3 continuity.
- Task B uses 0.30 closure + 0.25 loops + 0.20 ink balance + 0.25 symmetry.
- GA operators: tournament selection, crossover, mutation and elitism.
- Baselines: hill climbing and random search.

The report's own results say hill climbing solves the separable synthetic recreation targets faster than GA, while symmetric GA achieves higher mean fitness than hill climbing on target-free generative design. It also states that its targets were synthetic geometric patterns, not a collection of traditional kolams. This code does not claim to reproduce the report's exact numerical tables: its benchmark runs must be executed locally, and the app displays measured results from those runs.

The term “accuracy” is not technically appropriate for this optimizer unless a labelled classification task is defined. Report fitness, target tile similarity, continuity, closure, loop score and runtime/evaluation count instead.
