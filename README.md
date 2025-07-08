# Surface Metal Organic Framework Cavity

This is a repository of the code that was used to generate the figures in [^1]
To reproduce the figures please run the respective plot script from the root directory.

We use `uv` to make dependency management convenient and reproducible. Make sure `uv` is installed. Then run 

```bash
uv sync
```

## Generating Plots

If you'd like to plot **Figure 1** run
```
uv run qnmsc/plot_story_opener.py
```

To plot **Figure 2** run
```
uv run qnmsc/plot_observable.py
```

For **Figure 3 and 5** run
```
uv run qnmsc/plot_trajectories.py
```

For **Figure 4** run
```
uv run qnmsc/plot_core_shell.py
```

For **Figure 6** run
```
uv run qnmsc/plot_splitting.py
```

## Generating the Data

To generate the data for Figures 1,2,3 and 4 run
```
uv run qnmsc/track_qnms.py
```

## Generating the Core Shell Data and RSE overlaps

TODO

## Solving the inverse eigenproblem
If access to the resonant state fields is not available one can phenomenologically extract the coupling coefficients from the resonance frequencies alone. The procedure is explained in the SI. The implementation can be found in `qnmsc/inverse_eigenproblem.py`

