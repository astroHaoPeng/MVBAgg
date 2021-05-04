# Multi-resolution Aggregated Output Spatial Regression with Gaussian Processes

Authors: Harrison Zhu*, Adam Howes, Owen van Eer, Maxime Rischard, Dino Sejdinovic, Seth Flaxman

*Corresponding author: harrison.zhu15@imperial.ac.uk

## Paper
Coming soon!

## Data
Data available upon request of the authors of Mateo-Sanchis, A., Piles, M., Muñoz-Marí, J., Adsuara, J.E., Pérez-Suay, A. and Camps-Valls, G., 2019. Synergistic integration of optical and microwave satellite data for crop yield estimation. Remote sensing of environment, 234, p.111460.

## Requirements

```python
tensorflow
gpflow
matplotlib
scipy
numpy
pyjson
contextily
geopandas
pandas
sklearn
tqdm
```

## Experiments

We apply `MVBAgg` to crop yield modelling and prediction.

The notebooks in `notebooks/` can be run to reproduce the results in the paper.

## Main GP Classes

Can be found in `src/svgp.py`. For example, one can easily use the `MultiResolutionSpatialVBagg` class for arbitrary likelihoods by switching the likelihood to a `gpflow` `Likelihood` object.

```python
m = MultiResolutionSpatialVBagg(kernel, likelihood: gpflow.likelihoods.Likelihood, zs=zs, z1=z1, z2=z2,num_data=train_bag.num_bags)
```

## License
License available at [License](LICENSE)
