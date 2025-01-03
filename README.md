# MANormalizingFlows

This repository contains a script implementing the MADE architecture (Masked Autoencoders for Distribution Estimation, see arXiv:1502.03509) as a `tf.layers.Layer`, as well as Masked Autoregressive Normalizing Flows (see arXiv:1705.07057) for conditional and non-conditional density estimation as a `tf.keras.Model`.

For example usage, see the Example jupyter notebook.

## Usage

### Non-conditional density estimation

For the estimation of a non-conditional probability density, the class `MANormalizingFlows.MAFlowModel` is used. It inherits from `tf.keras.Model` and can thus be compiled and fitted as a usual keras model. The parameters are as follows:
```python
model = MANormalizingFlows.MAFlowModel(n_coupling, in_shape, num_hidden_layers=1, num_nodes=128, permutations=None)
```
- `n_coupling: int` The number of coupling layers. Each coupling layer consists of one autoregressive transformation modeled by a MADE network.
- `in_shape: int` The number of dimensions of the input dataset. Minimum 2.
- `num_hidden_layers: int` The number of hidden layers of the MADE networks. Default 1.
- `num_nodes: int` The number of nodes of the hidden layers of the MADE networks. Default 128.
- `permutations: array-like or None` Permutations used in between coupling layers. Either `None` (then they are randomly generated) or of shape `(n_coupling-1, in_shape)`. Default `None`.

The inputs for training/fitting should be of shape `(n_points, in_shape)` where `n_points` is the number of data points.

Given a point `x` in data space, `model(x)` transforms it into the latent space. Using `model.infer(x)` the learned probability density at `x` can be extracted. Given a point `z` in latent space, `model.predict(z)` goes back to data space.

The model can be used to draw samples from the learned distribution using `model.sample(n_points)` where `n_points` is the number of points to sample.

### Conditional density estimation

For the estimation of a conditional probability density, the class `MANormalizingFlows.MAConditionalFlowModel` is used. It inherits from `tf.keras.Model` and can thus be compiled and fitted as a usual keras model. The parameters are as follows:
```python
model = MANormalizingFlows.MAConditionalFlowModel(n_coupling, in_shape, param_hists, num_hidden_layers=1, num_nodes=128, permutations=None)
```
- `n_coupling: int` The number of coupling layers. Each coupling layer consists of one autoregressive transformation modeled by a MADE network.
- `in_shape: int` The number of dimensions of the input dataset excluding conditional parameters. Minimum 2.
- `param_hists: list` List of histograms of the distributions of each conditional parameter. Each histogram must be of the form `[values, bin_edges]` (as produced by `np.histogram`) with `len(values) = len(bin_edges) - 1`. The number of conditional parameters is automatically inferred from `len(param_hists)`.
- `num_hidden_layers: int` The number of hidden layers of the MADE networks. Default 1.
- `num_nodes: int` The number of nodes of the hidden layers of the MADE networks. Default 128.
- `permutations: array-like or None` Permutations used in between coupling layers. Either `None` (then they are randomly generated) or of shape `(n_coupling-1, in_shape)`. Default `None`.

The inputs for training/fitting should be of shape `(n_points, n_params+in_shape)` where `n_points` is the number of data points and `n_params` the number of conditional parameters. The model assumes the conditional parameters are the first `n_params` dimensions of the dataset in the order their histograms were supplied.

Given a point `x` in data space, `model(x)` transforms it into the latent space. Using `model.infer(x)` the learned probability density at `x` can be extracted. Given a point `z` in latent space, `model.predict(z)` goes back to data space. The conditional parameters must be supplied to these functions even though they are obviously not transformed.

The model can be used to draw samples from the learned distribution using `model.sample(n_points, params=None)` where `n_points` is the number of points to sample and `params` either an array-like of shape `(n_points, n_params)` giving the values of the conditional parameters for the sample or `None`, in which case the parameter values are randomly sampled from the supplied histograms.