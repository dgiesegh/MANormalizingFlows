import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp


class MADE(tf.layers.Layer):
    """
    A class representing the MADE architecture (Masked Autoencoders for Distribution Estimation, see arXiv:1502.03509).

    Inherits from tensorflow.layers.Layer and can be treated as such.
    """
    
    def __init__(self, in_shape, num_layers=1, num_nodes=128, activation="relu", random_nums=[], n_params=0, silent=False):
        """
        Parameters
        ----------
        in_shape: int
            number of inputs of the MADE network (excluding conditional parameters, see below)
        num_layers: int
            number of hidden layers (default 1)
        num_nodes: int
            number of nodes per hidden layer (default 128)
        activation: string
            type of activation function (supports 'tanh', 'sigmoid', 'linear' and 'relu', default 'relu')
        random_nums: array-like
            random numbers assigned to each hidden node to determine its connection to previous nodes (see original paper, default [])
            if [] random numbers are generated by the class, else shape must be (num_layers, num_nodes) and must be convertible to a np array
        n_params: int
            number of conditional parameters (i.e. additional inputs that every node may depend on, default 0)
        silent: boolean
            whether the network prints info messages outside of tensorflow (default False)
        """
        
        super().__init__()
        self.in_shape = in_shape
        if in_shape < 2:
            raise Exception("MADE Layer only supports at least two inputs.")
        self.num_layers = num_layers
        self.num_nodes = num_nodes
        self.masks = []
        self.kernels = []
        self.biases = []
        self.nums = np.array(random_nums)
        if self.nums.shape != (num_layers, num_nodes) and random_nums != []:
            print("Warning: MADE Layer number mismatch: random_nums must have shape (num_layers, num_nodes), i.e. ("+str(num_layers)+
                  ", "+str(num_nodes)+", but has shape "+str(self.nums.shape)+". Using internal rng instead.")
            self.nums = []
        self.activation = activation
        if activation not in ["relu", "tanh", "sigmoid", "linear"]:
            raise Exception("Activation function must be 'linear', 'relu', 'tanh' or 'sigmoid'.")
        self.n_params = n_params
        if n_params != 0 and not silent:
            print("Info: if you use conditional MADE layers, the conditional parameter(s) must be given as the first input(s) in "+
                  "the input vector.")
    
    def build(self, input_shape):
        if input_shape[-1] != self.in_shape+self.n_params:
            raise Exception("Input shape mismatch. This MADE layer was initialized to have "+str(self.in_shape)+
                            "+"+str(self.n_params)+" inputs, but got "+str(input_shape[-1])+" inputs.")
        rng = np.random.default_rng()
        if len(self.nums) == 0:
            if self.in_shape > 2:
                self.nums = rng.integers(low=1, high=self.in_shape, size=(self.num_layers,self.num_nodes))
            elif self.in_shape == 2:
                self.nums = np.ones((self.num_layers, self.num_nodes))
        # If there are params, add a zero at beginning and add n_params to nums
        x = 1 if self.n_params > 0 else 0
        self.nums = np.concatenate([np.zeros((self.num_layers, x)), self.nums], axis=1)
        self.nums += self.n_params
        
        # Input to hidden layer
        self.kernels.append(self.add_weight(
            shape=(self.in_shape+self.n_params, self.num_nodes+x), 
            initializer="glorot_uniform", 
            trainable=True, 
            name="input_kernel"
        ))
        self.biases.append(self.add_weight(
            shape=(self.num_nodes+x,),
            initializer="zeros",
            trainable=True,
            name="input_bias",
        ))
        self.masks.append(np.array(
            [[(1 if self.nums[0,k] >= d else 0) for d in range(1, self.in_shape+1+self.n_params)] 
             for k in range(0, self.num_nodes+x)], dtype=np.float32
        ).T)
        
        # hidden layer to hidden layer
        for l in range(1, self.num_layers):
            self.kernels.append(self.add_weight(
                shape=(self.num_nodes+x, self.num_nodes+x),
                initializer="glorot_uniform", 
                trainable=True, 
                name="hidden_kernel_"+str(l)
            ))
            self.biases.append(self.add_weight(
                shape=(self.num_nodes+x,),
                initializer="zeros",
                trainable=True,
                name="input_bias",
            ))
            self.masks.append(np.array(
                [[(1 if self.nums[l,k] >= self.nums[l-1,j] else 0) for j in range(0, self.num_nodes+x)] 
                 for k in range(0, self.num_nodes+x)], dtype=np.float32
            ).T)
        
        # Hidden layer to output layer
        self.kernels.append(self.add_weight(
            shape=(self.num_nodes+x, self.in_shape), 
            initializer="glorot_uniform", 
            trainable=True, 
            name="output_kernel"
        ))
        self.biases.append(self.add_weight(
            shape=(self.in_shape,),
            initializer="zeros",
            trainable=True,
            name="output_bias",
        ))
        self.masks.append(np.array(
            [[(1 if self.nums[self.num_layers-1,k] < d+self.n_params else 0) for k in range(0, self.num_nodes+x)] 
             for d in range(1, self.in_shape+1)],
            dtype=np.float32
        ).T)
    
    def call(self, x, training=True):
        #Use relu for hidden layer and activation parameter for output
        for i in range(self.num_layers):
            x = tf.maximum(tf.matmul(x, tf.multiply(self.masks[i], self.kernels[i])) + self.biases[i], 0)
        x = tf.matmul(x, tf.multiply(self.masks[self.num_layers], self.kernels[self.num_layers])) + self.biases[self.num_layers]
        if self.activation == "relu":
            x = tf.maximum(x, 0)
        elif self.activation == "tanh":
            x = tf.tanh(x)
        elif self.activation == "sigmoid":
            x = tf.sigmoid(x)
        return x


# Simple Model ------------------------

def MACoupling(in_shape, n_models=2, activations=[], num_layers=1, num_nodes=128):
    """
    Convenience function to generate a stack of MADE layers for a masked autoregressive normalizing flow.

    All MADE layers take the same inputs and produce different outputs.
    This is useful as most transformations used for Masked Autoregressive Normalizing Flows have more than one parameter.

    Parameters
    ----------
    in_shape: int
        number of inputs for the MADE layers
    n_models: int
        number of MADE layers (i.e. the number of outputs of the coupling layer is in_shape*n_models, default 2)
    activations: array-like
        activation functions used for the MADE layers (either [], then all layers use relu, or a list of activation functions for each MADE layer, 
        supports 'tanh', 'sigmoid', 'linear' and 'relu', default [])
    num_layers: int
        number of hidden layers of the MADE layers (default 1)
    num_nodes: int
        number of nodes in each hidden layer of the MADE layers (default 128)
    """
    
    if activations == []:
        activations = n_models * ["relu"]
    if len(activations) != n_models:
        raise Exception("Either give no activations [] or one for each parameter!")
    
    rng = np.random.default_rng()
    if in_shape > 2:
        random_nums = rng.integers(low=1, high=in_shape, size=(num_layers, num_nodes))
    else:
        random_nums = []
    
    _input = tf.layers.Input(shape=in_shape)
    
    models = []
    for i in range(n_models):
        m = MADE(in_shape, num_layers=num_layers, num_nodes=num_nodes, activation=activations[i], 
                 random_nums=random_nums, silent=True)(_input)
        models.append(m)
    
    return keras.Model(inputs=_input, outputs=models)

class MAFlowModel(keras.Model):
    """
    Model to represent a Masked Autoregressive Normalizing Flow, see arXiv:1705.07057.

    Inherits from tensorflow.keras.Model and can be treated as such.
    Given a data-space vector x and a latent-space vector z, model(x) transforms into the latent space and model.predict(z) transforms into the data space.
    Points in data-space can be sampled using model.sample(n_points).
    For conditional density estimation use MAConditionalFlowModel.
    """
    
    def __init__(self, n_coupling, in_shape, num_hidden_layers=1, num_nodes=128, permutations=None):
        """
        Parameters
        ----------
        n_coupling: int
            number of coupling layers
        in_shape: int
            number of inputs
        num_hidden_layers: int
            number of hidden layers of the used MADE layers (default 1)
        num_nodes:
            number of nodes in the hidden layers of the used MADE layers (default 128)
        permutations: array-like
            permutations used in between coupling layers, either None (then they are generated by the model) or with shape (n_coupling-1, in_shape), default None
        """
        
        super().__init__()
        self.n_coupling = n_coupling
        self.in_shape = in_shape
        # Latent distribution: multivariate standard gaussian
        loc = [0.0 for i in range(in_shape)]
        scale = [1.0 for i in range(in_shape)]
        self.distribution = tfp.distributions.MultivariateNormalDiag(loc=loc, scale_diag=scale)
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.logprob_tracker = keras.metrics.Mean(name="logprob")
        self.logdet_tracker = keras.metrics.Mean(name="logdet")
        self.layers_list = [MACoupling(in_shape, n_models=2, activations=["tanh", "linear"], num_layers=num_hidden_layers, 
                                       num_nodes=num_nodes) for i in range(n_coupling)]
        self.permutations = permutations
        if permutations == None:
            self.permutations = []
            rng = np.random.default_rng()
            for i in range(self.n_coupling-1):
                self.permutations.append(rng.permutation(in_shape))
    
    @property
    def metrics(self):
        return [self.loss_tracker, self.logprob_tracker, self.logdet_tracker]
    
    # training=True makes it so that model(x) gives latent space and model.predict(z) gives data space
    def call(self, x, training=True):
        direction = 1 if training else -1
        log_det = 0
        # Loop over coupling layers forwards if direction is 1, else backwards
        for i in range(self.n_coupling)[::direction]:
            if training:
                s, t = self.layers_list[i](x)
                x = x * tf.exp(s) + t
                log_det += tf.reduce_sum(s, axis=-1)
            else:
                for k in range(self.in_shape):
                    s, t = self.layers_list[i](x)
                    mask = np.array([(1 if j == k else 0) for j in range(self.in_shape)])
                    inv_mask = 1 - mask
                    x = x * inv_mask + (tf.exp(-s) * (x - t)) * mask
            # Permutations between coupling layers
            if (training and i != self.n_coupling-1) or (not training and i != 0):
                j = i if training else i-1
                # Perm on forward pass and inverse perm on backward pass
                perm = self.permutations[j] if training else np.argsort(self.permutations[j])
                x = tf.gather(x, perm, axis=-1)
        return x, log_det
    
    def sample(self, n_points):
        """
        Method to sample points from the learned distribution.

        Parameters
        ----------
        n_points: int
            number of points to sample
        """
        z = self.distribution.sample(n_points)
        x, _ = self.predict(z)
        return x
    
    def infer(self, x):
        """
        Method returning the estimated probability density for a data space sample x.

        Parameters
        ----------
        x: array-like
            the data space sample whose density should be inferred
        """
        z, log_det = self(x)
        log_density = self.distribution.log_prob(z) + log_det
        return tf.exp(log_density)
    
    def log_loss(self, x):
        z, log_det = self(x)
        log_likelihood = self.distribution.log_prob(z) + log_det
        return -tf.reduce_mean(log_likelihood), -tf.reduce_mean(self.distribution.log_prob(z)), -tf.reduce_mean(log_det)
    
    def train_step(self, data):
        with tf.GradientTape() as tape:
            loss, prob, det = self.log_loss(data)
        g = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(g, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        self.logprob_tracker.update_state(prob)
        self.logdet_tracker.update_state(det)
        return {"loss":self.loss_tracker.result(), "logprob":self.logprob_tracker.result(), 
                "logdet":self.logdet_tracker.result()}
    
    def test_step(self, data):
        loss, prob, det = self.log_loss(data)
        self.loss_tracker.update_state(loss)
        self.logprob_tracker.update_state(prob)
        self.logdet_tracker.update_state(det)
        return {"loss":self.loss_tracker.result(), "logprob":self.logprob_tracker.result(), 
                "logdet":self.logdet_tracker.result()}

    
# Conditional Model ---------------------


def MAConditionalCoupling(in_shape, n_params, n_models=2, activations=[], num_layers=1, num_nodes=128):
    """
    Convenience function to generate a stack of MADE layers for a conditional masked autoregressive normalizing flow.

    All MADE layers take the same inputs and produce different outputs.
    This is useful as most transformations used for Masked Autoregressive Normalizing Flows have more than one parameter.

    Parameters
    ----------
    in_shape: int
        number of inputs for the MADE layers (excluding conditional parameters)
    n_params: int
        number of conditional parameters
    n_models: int
        number of MADE layers (i.e. number of outputs of the coupling layer is in_shape*n_models, default 2)
    activations: array-like
        activation functions used for the MADE layers (either [], then all layers use relu, or a list of activation functions for each MADE layer, 
        supports 'tanh', 'sigmoid', 'linear' and 'relu', default [])
    num_layers: int
        number of hidden layers of the MADE layers (default 1)
    num_nodes: int
        number of nodes in each hidden layer of the MADE layers (default 128)
    """
    
    if activations == []:
        activations = n_models * ["relu"]
    if len(activations) != n_models:
        raise Exception("Either give no activations [] or one for each parameter!")
    
    rng = np.random.default_rng()
    if in_shape > 2:
        random_nums = rng.integers(low=1, high=in_shape, size=(num_layers, num_nodes))
    else:
        random_nums = []
    
    _input = tf.layers.Input(shape=in_shape+n_params)
    
    models = []
    for i in range(n_models):
        m = MADE(in_shape, num_layers=num_layers, num_nodes=num_nodes, activation=activations[i], random_nums=random_nums, 
                 n_params=n_params, silent=True)(_input)
        models.append(m)
    
    return keras.Model(inputs=_input, outputs=models)

class MAConditionalFlowModel(MAFlowModel):
    """
    Model to represent a Conditional Masked Autoregressive Normalizing Flow, see arXiv:1705.07057.

    Inherits from MAFlowModel and therefore from tensorflow.keras.Model and can be treated as such.
    Given a data-space vector x and a latent-space vector z, model(x) transforms into the latent space and model.predict(z) transforms into the data space.
    Points in data-space can be sampled using model.sample(n_points, ...), see doc.
    """
    
    def __init__(self, n_coupling, in_shape, param_hists, num_hidden_layers=1, num_nodes=128, permutations=None):
        """
        Parameters
        ----------
        n_coupling: int
            number of coupling layers
        in_shape: int
            number of inputs (excluding conditional parameters)
        param_hists: list
            list of histograms giving the distributions of conditional parameters
            each histogram must be of the form [values, bin_edges] (as produced by numpy.histogram) with len(values) = len(bin_edges)-1
            number of conditional parameters is inferred form len(param_hists)
        num_hidden_layers: int
            number of hidden layers of the used MADE layers (default 1)
        num_nodes:
            number of nodes in the hidden layers of the used MADE layers (default 128)
        permutations: array-like
            permutations used in between coupling layers, either None (then they are generated by the model) or with shape (n_coupling-1, in_shape), default None
        """
        
        super().__init__(n_coupling, in_shape, num_hidden_layers, num_nodes, permutations)
        self.param_hists = []
        # Check histogram shape and calculate cumulative distributions
        for hist in param_hists:
            if len(hist[0]) != len(hist[1]) - 1:
                raise Exception("Parameter histograms must be of the shape [values, bin_edges] where len(values) = len(bin_edges) - 1")
            # Calculate cumulative distr and add extra bins at front and end
            cdf = np.concatenate([np.zeros(1), np.cumsum(hist[0])])
            cdf = np.concatenate([cdf, np.array([cdf[-1]])])
            cdf = cdf / cdf[-1]
            bin_upper_edges = hist[1][1:]
            self.param_hists.append([cdf.astype(np.float32), bin_upper_edges.astype(np.float32)])
        self.n_params = len(param_hists)
        if self.n_params == 0:
            raise Exception("If you don't want conditional parameters, just use MAFlowModel.")
        self.layers_list = [MAConditionalCoupling(in_shape, self.n_params, activations=["tanh", "linear"], 
                                                  num_layers=num_hidden_layers, num_nodes=num_nodes) for i in range(n_coupling)]
        self.left_zero_matrix = tf.concat([tf.zeros((self.in_shape, self.n_params)), tf.eye(self.in_shape)], axis=1)
    
    def call(self, x, training=True):
        direction = 1 if training else -1
        log_det = 0
        # Loop over coupling layers forwards if direction is 1, else backwards
        for i in range(self.n_coupling)[::direction]:
            if training:
                s, t = self.layers_list[i](x)
                s = tf.matmul(s, self.left_zero_matrix)
                t = tf.matmul(t, self.left_zero_matrix)
                x = x * tf.exp(s) + t
                log_det += tf.reduce_sum(s, axis=-1)
            else:
                for k in range(self.in_shape):
                    s, t = self.layers_list[i](x)
                    s = tf.matmul(s, self.left_zero_matrix)
                    t = tf.matmul(t, self.left_zero_matrix)
                    mask = np.array([(1 if j == k+self.n_params else 0) for j in range(self.n_params+self.in_shape)])
                    inv_mask = 1 - mask
                    mask = tf.constant(mask, dtype=tf.float32)
                    inv_mask = tf.constant(inv_mask, dtype=tf.float32)
                    x = x * inv_mask + (tf.exp(-s) * (x - t)) * mask
            # Permutations between coupling layers
            if (training and i != self.n_coupling-1) or (not training and i != 0):
                j = i if training else i-1
                # Perm on forward pass and inverse perm on backward pass
                perm = self.permutations[j] if training else np.argsort(self.permutations[j])
                # Don't change params
                perm = np.concatenate([np.arange(self.n_params), perm+self.n_params])
                x = tf.gather(x, perm, axis=-1)
        return x, log_det
    
    def sample(self, n_points, params=None, seed=None):
        """
        Method to sample points from the learned distribution.

        Parameters
        ----------
        n_points: int
            number of points to sample
        params: array-like or None
            if None, values for conditional parameters are sampled from provided histograms, else must be of shape (n_points, n_params), default None
        """
        if params is not None:
            if params.shape[0] != n_points or params.shape[1] != self.n_params:
                raise Exception("params argument of MAConditionalFlowModel.sample function must have shape (n_points, n_params).")
        else:
            params = self.ITS(n_points)
        z = self.distribution.sample(n_points, seed=seed)
        z = tf.concat([params, z], axis=1)
        x, _ = self.predict(z)
        return x
    
    def param_density(self, x):
        params = x[:,:self.n_params]
        densities = tf.ones_like(x[:,0])
        for i in range(self.n_params):
            m = params[:,i]
            #Sort m vals into bins of histogram
            m_ind = tf.searchsorted(self.param_hists[i][1], m) + 1
            #Calculate densities from cdf
            densities *= (tf.gather(self.param_hists[i][0], m_ind) - tf.gather(self.param_hists[i][0], tf.maximum(m_ind-1,0)))
        return densities
    
    #Inverse transform sampling from histograms
    def ITS(self, n_points):
        samples = [np.array([]).reshape(n_points,-1)]
        for hist in self.param_hists:
            vals = np.random.rand(n_points)
            val_bins = np.searchsorted(hist[0], vals) - 1
            points = hist[1][val_bins]
            samples.append(points.reshape(-1,1))
        return tf.constant(np.concatenate(samples, axis=1), dtype=tf.float32)
    
    def infer(self, x):
        """
        Method returning the estimated probability density for a data space sample x.

        Parameters
        ----------
        x: array-like
            the data space sample whose density should be inferred
        """
        z, log_det = self(x)
        log_density = self.distribution.log_prob(z[:,self.n_params:]) + log_det
        return tf.exp(log_density)
    
    def log_loss(self, x):
        z, log_det = self(x)
        log_likelihood = self.distribution.log_prob(z[:,self.n_params:]) + log_det
        return (-tf.reduce_mean(log_likelihood), -tf.reduce_mean(self.distribution.log_prob(z[:,self.n_params:])), 
                -tf.reduce_mean(log_det))


