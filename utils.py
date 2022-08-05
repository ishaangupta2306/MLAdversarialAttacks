import numpy as np
from six.moves import xrange


# Epsilon Search Parameter
def eps_search(ep, order):
    if (ep == 0.):
        if order == np.inf:
            step_size = 1e-3
            eps = np.arange(1e-3, 1e+0, step_size)
        elif order == 2:
            step_size = 1e-2
            eps = np.arange(1e-2, 1e+1, step_size)
        elif order == 1:
            step_size = 1e+0
            eps = np.arange(1e+0, 1e+3, step_size)
    else:
        eps = [ep]
    return eps


# Norm Ball Projection
def normal_ball_projection_inner(eta, order, eps):
    if order == np.inf:
        eta = np.clip(eta, -eps, eps)
    elif order in [1, 2]:
        reduc_ind = list(xrange(1, len(eta.shape)))
        if order == 1:
            norm = np.sum(np.abs(eta), axis=tuple(reduc_ind), keepdims=True)
        elif order == 2:
            norm = np.sqrt(np.sum(np.square(eta), axis=tuple(reduc_ind), keepdims=True))

        if norm > eps:
            eta = np.multiply(eta, np.divide(eps, norm))
    return eta


# Gradient Normalization
def gradient_normalization(gradients, order):
    if order == np.inf:
        signed_grad = np.sign(gradients)
    elif order in [1, 2]:
        reduc_ind = list(xrange(1, len(gradients.shape)))
        if order == 1:
            norm = np.sum(np.abs(gradients), axis=tuple(reduc_ind), keepdims=True)
        elif order == 2:
            norm = np.sqrt(np.sum(np.square(gradients), axis=tuple(reduc_ind), keepdims=True))
        signed_grad = gradients / norm
    return signed_grad


# Get Distortion
def get_distortion(a, b, order):
    if order == np.inf:
        dist = np.max(np.abs(a - b))
    elif order == 1:
        dist = np.sum(np.abs(a - b))
    elif order == 2:
        dist = np.sum((a - b) ** 2) ** .5
    return dist


def tensor_image_setup(self, tf):
    # Vector of raw predictions made by model
    logits, predictions = self.cnn_model.predict(self.image)
    # Calculation of loss using softmax function to normalize probabilities with one value for each class
    softmax_cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.model_logits, labels=self.labels)
    # Comparison between the predictions and labels of the image
    eval_adv = tf.equal(self.predictions, tf.argmax(self.labels, 1))
    # Dimensions of tensors reduced to calculate a tensor with single loss
    reducedSingleTensorLoss = tf.reduce_sum(self.softmax_cross_entropy_loss)
    # Calculation of gradient of softmax_cross_entropy_loss wrt image
    gradients, = tf.gradients(self.softmax_cross_entropy_loss, self.image)
    return logits, predictions, softmax_cross_entropy_loss, eval_adv, reducedSingleTensorLoss, gradients


# def evaluate_image(self, inputs, targets):
#        return self.session.run([self.reduced_single_tensor_loss, self.predictions, self.eval_adv], {self.image: inputs, self.labels: targets})



def get_black_box_gradient_estimate(self, x, batch_lab, gradient_estimation_sample_size):
    gradients = []
    for _ in range(gradient_estimation_sample_size):
        gradient_estimate = self.session.run([self.gradient_estimate], {self.image: x, self.labels: batch_lab})
        gradients.append(gradient_estimate)
    gradients = np.array(gradients)
    return np.mean(gradients, axis=0, keepdims=True)


