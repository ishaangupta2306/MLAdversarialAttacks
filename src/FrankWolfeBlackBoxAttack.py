import time
from utils import *
import tensorflow as tf
import numpy as np


class FrankWolfeBlackBoxAttack:
    def __init__(self, session, cnn_model, number_of_iteration=10000, gradient_estimate_batch_size=25, order=np.inf, epsilon=0.05, clip_min=0,
                 clip_max=1, targeted_attack=True, step_size=0.01, delta=0.01, sensing_type='sphere', query_limit=50000, beta=0.99):

        self.session = session
        self.cnn_model = cnn_model
        self.order = order
        self.targeted_attack = targeted_attack
        self.box_attack_type = 'black'
        # Variables for the black box attack
        self.number_of_iteration = number_of_iteration
        self.gradient_estimate_batch_size = gradient_estimate_batch_size
        self.step_size = step_size
        self.beta = beta
        self.epsilon = epsilon
        self.delta = delta
        self.sensing_type = sensing_type
        self.query_limit = query_limit

        self.clip_min = clip_min
        self.clip_max = clip_max
        # Initialization for the shape
        self.shape = (None, cnn_model.image_size, cnn_model.image_size, cnn_model.num_channels)
        self.single_shape = (cnn_model.image_size, cnn_model.image_size, cnn_model.num_channels)
        # Initialization for the image
        self.image = tf.placeholder(tf.float32, self.shape)
        # Initialization for the label of the image
        self.labels = tf.placeholder(tf.float32, (None, cnn_model.num_labels))
        # Setup of fields for evaluation of image
        self.logits, self.predictions, self.softmax_cross_entropy_loss, self. eval_adv, self.reducedSingleTensorLoss, self.gradients = tensor_image_setup(self, tf)

        noise_pos = tf.random_normal((self.gradient_estimate_batch_size,) + self.single_shape)
        if self.sensing_type == 'sphere':
            noise = self.sample_vectors_euclidean_unit_sphere()
        elif self.sensing_type == 'gaussian':
            noise = tf.concat([noise_pos, -noise_pos], axis=0)
        else:
            raise Exception("Invalid Sensing type. Can't utilize natural bias as sensing vectors")

        self.gradient_estimate_images = self.image + self.delta * noise
        self.gradient_estimate_labels = tf.ones([self.gradient_estimate_batch_size * 2, 1]) * self.labels


    def sample_vectors_euclidean_unit_sphere(self):
        noise_pos = tf.random_normal((self.gradient_estimate_batch_size,) + self.single_shape)
        reduc_ind = list(range(1, len(self.shape)))
        noise_norm = tf.sqrt(tf.reduce_sum(tf.square(noise_pos), reduction_indices=reduc_ind, keep_dims=True))
        noise_pos = noise_pos / noise_norm
        d = np.prod(self.single_shape)
        noise_pos = noise_pos * (d ** 0.5)
        return tf.concat([noise_pos, -noise_pos], axis=0)

    def check_query_limit(self, stop_query, gradient_estimation_sample_size, i):
        stop_query[i] += gradient_estimation_sample_size * self.gradient_estimate_batch_size * 2
        if stop_query[i] > self.query_limit:
            stop_query[i] = self.query_limit
            return False

    def update_step_size(self, iteration, start_decay):
        if start_decay == 0:
            return self.step_size
        else:
            self.step_size / (iteration - start_decay + 1) ** 0.5

    def get_black_box_gradient_estimate(self, x, batch_lab, gradient_estimation_sample_size):
        gradients = []
        for _ in range(gradient_estimation_sample_size):
            gradient_estimate = self.session.run([self.gradient_estimate], {self.image: x, self.labels: batch_lab})
            gradients.append(gradient_estimate)
        gradients = np.array(gradients)
        return np.mean(gradients, axis=0, keepdims=True)

    def attack(self, inputs, targets, data_ori):
        adversary = np.copy(inputs)
        stop_query = np.zeros((len(inputs)))
        eval_adv, finished_mask, succ_sum, distortion = initial_setup(self, inputs, targets, data_ori)

        if succ_sum == len(inputs):
            return inputs, stop_query, finished_mask

        for i in range(len(inputs)):

            data = inputs[i:i + 1]
            lab = targets[i:i + 1]
            ori = data_ori[i:i + 1]
            x = data
            gradient_estimation_sample_size = 1
            momentum = np.zeros_like(data)

            last_loss = []
            hist_len = 5
            min_step_size = 1e-3
            current_step_size = self.step_size
            start_decay = 0

            # for t = 0, . . . , T − 1 do
            for iteration in range(self.number_of_iteration):
                # Check if crossed the query limit set by user
                if not self.check_query_limit(stop_query, gradient_estimation_sample_size, i):
                    break
                # Get zeroth-order gradient estimates qt = GRAD EST(xt , b, δ) // Alg 3
                gradient = get_black_box_gradient_estimate(x, lab, gradient_estimation_sample_size)  #Unlike white-box attack -> Can't peform back propagation
                # momentum: mt = β · mt−1 + (1 − β) · qt
                momentum = (self.beta * momentum) + (1 - self.beta) * gradient
                # Normalizing the momentum
                normalized_gradient = gradient_normalization(momentum, self.order)
                # Solution(vt = argminx∈X hx, mti) -> vt = −epsilon · sign(mt) + xori (Linear Minimization Oracle)
                v_t = -1 * self.epsilon * (-1 if not self.targeted else 1) * normalized_gradient + ori
                # dt = vt − xt
                d_t = v_t - x
                current_step_size = self.update_step_size(iteration, start_decay)
                # xt+1 = xt + γtdt
                new_x = x + current_step_size * d_t
                new_x = np.clip(new_x, self.clip_min, self.clip_max)
                x = new_x

                loss, pred, eval_adv = evaluate_image(x, lab)

                last_loss.append(loss)
                last_loss = last_loss[-hist_len:]
                if last_loss[-1] > 0.999 * last_loss[0] and len(last_loss) == hist_len:
                    if start_decay == 0:
                        start_decay = iteration - 1
                        print ("[log] start decaying lr")
                    last_loss = []

                finished_mask[i] = np.logical_not(eval_adv[0]) if not self.targeted else eval_adv[0]

                if iteration % 10 == 0:
                    distortion = get_distortion(x, ori, self.order)
                    print ("Iter: %3d, Loss: %5.3f, Dist: %5.3f, Lr: %5.4f, Finished: %3d, Query: %3d"
                        % (iteration, loss, distortion, current_step_size, succ_sum, stop_query[i]))

                if finished_mask[i]:
                    break

            adversary[i] = new_x

            distortion = get_distortion(x, ori, self.order)
            print ("End Loss : % 5.3f, Distortion: % 5.3f, Finished: % 3d,  Query: % 3d " % (
                loss, distortion, finished_mask[i], stop_query[i]))

        return adversary, stop_query, finished_mask
