import sys
import tensorflow as tf
import numpy as np
import time
from utils import *


class FrankWolfeWhiteBoxAttack:
    def __init__(self, session, cnn_model, order, targeted_attack, step_size,
                 number_of_iteration=20, gradient_estimate_batch_size=50, beta=0.99, epsilon=0.3
                  ): #delta=0.01, sensing_type='sphere', query_limit=50000,

        # Initialization for Attack algorithm
        self.session    = session
        self.cnn_model = cnn_model
        self.order = order
        self.targeted_attack = targeted_attack
        self.box_attack_type = 'white'
        # Variables for the white box attack
        self.number_of_iteration = number_of_iteration
        self.gradient_estimate_batch_size = gradient_estimate_batch_size
        self.step_size = step_size
        self.beta = beta
        self.epsilon = epsilon

        # Initialization for the image
        self.image = tf.placeholder(tf.float32, (None,  cnn_model.image_size,  cnn_model.image_size, cnn_model.num_channels))
        # Initialization for the label of the image
        self.labels = tf.placeholder(tf.float32, (None, cnn_model.num_labels))
        # Setup of fields for evaluation of image
        # Vector of raw predictions made by model
        self.model_logits, self.predictions = self.cnn_model.predict(self.image)
        # Calculation of loss using softmax function to normalize probabilities with one value for each class
        self.softmax_cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.model_logits, labels=self.labels)
        # Comparison between the predictions and labels of the image
        self.eval_adv = tf.equal(self.predictions, tf.argmax(self.labels, 1))
        # Dimensions of tensors reduced to calculate a tensor with single loss
        self.reduced_single_tensor_loss = tf.reduce_sum(self.softmax_cross_entropy_loss)
        # Calculation of gradient of softmax_cross_entropy_loss wrt image
        self.gradients, = tf.gradients(self.softmax_cross_entropy_loss, self.image)

    def evaluate_image(self, inputs, targets):
        if self.box_attack_type == 'white':
            return self.session.run([self.reduced_single_tensor_loss, self.softmax_cross_entropy_loss, self.eval_adv],
                                    {self.image: inputs, self.labels: targets})

    def get_white_box_gradient_estimate(self, inputs, targets):
        if self.box_attack_type == 'white':
            return self.session.run(self.gradients, {self.image: inputs, self.labels: targets})

    def initial_attack_setup(self, inputs, targets, data_ori):
        loss_init, _, eval_adv = self.evaluate_image(inputs, targets)
        finished_mask = self.set_finished_mask(self, eval_adv)
        succ_sum = sum(finished_mask)

        distortion = get_distortion(inputs, data_ori, self.order)
        print("Init Loss : % 5.3f, Dist: % 5.3f, Finished: % 3d " % (
            loss_init, distortion, succ_sum))
        return eval_adv, finished_mask, succ_sum, distortion

    def set_finished_mask(self, eval_adv):
        if self.targeted_attack:
            return eval_adv
        else:
            return np.logical_not(eval_adv)



    def attack(self, inputs, targets, data_ori):
        x = np.copy(inputs)
        stop_iteration = np.zeros((len(inputs)))
        momentum = np.zeros_like(inputs)
        eval_adv, finished_mask, succ_sum, distortion = self.initial_attack_setup(self, inputs, targets, data_ori)
        # Step Size
        minimum_step_size = 1e-3
        current_step_size = self.step_size

        last_ls = []
        hist_len = 2
        
        # for t = 0, . . . , T − 1 do
        for iteration in range(self.number_of_iteration):
            # ∇f(xt)
            gradient = self.get_white_box_gradient_estimate(x, targets)
            # mt = β · mt−1 + (1 − β) · ∇f(xt)
            momentum = (self.beta * momentum) + (1 - self.beta) * gradient
            # Normalizing the momentum
            normalized_gradient = gradient_normalization(momentum, self.order)
            # Solution(vt = argminx∈X hx, mti) -> vt = −epsilon · sign(mt) + xori (Linear Minimization Oracle)
            value = -1 * self.epsilon * normalized_gradient + data_ori
            # dt = vt − xt
            d_value = value - x
            # xt+1 = xt + γtdt
            temp = x + (-1 if not self.targeted_attack else 1) * current_step_size * d_value
            temp2 = normal_ball_projection_inner(temp - data_ori, self.order, self.epsilon)
            x_updated = np.clip(data_ori + temp2, 0, 1)

            mask = finished_mask.reshape(-1, *[1] * 3)
            x = x_updated * (1. - mask) + x * mask
            # stop_time += (time.time() - start_time) * (1. - finished_mask)
            stop_iteration += 1 * (1. - finished_mask)

            loss, _, eval_adv = self.evaluate_image(x, targets)
            tmp = self.set_finished_mask(self, eval_adv)
            finished_mask = np.logical_or(finished_mask, tmp)
            succ_sum = sum(finished_mask)

            if iteration % 1 == 0:
                distortion = get_distortion(x, data_ori, self.order)
                print ("Iter : % 3d, Loss : % 5.3f, Dist: % 5.3f, lr: % 5.3f, Finished: % 3d " % (
                    iteration, loss, distortion, current_step_size, succ_sum))

            if succ_sum == len(inputs):
                break
        return x, stop_iteration, finished_mask
