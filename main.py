# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import argparse
import tensorflow as tf
import numpy as np

from src.FrankWolfeWhiteBoxAttack import FrankWolfeWhiteBoxAttack


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Frank Wolf Adversarial Attacks')
    parser.add_argument('--method', default="black", help='Adversarial attack method')
    parser.add_argument('--order', default="inf", help='Order of the Attack Algorithm')
    parser.add_argument('--arch', '-a', default='inception', help='target architecture')
    parser.add_argument('--sample', default=10000, type=int, help='number of samples to attack')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--step_size', default=0.01, type=float, help='Step_Size of algorithm')
    parser.add_argument('--epsilon', default=0.3, type=float, help='Epsilon for the algorithm')
    parser.add_argument('--att_iter', default=1000, type=int, help='max number of attack iterations')
    parser.add_argument('--targeted', default=1, type=int, help='targeted attack: 1 nontargeted attack: -1')
    parser.add_argument('--beta1', default=0.99, type=float, help='beta1 for FW')

    args = parser.parse_args()
    print(args)

    with tf.Session() as session:
        print('Loading model...')
        if args.arch == 'resnet':
            from setup_resnet import ImageNet, resnet_model

            data, model = ImageNet(), resnet_model(session)
            is_imagenet = 1
            print('ImageNet Resnet Model Loaded')
        elif args.arch == 'inception':
            from setup_inception_v3 import ImageNet, inception_model

            data, model = ImageNet(), inception_model(session)
            is_imagenet = 1
            print('ImageNet Inception Model Loaded')
        elif args.arch == 'cifar':
            from setup_cifar import CIFAR, CIFARModel

            data, model = CIFAR(), CIFARModel("models/cifar", session)
            is_imagenet = 0
            print('CIFAR10 Model Loaded')
        elif args.arch == 'mnist':
            from setup_mnist import MNIST, MNISTModel

            data, model = MNIST(), MNISTModel("models/mnist", session)
            is_imagenet = 0
            print('MNIST Model Loaded')
        else:
            raise Exception("Invalid or unknown architecture for the algorithm")

        # Initialization of White Box Attack
        is_targeted = True if args.targeted == 1 else False
        attack_order = 2 if args.order == "2" else np.inf
        WhiteBoxAttack = FrankWolfeWhiteBoxAttack(session, model, order=attack_order, targeted_attack=is_targeted, step_size=args.step_size,
                                                  number_of_iteration=args.number_of_iteration, gradient_estimate_batch_size=args.batch_size,
                                                  beta=args.beta, epsilon=args.epsilon)
        print('Initialization of White Box Attack complete')
        # generate data
        print('Generate data')
        inputs, targets = generate_data(data, samples=args.sample, targeted=True, is_imagenet=is_imagenet)
        print('Inputs Shape: ', inputs.shape)

        # start attacking
        success_count = 0
        stop_iterations = []
        total_iterations = 0
        adv = []
        stop_time = []
        finished = []

        total_batch = int(np.ceil(len(inputs) / args.batch))
        # timestart = time.time()
        for i in range(total_batch):
            start = i * args.batch_size
            end = min((i + 1) * args.batch_size, len(inputs))
            ind = range(start, end)

            adv_b, stop_iter_b, finished_b = WhiteBoxAttack.attack(inputs[ind], targets[ind], data_ori=inputs[ind])

            adv.extend(adv_b)
            # stop_time.extend(stop_time_b)
            stop_iterations.extend(stop_iter_b)
            finished.extend(finished_b)

            total_iterations += sum(stop_iter_b)
            success_count += sum(finished_b)

            print('batch: ', i + 1, ' avg iter: ', total_iterations / (i + 1), 'total succ: ', success_count)
            print('===========================')

        # timeend = time.time()
        # print("Took", timeend - timestart, "seconds to run", len(inputs), "samples.")

        adv = np.array(adv)
        # stop_time = np.array(stop_time)
        stop_iterations = np.array(stop_iterations)
        finished = np.array(finished)

        l2 = []
        linf = []
        for i in range(len(adv)):
            l2_sample = np.sum((adv[i] - inputs[i]) ** 2) ** .5
            linf_sample = np.max(np.abs(adv[i] - inputs[i]))
            if finished[i]:
                l2.append(l2_sample)
                linf.append(linf_sample)

        l2 = np.array(l2)
        linf = np.array(linf)
        print("======================================")
        print("Total L2 distortion: ", np.mean(l2))
        print("Total Linf distortion: ", np.mean(linf))
        print("Mean Time: ", np.mean(stop_time))
        print("Mean Iter: ", np.mean(stop_iterations))
        print("Succ Rate: ", np.mean(finished))

        summary_txt = 'L2 distortion: ' + str(np.mean(l2)) + ' Linf distortion: ' + str(
            np.mean(linf)) + ' Mean Time: ' + str(np.mean(stop_time)) + ' Total Time: ' + str(
            timeend - timestart) + ' Mean Iter: ' + str(np.mean(stop_iterations)) + ' Succ Rate: ' + str(np.mean(finished))
        with open(args.method + '_' + args.order + '_' + args.arch + '_whitebox' + '_summary' + '.txt', 'w') as f:
            json.dump(summary_txt, f)
