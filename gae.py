import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm

from config import cfg
from data import Dataset, dataset
from models import Generator, Discriminator, target_model

os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPU_IDS


def check_dir(dir):
    if not os.path.exists(dir):
        print("Make", dir)
        os.makedirs(dir)

def adv_loss(preds, labels, is_targeted):
    confidence = cfg.CONFIDENCE
    real = tf.reduce_sum(labels * preds, 1)
    other = tf.reduce_max((1 - labels) * preds - (labels * 10000), 1)
    if is_targeted:
        return tf.reduce_sum(tf.maximum(0.0, other - real + confidence))
    return tf.reduce_sum(tf.maximum(0.0, real - other + confidence))

def perturb_loss(preds, thresh=0.3):
    zeros = tf.zeros((tf.shape(preds)[0]))
    return tf.reduce_mean(tf.maximum(zeros, tf.norm(tf.reshape(preds, (tf.shape(preds)[0], -1)), axis=1) - thresh))


class GAE:
    def __init__(self, train_ds, test_ds, tmodel):
        self.thresh = cfg.THRESH
        self.smooth = cfg.SMOOTH
        self.is_targeted = cfg.TARGETED

        self.tmodel = tmodel

        self.train_data, self.train_steps_per_epoch = train_ds
        self.test_data, self.test_steps_per_epoch = test_ds

        self.discriminator = Discriminator()
        self.generator = Generator()

        self.d_opt = tf.keras.optimizers.Adam(4e-4, beta_1=0.5)
        self.g_opt = tf.keras.optimizers.Adam(4e-4, beta_1=0.5)

        self.loss_object = tf.keras.losses.BinaryCrossentropy()

    def train_step(self, images, targets_tensor_onehot):
        with tf.GradientTape() as disc_tape, tf.GradientTape(persistent=True) as gen_tape:
            disc_tape.watch(self.discriminator.trainable_variables)
            gen_tape.watch(self.generator.trainable_variables)

            perturb = tf.clip_by_value(self.generator(images, training=True), -self.thresh, self.thresh)
            images_perturbed = perturb + images
            images_perturbed = tf.clip_by_value(images_perturbed, -0.5, 0.5)

            d_real_logits = self.discriminator(images, training=True)
            d_real_probs = tf.nn.sigmoid(d_real_logits)

            d_fake_logits = self.discriminator(images_perturbed, training=True)
            d_fake_probs = tf.nn.sigmoid(d_fake_logits)

            f_fake_logits, f_fake_probs = self.tmodel.predict_logits(images_perturbed)

            d_labels_real = tf.ones_like(d_real_probs) * (1 - self.smooth)
            d_labels_fake = tf.zeros_like(d_fake_probs)

            d_loss_real = self.loss_object(d_labels_real, d_real_probs)
            d_loss_fake = self.loss_object(d_labels_fake, d_fake_probs)

            d_loss = d_loss_real + d_loss_fake

            g_loss_fake = self.loss_object(tf.ones_like(d_fake_probs), d_fake_probs)

            l_perturb = perturb_loss(perturb, self.thresh)

            l_adv = adv_loss(f_fake_logits, targets_tensor_onehot, self.is_targeted)

            alpha = cfg.ALPHA
            beta = cfg.BETA
            g_loss = l_adv + alpha * g_loss_fake + beta * l_perturb

        for _ in range(1):
            disc_grad = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.d_opt.apply_gradients(zip(disc_grad, self.discriminator.trainable_variables))

        for _ in range(4):
            gen_grad = gen_tape.gradient(g_loss, self.generator.trainable_variables)
            self.g_opt.apply_gradients(zip(gen_grad, self.generator.trainable_variables))

        return g_loss, d_loss

    def fit(self, epoch):

        train_it = iter(self.train_data.take(self.train_steps_per_epoch + 1))
        next(train_it)
        pbar = tqdm(train_it)
        for images, labels, targets, paths in pbar:
            targets_tensor = tf.convert_to_tensor(targets, dtype=tf.int32)
            targets_tensor_onehot = tf.one_hot(targets_tensor, depth=cfg.NUM_CLASS)

            g_loss, d_loss = self.train_step(images, targets_tensor_onehot)

            pbar.set_description("epoch: %d, g_loss: %.8f, d_loss: %.8f" % (epoch, g_loss.numpy(), d_loss.numpy()))

        if epoch % 1 == 0:
            self.discriminator.save(cfg.DISC_SAVE_DIR+str(epoch)+".h5", save_format='h5')
            self.generator.save(cfg.GEN_SAVE_DIR+str(epoch)+".h5", save_format='h5')

    def test(self):
        suc_num = 0
        sum_num = 0
        p_dis = 0
        sum_prob = 0

        test_it = iter(self.test_data.take(self.test_steps_per_epoch + 1))
        next(test_it)
        for images, labels, targets, paths in test_it:

            perturbs = tf.clip_by_value(self.generator(images, training=False), -self.thresh, self.thresh)
            images_perturbed = perturbs + images
            images_perturbed = tf.clip_by_value(images_perturbed, -0.5, 0.5)

            probs = self.tmodel.predict_softmax(images_perturbed)

            for i, prob in enumerate(probs):
                sum_num += 1

                ind = tf.argmax(prob)
                if ind == targets[i].numpy(): suc_num += 1
                if cfg.TARGETED:
                    sum_prob += prob.numpy()[np.int(targets[i])]
                elif not cfg.TARGETED and ind == targets[i]:
                    sum_prob += (1. - prob.numpy()[np.int(targets[i])]) / (cfg.NUM_CLASS - 1 + 1e-7)
                else:
                    sum_prob += prob.numpy()[ind]

                p_dis += np.sum((perturbs[i] ** 2) ** .5)

        acc = suc_num / sum_num
        if not cfg.TARGETED:
            acc = 1.0 - acc
        dis = p_dis / sum_num
        ave_prob = sum_prob / sum_num
        print("Attack Success Rate:", acc, "Average Confidence:", ave_prob, "Perturb Distance:", dis)

if __name__ == '__main__':

    train_data, train_num = dataset(istrain=True)
    test_data, test_num = dataset(istrain=False)

    train_steps_per_epoch = int(train_num / cfg.BATCH_SIZE)
    test_steps_per_epoch = int(test_num / cfg.BATCH_SIZE)

    tmodel = target_model()

    check_dir(cfg.GEN_SAVE_DIR)
    check_dir(cfg.DISC_SAVE_DIR)

    GAN = GAE([train_data, train_steps_per_epoch], [test_data, test_steps_per_epoch], tmodel)
    for epoch in range(cfg.EPOCHS):
        print("---------------------------------------", epoch, "---------------------------------------")
        GAN.fit(epoch)
        GAN.test()

