from tensorflow.keras import Model
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import cv2
import os

from data import Dataset, dataset
from config import cfg

os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPU_IDS


def check_dirs(dirs):
    for dir in dirs:
        if not os.path.exists(dir):
            print("Make", dir)
            os.makedirs(dir)


if __name__ == '__main__':

    # 设置输出的地址
    save_adv_dir = cfg.SAVE_ADV_DIR
    adv_img_dir = os.path.join(save_adv_dir, "adv/")   # 对抗样本保存的地址
    perturb_jpg_dir = os.path.join(save_adv_dir, "perturb/jpg/")  # 对应扰动jpg保存的地址
    check_dirs([adv_img_dir, perturb_jpg_dir])

    test_data, test_num = dataset(istrain=False)
    test_steps_per_epoch = int(test_num / cfg.BATCH_SIZE)

    generator = tf.keras.models.load_model(cfg.GENERATOR_PATH, compile=False)
    gen_model = Model(inputs=generator.input, outputs=generator.layers[-1].output)

    model = tf.keras.models.load_model(cfg.MODEL_PATH, compile=False)
    tmodel = Model(inputs=model.input, outputs=model.layers[-1].output)

    suc_num = 0
    sum_num = 0
    p_dis = 0
    sum_prob = 0

    it = iter(test_data.take(test_steps_per_epoch + 1))
    next(it)
    pbar = tqdm(it)
    for images, labels, targets, paths in pbar:

        perturbs = gen_model(images)
        perturbs = tf.clip_by_value(perturbs, -0.5, 0.5)
        adv_images = images + perturbs
        adv_images = tf.clip_by_value(adv_images, -0.5, 0.5)

        adv_probs = tmodel(adv_images)

        for i, adv_prob in enumerate(adv_probs):
            sum_num += 1
            ind = tf.argmax(adv_prob)
            if ind == targets[i].numpy(): suc_num += 1

            if cfg.TARGETED:
                sum_prob += adv_prob.numpy()[np.int(targets[i])]
            elif not cfg.TARGETED and ind == targets[i]:
                sum_prob += (1. - adv_prob.numpy()[np.int(targets[i])]) / (cfg.NUM_CLASS - 1 + 1e-7)
            else:
                sum_prob += adv_prob.numpy()[ind]

            p_dis += np.sum((perturbs[i] ** 2) ** .5)

            path = str(paths[i]).split('\'')[1]
            name = path.split('/')[-1]   # XXX.jpg

            adv_img = (adv_images[i].numpy() + .5) * 255.
            adv_img = cv2.cvtColor(adv_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(adv_img_dir, name), adv_img)

            perturb = (perturbs[i] + .5) * 255.
            perturb = perturb[..., ::-1]  # rgb2bgr
            perturb = np.array(perturb, dtype=np.uint8)
            cv2.imwrite(os.path.join(perturb_jpg_dir, name), perturb)

        acc = suc_num / sum_num
        if not cfg.TARGETED: acc = 1.0 - acc
        dis = p_dis / sum_num
        ave_prob = sum_prob / sum_num
        pbar.set_description("acc: %.4f, ave_prob: %.6f, dis: %.6f" % (acc, ave_prob, dis))

