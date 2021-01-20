#!/usr/bin/env python
# coding: utf-8

# In[18]:


# where is the image functions skiimage write it right now or go 
import functools
import imlib as im
import numpy as np
#import pylib as py
from pylib import argument , path
import tensorflow as tf
import tensorflow.keras as keras
#import tf2lib as tl
import check_point as checkpoint__
#import tf2gan as gan
import loss as loss__
import tqdm

import data
import module


# ==============================================================================
# =                                   param                                    =
# ==============================================================================

#argument.arg('--dataset', default='horse2zebra')
#argument.arg('--datasets_dir', default='datasets')
#argument.arg('--load_size', type=int, default=286)  # load image to this size
#argument.arg('--crop_size', type=int, default=256)  # then crop to this size
#argument.arg('--batch_size', type=int, default=1)
#argument.arg('--epochs', type=int, default=200)
#argument.arg('--epoch_decay', type=int, default=100)  # epoch to start decaying learning rate
#argument.arg('--lr', type=float, default=0.0002)
#argument.arg('--beta_1', type=float, default=0.5)
#argument.arg('--adversarial_loss_mode', default='lsgan', choices=['gan', 'hinge_v1', 'hinge_v2', 'lsgan', 'wgan'])
#argument.arg('--gradient_penalty_mode', default='none', choices=['none', 'dragan', 'wgan-gp'])
#argument.arg('--gradient_penalty_weight', type=float, default=10.0)
#argument.arg('--cycle_loss_weight', type=float, default=10.0)
#argument.arg('--identity_loss_weight', type=float, default=0.0)
#argument.arg('--pool_size', type=int, default=50)  # pool size to store fake samples
#args = argument.args()

dataset = 'horse2zebra'
datasets_dir = r"/content"
load_size = 286
crop_size = 256
batch_size = 1
epochs = 200
epoch_decay = 100 # epoch to start decaying learning rate
lr = 0.0002
beta_1 = 0.5
adversarial_loss_mode = 'lsgan'
gradient_penalty_mode = None
gradient_penalty_weight = 10.0
cycle_loss_weight = 10.0
identity_loss_weight = 0.0
pool_size = 50 # pool size to store fake samples





# output_dir
output_dir = path.join('output', dataset)
path.mkdir(output_dir)

# save settings
#argument.args_to_yaml(path.join(output_dir, 'settings.yml'), args)


# ==============================================================================
# =                                    data                                    =
# ==============================================================================

A_img_paths = path.glob(path.join(datasets_dir, dataset, 'trainA'), '*.jpg')
B_img_paths = path.glob(path.join(datasets_dir, dataset, 'trainB'), '*.jpg')
A_B_dataset, len_dataset = data.make_zip_dataset(A_img_paths, B_img_paths, batch_size, load_size, crop_size, training=True, repeat=False)

A2B_pool = data.ItemPool(pool_size)
B2A_pool = data.ItemPool(pool_size)

A_img_paths_test = path.glob(path.join(datasets_dir, dataset, 'testA'), '*.jpg')
B_img_paths_test = path.glob(path.join(datasets_dir, dataset, 'testB'), '*.jpg')
A_B_dataset_test, _ = data.make_zip_dataset(A_img_paths_test, B_img_paths_test, batch_size, load_size, crop_size, training=False, repeat=True)


# ==============================================================================
# =                                   models                                   =
# ==============================================================================

G_A2B = module.ResnetGenerator(input_shape=(crop_size, crop_size, 3))
G_B2A = module.ResnetGenerator(input_shape=(crop_size, crop_size, 3))

D_A = module.ConvDiscriminator(input_shape=(crop_size, crop_size, 3))
D_B = module.ConvDiscriminator(input_shape=(crop_size, crop_size, 3))

d_loss_fn, g_loss_fn = loss__.get_adversarial_losses_fn(adversarial_loss_mode)
cycle_loss_fn = tf.losses.MeanAbsoluteError()
identity_loss_fn = tf.losses.MeanAbsoluteError()

G_lr_scheduler = module.LinearDecay(lr, epochs * len_dataset, epoch_decay * len_dataset)
D_lr_scheduler = module.LinearDecay(lr, epochs * len_dataset, epoch_decay * len_dataset)
G_optimizer = keras.optimizers.Adam(learning_rate=G_lr_scheduler, beta_1 = beta_1)
D_optimizer = keras.optimizers.Adam(learning_rate=D_lr_scheduler, beta_1 = beta_1)


# ==============================================================================
# =                                 train step                                 =
# ==============================================================================

@tf.function
def train_G(A, B):
    with tf.GradientTape() as t:
        A2B = G_A2B(A, training=True)
        B2A = G_B2A(B, training=True)
        A2B2A = G_B2A(A2B, training=True)
        B2A2B = G_A2B(B2A, training=True)
        A2A = G_B2A(A, training=True)
        B2B = G_A2B(B, training=True)

        A2B_d_logits = D_B(A2B, training=True)
        B2A_d_logits = D_A(B2A, training=True)

        A2B_g_loss = g_loss_fn(A2B_d_logits)
        B2A_g_loss = g_loss_fn(B2A_d_logits)
        A2B2A_cycle_loss = cycle_loss_fn(A, A2B2A)
        B2A2B_cycle_loss = cycle_loss_fn(B, B2A2B)
        A2A_id_loss = identity_loss_fn(A, A2A)
        B2B_id_loss = identity_loss_fn(B, B2B)

        G_loss = (A2B_g_loss + B2A_g_loss) + (A2B2A_cycle_loss + B2A2B_cycle_loss) * cycle_loss_weight + (A2A_id_loss + B2B_id_loss) * identity_loss_weight

    G_grad = t.gradient(G_loss, G_A2B.trainable_variables + G_B2A.trainable_variables)
    G_optimizer.apply_gradients(zip(G_grad, G_A2B.trainable_variables + G_B2A.trainable_variables))

    return A2B, B2A, {'A2B_g_loss': A2B_g_loss,
                      'B2A_g_loss': B2A_g_loss,
                      'A2B2A_cycle_loss': A2B2A_cycle_loss,
                      'B2A2B_cycle_loss': B2A2B_cycle_loss,
                      'A2A_id_loss': A2A_id_loss,
                      'B2B_id_loss': B2B_id_loss}


@tf.function
def train_D(A, B, A2B, B2A):
    with tf.GradientTape() as t:
        A_d_logits = D_A(A, training=True)
        B2A_d_logits = D_A(B2A, training=True)
        B_d_logits = D_B(B, training=True)
        A2B_d_logits = D_B(A2B, training=True)

        A_d_loss, B2A_d_loss = d_loss_fn(A_d_logits, B2A_d_logits)
        B_d_loss, A2B_d_loss = d_loss_fn(B_d_logits, A2B_d_logits)
        D_A_gp = loss__.gradient_penalty(functools.partial(D_A, training=True), A, B2A, mode=gradient_penalty_mode)
        D_B_gp = loss__.gradient_penalty(functools.partial(D_B, training=True), B, A2B, mode=gradient_penalty_mode)

        D_loss = (A_d_loss + B2A_d_loss) + (B_d_loss + A2B_d_loss) + (D_A_gp + D_B_gp) * gradient_penalty_weight

    D_grad = t.gradient(D_loss, D_A.trainable_variables + D_B.trainable_variables)
    D_optimizer.apply_gradients(zip(D_grad, D_A.trainable_variables + D_B.trainable_variables))

    return {'A_d_loss': A_d_loss + B2A_d_loss,
            'B_d_loss': B_d_loss + A2B_d_loss,
            'D_A_gp': D_A_gp,
            'D_B_gp': D_B_gp}


def train_step(A, B):
    A2B, B2A, G_loss_dict = train_G(A, B)

    # cannot autograph `A2B_pool`
    A2B = A2B_pool(A2B)  # or A2B = A2B_pool(A2B.numpy()), but it is much slower
    B2A = B2A_pool(B2A)  # because of the communication between CPU and GPU

    D_loss_dict = train_D(A, B, A2B, B2A)

    return G_loss_dict, D_loss_dict


@tf.function
def sample(A, B):
    A2B = G_A2B(A, training=False)
    B2A = G_B2A(B, training=False)
    A2B2A = G_B2A(A2B, training=False)
    B2A2B = G_A2B(B2A, training=False)
    return A2B, B2A, A2B2A, B2A2B


# ==============================================================================
# =                                    run                                     =
# ==============================================================================

# epoch counter
ep_cnt = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)

# checkpoint
checkpoint = checkpoint__.Checkpoint(dict(G_A2B=G_A2B,
                                G_B2A=G_B2A,
                                D_A=D_A,
                                D_B=D_B,
                                G_optimizer=G_optimizer,
                                D_optimizer=D_optimizer,
                                ep_cnt=ep_cnt),
                           py.join(output_dir, 'checkpoints'),
                           max_to_keep=5)
try:  # restore checkpoint including the epoch counter
    checkpoint.restore().assert_existing_objects_matched()
except Exception as e:
    print(e)

# summary
train_summary_writer = tf.summary.create_file_writer(path.join(output_dir, 'summaries', 'train'))

# sample
test_iter = iter(A_B_dataset_test)
sample_dir = path.join(output_dir, 'samples_training')
path.mkdir(sample_dir)

# main loop
with train_summary_writer.as_default():
    for ep in tqdm.trange(epochs, desc='Epoch Loop'):
        if ep < ep_cnt:
            continue

        # update epoch counter
        ep_cnt.assign_add(1)

        # train for an epoch
        for A, B in tqdm.tqdm(A_B_dataset, desc='Inner Epoch Loop', total=len_dataset):
            G_loss_dict, D_loss_dict = train_step(A, B)

            # # summary
            checkpoint__.summary(G_loss_dict, step=G_optimizer.iterations, name='G_losses')
            checkpoint__.summary(D_loss_dict, step=G_optimizer.iterations, name='D_losses')
            checkpoint__.summary({'learning rate': G_lr_scheduler.current_learning_rate}, step=G_optimizer.iterations, name='learning rate')

            # sample
            if G_optimizer.iterations.numpy() % 100 == 0:
                A, B = next(test_iter)
                A2B, B2A, A2B2A, B2A2B = sample(A, B)
                img = im.immerge(np.concatenate([A, A2B, A2B2A, B, B2A, B2A2B], axis=0), n_rows=2)
                im.imwrite(img, path.join(sample_dir, 'iter-%09d.jpg' % G_optimizer.iterations.numpy()))

        # save checkpoint
        checkpoint.save(ep)


# In[ ]:




