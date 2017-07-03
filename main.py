# coding=utf-8

import random
random.seed(0)

import nnabla as nn
import nnabla.logger as logger
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solver as S
from nnabla.contrib.context import extension_context
from nnabla.monitor import Monitor, MonitorSeries, MonitorTimeElapsed

from scipy.misc import imresize
from skimage.color import rgb2gray

import gym
import numpy as np
np.random.seed(0)

import requests

from config import Config
from environment import Environment

def dqn(image, test=False):
    image /= 255.0
    c1 = PF.convolution(image, 32, (8, 8), name='conv1')
    c1 = F.relu(F.max_pooling(c1, (4, 4)), inplace=True)
    c2 = PF.convolution(c1, 64, (4, 4), name='conv2')
    c2 = F.relu(F.max_pooling(c2, (2, 2)), inplace=True)
    c3 = PF.convolution(c2, 64, (3, 3), name='conv3')
    c4 = F.relu(PF.affine(c3, 512, name='fc4'), inplace=True)
    c5 = PF.affine(c4, 6, name='fc5')
    return c5

def preprocess_frame(observation):
    observation = imresize(observation, (Config.FRAME_WIDTH, Config.FRAME_HEIGHT))
    observation = rgb2gray(observation)
    observation = observation.astype(np.uint8)
    return observation

def train():
    if Config.USE_NW:
        env = Environment('Pong-v0')
    else:
        env = gym.make('Pong-v0')

    extension_module = Config.CONTEXT
    logger.info("Running in {}".format(extension_module))
    ctx = extension_context(extension_module, device_id=Config.DEVICE_ID)
    nn.set_default_context(ctx)

    monitor = Monitor(Config.MONITOR_PATH)
    monitor_loss = MonitorSeries("Training loss", monitor, interval=1)
    monitor_reward = MonitorSeries("Training reward", monitor, interval=1)
    monitor_q = MonitorSeries("Training q", monitor, interval=1)
    monitor_time = MonitorTimeElapsed("Training time", monitor, interval=1)

    # placeholder
    image = nn.Variable([Config.BATCH_SIZE, Config.STATE_LENGTH, Config.FRAME_WIDTH, Config.FRAME_HEIGHT])
    image_target = nn.Variable([Config.BATCH_SIZE, Config.STATE_LENGTH, Config.FRAME_WIDTH, Config.FRAME_HEIGHT])

    nn.clear_parameters()

    # create network
    with nn.parameter_scope("dqn"):
        q = dqn(image, test=False)
        q.prersistent = True # Not to clear at backward
    with nn.parameter_scope("target"):
        target_q = dqn(image_target, test=False)
        target_q.prersistent = True # Not to clear at backward

    # loss definition
    a = nn.Variable([Config.BATCH_SIZE, 1])
    q_val = F.sum(F.one_hot(a, (6,)) * q, axis=1, keepdims=True)
    t = nn.Variable([Config.BATCH_SIZE, 1])
    loss = F.mean(F.squared_error(t, q_val))

    if Config.RESUME:
        logger.info('load model: {}'.format(Config.RESUME))
        nn.load_parameters(Config.RESUME)

    # setup solver
    # update dqn parameter only
    solver = S.RMSprop(lr=Config.LEARNING_RATE, decay=Config.DECAY, eps=Config.EPSILON)
    with nn.parameter_scope("dqn"):
        solver.set_parameters(nn.get_parameters())

    # training
    epsilon = Config.INIT_EPSILON
    experiences = []
    step = 0
    for i in range(Config.EPISODE_LENGTH):
        logger.info("EPISODE {}".format(i))
        done = False
        observation = env.reset()
        for i in range(30):
            observation_next, reward, done, info = env.step(0)
            observation_next = preprocess_frame(observation_next)
        # join 4 frame
        state = [observation_next for _ in xrange(Config.STATE_LENGTH)]
        state = np.stack(state, axis=0)
        total_reward = 0
        while not done:
            # select action
            if step % Config.ACTION_INTERVAL == 0:
                if random.random() > epsilon or len(experiences) >= Config.REPLAY_MEMORY_SIZE:
                    # inference
                    image.d = state
                    q.forward()
                    action = np.argmax(q.d)
                else:
                    # random action
                    if Config.USE_NW:
                        action = env.sample()
                    else:
                        action = env.action_space.sample() # TODO refactor
                if epsilon > Config.MIN_EPSILON:
                    epsilon -= Config.EPSILON_REDUCTION_PER_STEP

            # get next environment
            observation_next, reward, done, info = env.step(action)
            observation_next = preprocess_frame(observation_next)
            total_reward += reward
            # TODO clip reward

            # update replay memory (FIFO)
            state_next = np.append(state[1:, :, :], observation_next[np.newaxis,:,:], axis=0)
            experiences.append((state_next, reward, action, state, done))
            if len(experiences) > Config.REPLAY_MEMORY_SIZE:
                experiences.pop(0)

            # update network
            if step % Config.NET_UPDATE_INTERVAL == 0 and len(experiences) > Config.INIT_REPLAY_SIZE:
                logger.info("update {}".format(step))
                batch = random.sample(experiences, Config.BATCH_SIZE)
                batch_observation_next = np.array([b[0] for b in batch])
                batch_reward = np.array([b[1] for b in batch])
                batch_action = np.array([b[2] for b in batch])
                batch_observation = np.array([b[3] for b in batch])
                batch_done = np.array([b[4] for b in batch], dtype=np.float32)

                batch_reward = batch_reward[:, np.newaxis]
                batch_action = batch_action[:, np.newaxis]
                batch_done = batch_done[:, np.newaxis]

                image.d = batch_observation.astype(np.float32)
                image_target.d = batch_observation_next.astype(np.float32)
                a.d = batch_action
                q_val.forward() # XXX
                target_q.forward()
                t.d = batch_reward + (1-batch_done) * Config.GAMMA * np.max(target_q.d, axis=1, keepdims=True)
                solver.zero_grad()
                loss.forward()
                loss.backward()

                monitor_loss.add(step, loss.d.copy())
                monitor_reward.add(step, total_reward)
                monitor_q.add(step, np.mean(q.d.copy()))
                monitor_time.add(step)
                # TODO weight clip
                solver.update()
                logger.info("update done {}".format(step))

            # update target network
            if step % Config.TARGET_NET_UPDATE_INTERVAL == 0:
                # copy parameter from dqn to target
                with nn.parameter_scope("dqn"):
                    src = nn.get_parameters()
                with nn.parameter_scope("target"):
                    dst = nn.get_parameters()
                for (s_key, s_val), (d_key, d_val) in zip(src.items(), dst.items()):
                    # Variable#d method is reference
                    d_val.d = s_val.d.copy()

            if step % Config.MODEL_SAVE_INTERVAL == 0:
                logger.info("save model")
                nn.save_parameters("model_{}.h5".format(step))

            step += 1
            observation = observation_next
            state = state_next

if __name__ == "__main__":
    train()
