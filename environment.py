# coding=utf-8
import requests
from pprint import pprint
import numpy as np

from config import Config

class Environment:
    """tcp/ipで接続
    trainer -> environment
    """
    def __init__(self, name=''):
        pass

    def reset(self):
        response = requests.get('http://{}:{}/reset'.format(Config.URL, Config.PORT))
        return np.array(response.json()['observation'])

    def step(self, action):
        response = requests.get('http://{}:{}/step/{}'.format(Config.URL, Config.PORT, action))
        data = response.json()
        return np.array(data['observation'])

    def sample(self):
        response = requests.get('http://{}:{}/sample'.format(Config.URL, Config.PORT))
        return response.json()['action']

if __name__ == '__main__':
    class Config:
        URL = '127.0.0.1'
        PORT = 5000

    env = Environment()
    env.reset()
    done = False
    while not done:
        env.step(env.sample())
