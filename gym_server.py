from flask import Flask, jsonify
import gym

env = gym.make('Pong-v0')

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route("/reset")
def reset():
    print('reset')
    observation = env.reset()
    env.render()
    ret = {
        'observation': observation.tolist(),
    }
    return jsonify(ret)

@app.route("/step/<int:action>")
def step(action):
    print('step')
    observation, reward, done, info = env.step(action)
    env.render()
    ret = {
        'observation': observation.tolist(),
        'reward': reward,
        'done': done,
        'info': info,
    }
    return jsonify(ret)

@app.route("/sample")
def sample():
    print('sample')
    action = env.action_space.sample()
    ret = {
        'action': action,
    }
    return jsonify(ret)

if __name__ == '__main__':
    # 127.0.0.1:5000
    app.run()
