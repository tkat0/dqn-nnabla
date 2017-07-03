import csv

import matplotlib.pyplot as plt

def get_data(filepath):
    reader = csv.reader(open(filepath, "r"), delimiter=" ")
    data = [row[1] for row in reader]
    return data

data = get_data("./monitor/Training-loss.series.txt")
plt.subplot(2, 2, 1)
plt.plot(data)
plt.title("loss")

data = get_data("./monitor/Training-reward.series.txt")
plt.subplot(2, 2, 2)
plt.plot(data)
plt.title("reward")

data = get_data("./monitor/Training-q.series.txt")
plt.subplot(2, 2, 3)
plt.plot(data)
plt.title("q")

plt.savefig("train.png")
