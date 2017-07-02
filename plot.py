import csv

import matplotlib.pyplot as plt

reader = csv.reader(open("./monitor/Training-loss.series.txt", "r"), delimiter=" ")

loss = [row[1] for row in reader]

plt.plot(loss)

plt.savefig("loss.png")
