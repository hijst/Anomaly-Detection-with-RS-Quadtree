import numpy
from pylab import *
import time


class StdOutListener():
    def __init__(self):
        self.start_time = time.time()
        self.x = []
        self.y = []
        self.my_average = []
        self.line_actual, = plot(self.x, self.y)  # line stores a Line2D we can update
        self.line_average, = plot(self.x, self.my_average)  # line stores a Line2D we can update

    def on_data(self, new_value):
        time_delta = time.time() - self.start_time  # on our x axis we store time since start
        self.x.append(time_delta)
        self.y.append(new_value)
        self.my_average.append(numpy.mean(self.y))
        self.line_actual.set_data(self.x, self.y)
        self.line_average.set_data(self.x, self.my_average)
        ylim([0, max(self.y)])  # update axes to fit the data
        xlim([0, max(self.x)])
        draw()  # redraw the plot


ion()  # ion() allows matplotlib to update animations.

out_listener = StdOutListener()
for i in range(1000):
    out_listener.on_data(i + numpy.random.randint(-5, 5))
