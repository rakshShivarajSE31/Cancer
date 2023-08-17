from matplotlib import pyplot as plt
from matplotlib import style

style.use('ggplot')

x1=[2,4,6]
y1=[1,5,6]
x2=[2,4,6]
y2=[1,5,6]
plt.plot(x1,y1,'black')
plt.plot(x2,y2,'red')
plt.xlabel('x_axis')
plt.xlabel('y_axis')
plt.title('info')
plt.show()
