import matplotlib.pyplot as plt
# line 1 points
x1 = [17,12,30, 50, 35]
# plotting the line 1 points
plt.plot(x1, label = "accuracy in epoch x")

tickpos = []

for i in range(5):
    tickpos.append(i)

plt.xticks(tickpos,tickpos)

#plt.grid(axis='y')

plt.ylim((0,100))

# Set the x axis label of the current axis.
plt.xlabel('epoch')
# Set the y axis label of the current axis.
plt.ylabel('accuracy in %')
# Set a title of the current axes.
plt.title('accuracy on validation set (3484 images)')
# show a legend on the plot
plt.legend()
# Display a figure.
plt.show()
