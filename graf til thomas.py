import matplotlib.pyplot as plt
# line 1 points
x1 = [2016,2017,2018]
y1 = [941351,980345,1031959]
# plotting the line 1 points
plt.plot(x1, y1, label = "SATS")


# line 2 points
x2 = [2016,2017,2018]
y2 = [223956,230703,235888]
# plotting the line 2 points
plt.plot(x2, y2, label = "Family Sports Club")

# line 3 points
x3 = [2016,2017,2018]
y3 = [270840,288042,316298]
# plotting the line 3 points
plt.plot(x3, y3, label = "Fresh Fitness")

# line 4 points
x4 = [2016,2017,2018]
y4 = [2652,3032,2842]
# plotting the line 4 points
plt.plot(x4, y4, label = "Spenst")


tickpos = [2016,2017,2018]

plt.xticks(tickpos,tickpos)

plt.grid(axis='y')

# Set the x axis label of the current axis.
plt.xlabel('Årstall')
# Set the y axis label of the current axis.
plt.ylabel('Omsetning (i hele tusen)')
# Set a title of the current axes.
plt.title('Oversikt over omsetning treningssentere 2016-2018')
# show a legend on the plot
plt.legend()
# Display a figure.
plt.show()
