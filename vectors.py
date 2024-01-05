import matplotlib.pyplot as plt

def plot_vector(x, y):
    plt.figure()
    plt.quiver(0, 0, x, y, angles='xy', scale_units='xy', scale=1)
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.grid(True)
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.title('Vector Visualization')
    plt.show()

# Example usage
x_component = -1
y_component = 1

plot_vector(x_component, y_component)
