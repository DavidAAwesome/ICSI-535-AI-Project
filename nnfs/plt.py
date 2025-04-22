import matplotlib as plt
import random
import numpy as np
import matplotlib.pyplot as plt

# Generate x values
x = np.linspace(1, 365, 365-int(365*(3/7)))  # Avoid log(0) by starting from 1

# Compute the logarithmic function
y = 5*np.log(x)

# Add noise
noise = np.random.normal(0, 0.1, size=y.shape)  # Mean = 0, Std = 0.1
y_noisy = y + noise * 5

# Create the plot
plt.figure(figsize=(10, 6))
# plt.plot(x, y, label='Logarithmic Function', color='blue')
plt.plot(x, y_noisy, label='weight lifted', color='red')
plt.title('Weight progress over a year')
plt.xlabel('time(days)')
plt.ylabel('weight increase(lbs)')
plt.legend()
plt.grid()
plt.show()
