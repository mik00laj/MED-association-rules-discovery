import matplotlib.pyplot as plt

# Data
lift =      [0.69, 0.77, 0.80, 0.89,  0.91, 0.93, 0.97, 0.99, 1.07, 1.08, 1.12, 1.19]
certainty = [1.44, 1.63, 1.68, 1.86,  1.92, 2.48, 2.70, 2.78, 3.00, 3.02, 3.16, 3.33]
jaccard =    [0.87, 1.10, 1.18, 1.50, 1.62, 1.88, 2.46, 2.71, 3.75, 3.86, 5.49, 7.13]
odds_ratio = [2.06, 2.32, 2.40, 2.66, 2.74, 3.54, 3.86, 3.97, 4.28, 4.31, 4.53, 4.76]

# Create individual plots for each metric

# Certainty vs Lift
plt.figure(figsize=(8, 6))
plt.plot(lift, certainty, label='Certainty (Lift)', marker='o')
plt.xlabel('Lift')
plt.ylabel('Certainty')
plt.title('Certainty as a Function of Lift')
plt.grid(True)
plt.legend()
plt.show()

# Jaccard vs Lift
plt.figure(figsize=(8, 6))
plt.plot(lift, jaccard, label='Jaccard (Lift)', marker='o')
plt.xlabel('Lift')
plt.ylabel('Jaccard')
plt.title('Jaccard as a Function of Lift')
plt.grid(True)
plt.legend()
plt.show()

# Odds Ratio vs Lift
plt.figure(figsize=(8, 6))
plt.plot(lift, odds_ratio, label='Odds Ratio (Lift)', marker='o')
plt.xlabel('Lift')
plt.ylabel('Odds Ratio')
plt.title('Odds Ratio as a Function of Lift')
plt.grid(True)
plt.legend()
plt.show()