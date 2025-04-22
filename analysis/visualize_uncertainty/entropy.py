import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(0, 1, 1000)
y_e = -x * np.log(x) 
#y_b = -x * np.log(x) - (1-x) * np.log(1-x)

y_e_max = 1/np.e
x_e_max = 1/np.e
#y_b_max = np.max(y_b)

plt.figure(figsize=(5, 4))
plt.plot(x, y_e, label='Entropy', color="#d95f02", linewidth=2)
#plt.plot(x, y_b)
plt.axvline(x=0.5, color='#b2df8a', linestyle='--', alpha=1)
plt.axvline(x=x_e_max, color='#7570b3', linestyle='--', alpha=0.9)

plt.axhline(y=y_e_max, color='#7570b3', linestyle='--', alpha=0.9)
#plt.axhline(y=y_b_max, color='red', linestyle='--')

plt.xlabel('GTV class probability', fontsize=11)
plt.ylabel('Uncertainty value', fontsize=11)
plt.legend()

# Save the plot to a PDF file
plt.savefig('entropy_new.pdf', format='pdf', bbox_inches='tight')

plt.show()

