import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(0, 1, 100)
y_e = -x * np.log(x) 
#y_b = -x * np.log(x) - (1-x) * np.log(1-x)

y_e_max = 1/np.e
x_e_max = 1/np.e
#y_b_max = np.max(y_b)

plt.plot(x, y_e, label='Entropy $E$', color='darkorange')
#plt.plot(x, y_b)

plt.axvline(x=0.5, color='lightseagreen', linestyle='--', alpha=0.4)
plt.axvline(x=x_e_max, color='orchid', linestyle='--', alpha=0.9)

plt.axhline(y=y_e_max, color='orchid', linestyle='--', alpha=0.9)
#plt.axhline(y=y_b_max, color='red', linestyle='--')

plt.xlabel('Positive class probability $p_i$')
plt.ylabel('Uncetainty value')
plt.legend()

# Save the plot to a PDF file
plt.savefig('entropy.pdf', format='pdf', bbox_inches='tight')

plt.show()

