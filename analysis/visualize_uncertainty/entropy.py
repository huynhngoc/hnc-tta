import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(0, 1, 100)
y_e = -x * np.log(x) 
#y_b = -x * np.log(x) - (1-x) * np.log(1-x)

y_e_max = np.max(y_e)
#y_b_max = np.max(y_b)

plt.plot(x, y_e, label='Entropy $E$', color='darkorange')
#plt.plot(x, y_b)

plt.axvline(x=0.5, color='orchid', linestyle='--', alpha=0.9)
#plt.axhline(y=y_e_max, color='red', linestyle='--')
#plt.axhline(y=y_b_max, color='red', linestyle='--')

plt.xlabel('Positive class probability $p$')
plt.ylabel('Uncetainty value')
plt.legend()

# Save the plot to a PDF file
plt.savefig('entropy.pdf', format='pdf', bbox_inches='tight')

plt.show()

