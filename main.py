import matplotlib.pyplot as plt
import numpy as np
from  generationSignal import triangular_pulse

t,g = triangular_pulse(peak_time=2, rise_angle=np.pi/4, 
                         fall_angle=np.pi/6, A=1.5)
plt.plot(t,g)

plt.xlabel('Время')
plt.ylabel('Амплитуда')
plt.grid(True)
plt.legend()
plt.show()


