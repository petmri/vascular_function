import os
import numpy as np

import matplotlib.pyplot as plt

# path to output directory
output_dir = '/media/network_mriphysics/USC-PPG/AI_training/results/aif_comparison'


plt.rcParams['text.usetex'] = True  # Enable LaTeX rendering
plt.rcParams['text.latex.preamble'] = r'\usepackage{sansmath}\sansmath'
#plt.rcParams['text.latex.preamble'] = r'\usepackage{mathpazo}'
#plt.rcParams['text.latex.preamble'] = r'\usepackage{mathastext}\usepackage{helvet}\renewcommand{\familydefault}{\sfdefault}'
#plt.rcParams['text.latex.preamble'] = r'\usepackage{lmodern}'
plt.rcParams.update({'font.size': 16})  # Increase font size for all text elements

# Define the constant
constant = 1

# Create x values
x = np.linspace(0, constant*1.2, 1000)

# Calculate y values using the formula 1-(x/constant)^2
y = 1 - (x/(1.1*constant))**2
plt.figure(figsize=(8, 6))
plt.plot(x, y, 'b-', label='end-to-mean')
plt.text(0.8, 0.8, 
         r"$[1 - \left(\frac{end}{1.1 \times mean}\right)^2] \alpha_1$", 
         fontsize=20, ha='center', va='center', transform=plt.gca().transAxes)
plt.xlabel('End/Mean')
plt.ylabel('Score')
plt.title('End to Mean Quality Score')
plt.axhline(y=0, color='k', linestyle='-', alpha=1)
plt.axvline(x=0, color='k', linestyle='-', alpha=1)
# save figure
plt.savefig(os.path.join(output_dir, 'end_to_mean.png'), dpi=300)


# Define the constant
constant = 1
# Create x values
x = np.linspace(0, constant*1.2, 1000)
# Calculate y values using the formula 1-(x/constant)^2
y = 1 - (x/(constant))**2
plt.figure(figsize=(8, 6))
plt.plot(x, y, 'b-', label='baseline-to-mean')
plt.text(0.8, 0.8, 
         r"$[1 - \left(\frac{baseline}{mean}\right)^2] \alpha_2$", 
         fontsize=20, ha='center', va='center', transform=plt.gca().transAxes)
plt.xlabel('Baseline/Mean')
plt.ylabel('Score')
plt.title('Baseline to Mean Quality Score')
plt.axhline(y=0, color='k', linestyle='-', alpha=1)
plt.axvline(x=0, color='k', linestyle='-', alpha=1)
# save figure
plt.savefig(os.path.join(output_dir, 'baseline_to_mean.png'), dpi=300)


# Define the constant
constant = 1
# Create x values
x = np.linspace(0, constant*4.5, 1000)
# Calculate y 
y = 1 / (1+np.exp(-3.5*x/constant+7.5))
plt.figure(figsize=(8, 6))
plt.plot(x, y, 'b-', label='peak-to-mean')
plt.text(0.8, 0.8, 
         r"$\frac{1}{1 + \mathrm{e}^{-3.5 \cdot \frac{peak}{mean} + 7.5}} \alpha_3$", 
         fontsize=20, ha='center', va='center', transform=plt.gca().transAxes)
plt.xlabel('Peak/Mean')
plt.ylabel('Score')
plt.title('Peak to Mean Quality Score')
plt.axhline(y=0, color='k', linestyle='-', alpha=1)
plt.axvline(x=0, color='k', linestyle='-', alpha=1)
# save figure
plt.savefig(os.path.join(output_dir, 'peak_to_mean.png'), dpi=300)


# Define the constant
constant = 1
# Create x values
x = np.linspace(0, constant, 1000)
# Calculate y 
y = (constant-x)/(constant)
plt.figure(figsize=(8, 6))
plt.plot(x, y, 'b-', label='arrival-time')
plt.text(0.8, 0.8, 
         r"$\frac{end_{time}-peak_{time}}{end_{time}} \alpha_4$", 
         fontsize=20, ha='center', va='center', transform=plt.gca().transAxes)
plt.xlabel(r'$\mathrm{Peak_{time}/End_{time}}$')
plt.ylabel('Score')
plt.title('Arrival Time Quality Score')
plt.axhline(y=0, color='k', linestyle='-', alpha=1)
plt.axvline(x=0, color='k', linestyle='-', alpha=1)
# save figure
plt.savefig(os.path.join(output_dir, 'arrival_time.png'), dpi=300)

plt.show()