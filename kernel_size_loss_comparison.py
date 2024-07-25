import os

log_files = []
lowest_loss = float('inf')
losses = {}

# Define the path to the weights folder
weights_folder = '/home/mrispec/AUTOAIF_DATA/weights/hp_tuning'

# Iterate over the subfolders in the weights folder
for folder_name in os.listdir(weights_folder):
    folder_path = os.path.join(weights_folder, folder_name)
    
    # Check if the subfolder name matches the specified pattern
    if folder_name.startswith('run-'):
        log_file_path = os.path.join(folder_path, 'log.txt')
        kernel_size = folder_name.split('-')[-2:]
        kernel_size = '-'.join(kernel_size)
        print(kernel_size)
        
        
        # Read the last line of the log file
        with open(log_file_path, 'r') as log_file:
            lines = log_file.readlines()
            for line in lines:
                if line.startswith('New lowest loss:'):
                    loss = float(line.split(':')[-1])
            
            if loss < 300:
                losses[kernel_size] = loss
                lowest_loss = min(lowest_loss, loss)
                print(loss)

# sort the losses by value
sorted_losses = sorted(losses.items(), key=lambda kv: kv[1])
print(sorted_losses)

print('Lowest loss: ', lowest_loss)

# variables
firstlast_kernel_z_options = (3, 5, 7, 9, 11)
firstlast_kernel_xy_options = (3, 5, 7, 9, 11)
body_kernel_z_options = (3, 5, 7, 9)
body_kernel_xy_options = (3, 5, 7, 9, 11)

# search losses for kernel size
possible_kernel_sizes = []
for kernel_z in firstlast_kernel_z_options:
    for kernel_xy in firstlast_kernel_xy_options:
        for body_kernel_z in body_kernel_z_options:
            for body_kernel_xy in body_kernel_xy_options:
                kernel_size = f'({kernel_z}, {kernel_xy}, {kernel_xy})-({body_kernel_z}, {body_kernel_xy}, {body_kernel_xy})'
                if kernel_size in losses:
                    possible_kernel_sizes.append(kernel_size)
                    print(kernel_size, losses[kernel_size])

print(possible_kernel_sizes)
x = []
y = []
size = []
shape = []
for kernel_size in possible_kernel_sizes:
    # split kernel size into numbers
    # (3, 3, 3)-(3, 3, 3) -> [3, 3, 3, 3, 3, 3]
    kernel_size = kernel_size.replace('(', '').replace(')', '').replace('-', ',')
    kernel_size = kernel_size.split(',')
    # kernel_size = [int(x) for x in kernel_size]
    print(kernel_size)
    x.append(int(kernel_size[0]))
    y.append(int(kernel_size[1]))
    size.append(int(kernel_size[3]))
    shape.append(int(kernel_size[4]))
    

# 5d plot using plotly
import plotly
import plotly.graph_objects as go
import numpy as np

# define the x, y, and z values
# x = firstlast_kernel_z_options
# y = firstlast_kernel_xy_options
# color = body_kernel_z_options
# size = body_kernel_xy_options
z = []
for ao_z, ao_xy, body_z, body_xy in zip(x, y, size, shape):
    # print(x, y, shape, size)
    z.append(losses[f'({ao_z}, {ao_xy}, {ao_xy})-({body_z}, {body_xy}, {body_xy})'])
print("Size: ", size)
for i in range(len(shape)):
    if shape[i] == 3:
        shape[i] = 'circle'
    elif shape[i] == 5:
        shape[i] = 'square'
    elif shape[i] == 7:
        shape[i] = 'diamond'
    elif shape[i] == 9:
        shape[i] = 'x'
    elif shape[i] == 11:
        shape[i] = 'cross'
# print(len(z))
# print(x)
# double size values

# fig1 = go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=size, symbol=shape, opacity=0.9))

# layout = go.Layout(scene=dict(xaxis=dict(title='First/Last Kernel Z'), yaxis=dict(title='First/Last Kernel XY'), zaxis=dict(title='Loss')))
size = [x*3 for x in size]
legend_groups = [f"{shape[i]}-{size[i]}" for i in range(len(shape))]
legend_entries = set(zip(shape, size))
shape_dict = {'circle': '3', 'square': '5', 'diamond': '7', 'x': '9', 'cross': '11'}
# Create scatter plot traces for each marker shape and size
traces = []
for i, (marker_shape, marker_size) in enumerate(legend_entries):
    # Filter indices where shape and size match
    indices = [j for j in range(len(shape)) if shape[j] == marker_shape and size[j] == marker_size]

    trace = go.Scatter3d(
        x=[x[j] for j in indices],
        y=[y[j] for j in indices],
        z=[z[j] for j in indices],
        mode='markers',
        marker=dict(size=[size[j] for j in indices], symbol=[shape[j] for j in indices], opacity=0.9),
        name=f"({int(marker_size/3)}, {shape_dict[marker_shape]}, {shape_dict[marker_shape]})",
        legendgroup=f"legendgroup_{i}"
    )
    traces.append(trace)

# Create layout
layout = go.Layout(
    scene=dict(xaxis=dict(title='First/Last Kernel Z'), yaxis=dict(title='First/Last Kernel XY'), zaxis=dict(title='Loss')),
    legend=dict(title=dict(text='Marker Legend')),
    showlegend=True
)

# Create figure with the traces and layout
fig = go.Figure(data=traces, layout=layout)

# Show the plot
fig.show()
# plot and save
plotly.offline.plot({"data": [fig], "layout": layout}, filename='kernel_size_loss_comparison.html', auto_open=True)