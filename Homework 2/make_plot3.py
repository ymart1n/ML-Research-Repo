# This script was produced by glue and can be used to further customize a
# particular plot.

### Package imports

from glue.core.state import load
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
matplotlib.use('Agg')

### Set up data

data_collection = load('make_plot3.py.data')

### Set up viewer

# Initialize figure
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, aspect='auto')

### Set up layers

## Layer 1: movie

layer_data = data_collection[0]

# Get main data values
x = layer_data['Score']
y = layer_data['RatingCount']
keep = ~np.isnan(x) & ~np.isnan(y)

ax.plot(x[keep], y[keep], 'o', color='#00aaff', markersize=3, alpha=1.0, zorder=1, mec='none')

### Finalize viewer

# Set limits
ax.set_xlim(1.484, 10.016)
ax.set_ylim(-41549.08, 1080465.08)

# Set scale (log or linear)
ax.set_xscale('linear')
ax.set_yscale('linear')

# Set axis label properties
ax.set_xlabel('Score', weight='normal', size=10)
ax.set_ylabel('RatingCount', weight='normal', size=10)

# Set tick label properties
ax.tick_params('x', labelsize=8)
ax.tick_params('y', labelsize=8)

# Save figure
fig.savefig('glue_plot.png')
plt.close(fig)