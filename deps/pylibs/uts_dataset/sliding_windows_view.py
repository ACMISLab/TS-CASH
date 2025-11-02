import pandas as pd
from pylibs.uts_dataset.dataset_loader import DatasetLoader
from pylibs.utils.util_file import FileUtil

dl = DatasetLoader("IOPS", "KPI-e0747cad-8dc8-38a9-a9ab-855b61f5551d.train.out")
data=dl.get_source_data()
values=data['value'].values.tolist()
l=64
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.pyplot as plt

# 创建图
fig: Figure = plt.figure(figsize=(10, 10), tight_layout=True)

# 添加子图
# add_subplot(n_rows, n_cols, exp_index)
ax: Axes = fig.subplots(5,1)
# ax.plot(train_x)
for i in range(5):
    ax[i].plot(values[i:i+l])
    ax[i].set_title(f"Rolling window {i}")

plt.show()