import matplotlib.pyplot as plt
import matplotlib as mpl
#####################################
from matplotlib import rcParams
rcParams['font.family'] = 'SimHei'
plt.rcParams.update({'figure.dpi':300})
fig, ax = plt.subplots(figsize=(0.5, 10))
fig.subplots_adjust(bottom=0.5)   # 设置子图到下边界的距离

cmap = mpl.cm.jet
norm = mpl.colors.Normalize(vmin=0, vmax=1)

fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
             cax=ax)


plt.show()
plt.savefig(('colorbar.jpg'),bbox_inches='tight',pad_inches = 0.0)  

