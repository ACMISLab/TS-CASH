import numpy as np
import pandas as pd

from pylibs.utils.util_gnuplot import Gnuplot

np.random.seed(5)

X_inliers1 = np.random.normal(0, 0.1, size=(50, 2))
X_inliers2 = np.random.normal(2, 0.2, size=(100, 2))
# X_inliers = np.r_[X_inliers + 2, X_inliers - 2]
X_inliers = np.r_[X_inliers1, X_inliers2]
X_outliers = np.asarray([
    [0.5, 0],
    [1.5, 0.2],
])
X = np.r_[X_inliers, X_outliers]

df = pd.DataFrame(X)

gp = Gnuplot()
gp.set_output_pdf("lof_demo", w=3, h=2)
gp.add_data(df, header=False)

gp.unset('unset xtics')
gp.unset('unset ytics')
gp.add_label('C_1', -0.1, 0.45)
gp.add_label('C_2', 1.2, 2.2)
gp.add_label('x_1', 0.55, 0.05)
gp.add_label('x_2', 1.55, 0.25)
gp.plot('plot $df using 1:2 with points title "" ')
gp.write_to_file()
gp.show()
