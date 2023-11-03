import matplotlib.pyplot as plt

def plot(m = 1, n = 1):
  plt.close()

  # figwith = 6.26894
  figwith = m*6
  figsize = (figwith, figwith*0.6*n)

  fig = plt.figure()
  ax = fig.add_subplot(111)

  ax.tick_params(axis='both', which='both', direction='in', bottom=True, left=True, top=True, right=True, labelbottom=True, labeltop=False, labelleft=True, labelright=False)
  ax.minorticks_on()

  ax.grid(True, which='major', axis='both', linewidth=1, alpha=0.5, linestyle='dashed', zorder=1)
  ax.grid(True, which='minor', axis='x', linewidth=0.5, alpha=0.5, linestyle='dashed', zorder=1)
  ax.grid(True, which='minor', axis='y', linewidth=0.5, alpha=0.5, linestyle='dashed', zorder=1)

  return fig, ax