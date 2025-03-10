import matplotlib
import matplotlib as mpl
import matplotlib.pylab as plt
import numpy as np

# Say, "the default sans-serif font is COMIC SANS"
matplotlib.rcParams['font.serif'] = "DejaVu Serif"

# Then, "ALWAYS use sans-serif fonts"
matplotlib.rcParams['font.family'] = "serif"

matplotlib.rcParams['axes.linewidth'] = 2

matplotlib.rcParams["mathtext.fontset"] = "dejavuserif"

mpl.rcParams['xtick.major.size'] = 9
mpl.rcParams['xtick.major.width'] = 1.2
mpl.rcParams['xtick.minor.size'] = 4
mpl.rcParams['xtick.minor.width'] = 1.2

mpl.rcParams['ytick.major.size'] = 9
mpl.rcParams['ytick.major.width'] = 1.2
mpl.rcParams['ytick.minor.size'] = 4
mpl.rcParams['ytick.minor.width'] = 1.2

mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'

mpl.rcParams['ytick.right'] = True

mpl.rcParams['ytick.minor.visible'] = True
mpl.rcParams['xtick.minor.visible'] = True

mpl.rcParams['xtick.top'] = True

colors = [
    '#1f77b4',
    '#ff7f0e',
    '#2ca02c',
    '#d62728',
    '#9467bd',
    '#8c564b',
    '#e377c2',
    '#7f7f7f',
    '#bcbd22'
]


def get_mosaic(mosaic='''A''', figsize=(10, 10)):
    mosaic = mosaic
    fig = plt.figure(constrained_layout=True, figsize=figsize)
    ax_dict = fig.subplot_mosaic(mosaic)
    return fig, ax_dict


class Layer:
    def __init__(self,
                 version='scatter',
                 X=None, Y=None, Z=None,
                 XERR=None, YERR=None, W=None,
                 STR=None,
                 rnge=None,
                 label=None, zlabel=None,
                 capsize=None,
                 color='k', shape='o', alpha=1.0, zorder=1.0, size=None,
                 linestyle='-', linewidth=3,
                 bar=None, cumulative=False, bins='auto', density=False,
                 contours=None, smooth=None,
                 bold=False, orientation='vertical', step=None,
                 histtype='step',
                 vmin=None, vmax=None
                 ):
        self.version = version
        self.X = X
        self.Y = Y
        self.Z = Z
        self.XERR = XERR
        self.YERR = YERR
        self.W = W
        self.STR = STR
        self.rnge = rnge
        self.label = label
        self.zlabel = zlabel
        self.color = color
        self.capsize = capsize
        self.size = size
        self.shape = shape
        self.alpha = alpha
        self.zorder = zorder
        self.linestyle = linestyle
        self.linewidth = linewidth
        self.bar = bar
        self.cumulative = cumulative
        self.bins = bins
        self.density = density
        self.contours = contours
        self.smooth = smooth
        self.bold = bold
        self.orientation = orientation
        self.step = step
        self.histtype = histtype
        self.vmin = vmin
        self.vmax = vmax


def add_layer(ax, layer):
    if layer.version == "line":
        ax.plot(layer.X, layer.Y, color=layer.color,
                linestyle=layer.linestyle, linewidth=layer.linewidth,
                label=layer.label, alpha=layer.alpha, zorder=layer.zorder)

    elif layer.version == "scatter":
        if not layer.rnge:
            rnge = [None, None]
        else:
            rnge = layer.rnge
        if isinstance(layer.Z, (np.ndarray, list)):

            if layer.bold:
                p1 = ax.scatter(layer.X, layer.Y, c=layer.Z,
                                cmap='coolwarm', label=layer.label,
                                marker=layer.shape, vmin=rnge[0], vmax=rnge[1], alpha=layer.alpha,
                                edgecolor='k', linewidth=layer.linewidth,
                                s=layer.size)
            else:
                p1 = ax.scatter(layer.X, layer.Y, c=layer.Z,
                                cmap=layer.color, label=layer.label, s=layer.size,
                                marker=layer.shape, vmin=rnge[0], vmax=rnge[1], alpha=layer.alpha, )
            if layer.bar:
                c1 = plt.colorbar(p1, ax=ax, orientation='vertical')
                if layer.zlabel:
                    c1.set_label(layer.zlabel, fontsize=layer.fontsize)
                c1.ax.tick_params(labelsize=layer.fontsize)
        else:
            ax.scatter(layer.X, layer.Y, label=layer.label,
                       color=layer.color, alpha=layer.alpha,
                       marker=layer.shape, s=layer.size)

    elif layer.version == "hist":

        ax.hist(layer.X, bins=layer.bins, zorder=layer.zorder,
                density=layer.density, weights=layer.W,
                color=layer.color, alpha=layer.alpha,
                range=layer.rnge, label=layer.label,
                linestyle=layer.linestyle, linewidth=layer.linewidth,
                histtype=layer.histtype, cumulative=layer.cumulative,
                orientation=layer.orientation)

    elif layer.version == "bar":

        ax.bar(layer.X, height=layer.Y, width=layer.W,
               color='none', alpha=layer.alpha, edgecolor=layer.color,
               label=layer.label,
               linestyle=layer.linestyle,
               align='center', linewidth=4)

    elif layer.version == "vert":

        ax.axvline(layer.X,
                   linestyle=layer.linestyle, color=layer.color,
                   linewidth=layer.linewidth, label=layer.label,
                   alpha=layer.alpha, zorder=layer.zorder)

    elif layer.version == "horiz":

        ax.axhline(layer.X, linestyle=layer.linestyle,
                   color=layer.color, linewidth=layer.linewidth,
                   label=layer.label, alpha=layer.alpha,
                   zorder=layer.zorder)
    elif layer.version == "arrow":

        ax.arrow(layer.X[0], layer.X[1], layer.X[2], layer.X[3],
                 color=layer.color, width=layer.linewidth,
                 label=layer.label, alpha=layer.alpha,
                 zorder=layer.zorder)

    elif layer.version == 'fill':

        ax.fill_between(layer.X, layer.Y[0], layer.Y[1], color=layer.color, alpha=layer.alpha,
                        step=layer.step, label=layer.label)

    elif layer.version == 'step':

        ax.step(layer.X, layer.Y, color=layer.color,
                linewidth=layer.linewidth, alpha=layer.alpha,
                label=layer.label, linestyle=layer.linestyle, where='post')

    elif layer.version == "errorbar":

        ax.errorbar(layer.X, layer.Y,
                    xerr=layer.XERR, yerr=layer.YERR, zorder=layer.zorder,
                    color=layer.color, alpha=layer.alpha, markersize=layer.size,
                    label=layer.label, linestyle=layer.linestyle,
                    capsize=layer.capsize, marker=layer.shape)

    elif layer.version == "text":
        ax.text(layer.X, layer.Y, layer.STR, fontsize=layer.fontsize, color=layer.color, alpha=layer.alpha)

    elif layer.version == 'hist2d':
        if not layer.rnge:
            rnge = [[min(layer.X), max(layer.X)], [min(layer.Y), max(layer.Y)]]
        else:
            rnge = layer.rnge
        H, xbins, ybins = np.histogram2d(layer.X, layer.Y,
                                         weights=layer.W, bins=layer.bins, range=rnge)

        H = np.rot90(H)
        H = np.flipud(H)

        X, Y = np.meshgrid(xbins[:-1], ybins[:-1])

        if layer.smooth is None:
            from scipy.signal import wiener
            H = wiener(H, mysize=layer.smooth)

        H = H / np.sum(H)
        Hmask = np.ma.masked_where(H == 0, H)

        cmin = 1e-4
        cmax = 1.0
        if not layer.vmin:
            vmin = cmin * np.max(Hmask)
        else:
            vmin = layer.vmin

        if not layer.vmax:
            vmax = cmax * np.max(Hmask)
        else:
            vmax = layer.vmax

        # norm = LogNorm(vmin,vmax),
        p1 = ax.pcolormesh(X, Y, (Hmask), cmap=layer.color, vmin=vmin, vmax=vmax,
                           linewidth=0., shading='auto', alpha=layer.alpha, edgecolors=None)
        p1.set_edgecolor('none')

        if layer.bar:
            c1 = plt.colorbar(p1, ax=ax, orientation='vertical')
            if layer.zlabel:
                c1.set_label(layer.zlabel, fontsize=layer.fontsize)
            c1.ax.tick_params(labelsize=layer.fontsize)


def make_plot(
        layers,
        ax=None,
        xlabel=None, ylabel=None,
        fontsize=20,
        title=None,
        xlim=None, ylim=None,
        legend=False, square=False,
        savename=None, show=False,
        figsize=(8, 5),
        frameon=False,
        xticks=None, yticks=None,
        xreverse=False, yreverse=False,
        xrotation=None, yrotation=None,
        xlog=False, ylog=False,
        loc="best", stack=False, ncol=1, dpi=100):
    if ax is None:
        fig, ((ax)) = plt.subplots(nrows=1, ncols=1)
        fig.set_size_inches(figsize)
    # add layers
    for layer in layers:
        layer.fontsize = fontsize
        add_layer(ax, layer)

    if xlabel:
        ax.set_xlabel(xlabel, fontsize=fontsize)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=fontsize)

    ax.tick_params(axis='both', which='major', labelsize=fontsize, length=10, width=2)
    ax.tick_params(axis='both', which='minor', labelsize=fontsize, length=3, width=1)

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    if legend:
        ax.legend(fontsize=fontsize, frameon=frameon, loc=loc, ncol=ncol)

    if title:
        ax.set_title(title, fontsize=fontsize)

    ax.ticklabel_format(useOffset=False)

    if ylog:
        ax.set_yscale("log")

    if xlog:
        ax.set_xscale("log")

    if xticks:
        if xticks == 'None':
            ax.set_xticklabels([], fontsize=fontsize)
        else:
            ax.set_xticks(xticks[0])
            ax.set_xticklabels(xticks[1], fontsize=fontsize, rotation=xrotation)

            ax.tick_params(axis='x', which='major', length=10, width=2)
            ax.tick_params(axis='x', which='minor', labelsize=0, length=5, width=1)

    if yticks:
        if yticks == 'None':
            ax.set_yticklabels([], fontsize=fontsize)
        else:
            ax.set_yticks(yticks[0])
            ax.set_yticklabels(yticks[1], fontsize=fontsize, rotation=yrotation)
            ax.tick_params(axis='y', which='major', length=10, width=2)
            ax.tick_params(axis='y', which='minor', labelsize=0, length=5, width=1)

    if yreverse:
        ax.invert_yaxis()
    if xreverse:
        ax.invert_xaxis()

    if type(ax) is None:
        fig.tight_layout()

    if square:
        ax.set_aspect('equal', adjustable='box')
    if show:
        plt.show()
    if savename:
        fig.tight_layout()
        fig.savefig(savename, dpi=dpi)
