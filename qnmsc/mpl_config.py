import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection

def config():
    colorcycle = plt.rcParams['axes.prop_cycle']
    # print(colorcycle)
    import scienceplots
    plt.style.use('science')
    plt.style.use('nature')

    params = {
        'text.latex.preamble': r'\usepackage{siunitx} \usepackage{amsmath} \usepackage{amssymb} \usepackage{sfmath} \sisetup{detect-all}',
        'axes.prop_cycle': colorcycle,
        'axes.titlesize': 7
    }
    plt.rcParams.update(params)
    import matplotlib as mpl
    mpl.rcParams['hatch.linewidth'] = 0.2  # previous pdf hatch linewidth

um = r"\unit{\micro \meter}"
inv_um = r"\unit{\per \micro \meter}"

# um = r"$\mu m$"
# inv_um = r"$\mu m ^ {-1}$"

# um = r"$\upmu$ m"

mm = 0.1/2.54

# define an object that will be used by the legend
class MulticolorPatch(object):
    def __init__(self, colors, width_scaling = 1, height_scaling = 0.7):
        self.colors = colors
        self.width_scaling = width_scaling
        self.height_scaling = height_scaling
        
# define a handler for the MulticolorPatch object
class MulticolorPatchHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        width, height = handlebox.width, handlebox.height
        patches = []
        for i, c in enumerate(orig_handle.colors):
            patches.append(plt.Rectangle([width/len(orig_handle.colors) * i - handlebox.xdescent, 
                                          -handlebox.ydescent],
                           width / len(orig_handle.colors) * orig_handle.width_scaling,
                           height*orig_handle.height_scaling, 
                           facecolor=c, 
                           edgecolor='none'))

        patch = PatchCollection(patches,match_original=True)

        handlebox.add_artist(patch)
        return patch

if __name__ == "__main__":
    config()
    plt.plot([1,2,3],[1,3,4])
    plt.plot([1,2,3],[3,1,4])
    plt.show()