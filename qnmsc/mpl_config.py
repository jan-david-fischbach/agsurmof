import matplotlib.pyplot as plt

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

if __name__ == "__main__":
    config()
    plt.plot([1,2,3],[1,3,4])
    plt.plot([1,2,3],[3,1,4])
    plt.show()