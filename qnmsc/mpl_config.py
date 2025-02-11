def config():
    import matplotlib.pyplot as plt
    import scienceplots
    plt.style.use('science')
    plt.style.use('nature')

    params = {'text.latex.preamble': r'\usepackage{siunitx} \usepackage{amsmath} \usepackage{amssymb} \usepackage{sfmath}'}
    plt.rcParams.update(params)
    import matplotlib as mpl
    mpl.rcParams['hatch.linewidth'] = 0.2  # previous pdf hatch linewidth

um = r"\unit{\micro \meter}"
inv_um = r"\unit{\per \micro \meter}"
mm = 0.1/2.54