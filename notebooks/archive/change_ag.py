wfreq = np.linspace(5, 10)

eps_Ag = 4.60853575 + 9055.04799147j * (1/(wfreq) - 1/(wfreq+0.21903558j))

Agw0 = 2*np.pi*134.39519365 / 300
Aggamma = 2*np.pi*10.61865288 / 300
Agwp = 2*np.pi*1867.27147966 / 300

eps_Ag_prev = 1 + Agwp**2 / (Agw0**2 - wfreq**2 - 1j*Aggamma*wfreq)

plt.plot(wfreq, eps_Ag.real, label="real")
plt.plot(wfreq, eps_Ag.imag, label="imag")

plt.plot(wfreq, eps_Ag_prev.real, label="real")
plt.plot(wfreq, eps_Ag_prev.imag, label="imag")

plt.xlabel("$\omega$")
plt.ylabel(r"$\varepsilon_\mathrm{Ag}$")
plt.legend()