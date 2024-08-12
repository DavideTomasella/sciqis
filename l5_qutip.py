# In[]
import qutip as q
import numpy as np
import matplotlib.pyplot as plt

def solve_qed(om_a, om_c, Om, gamma=0, alpha=0, t=np.linspace(0, 100, 200), return_states=True):
    """Solve the cavity QED system using qutip.mesolve.

    Parameters:
    om_a  : atomic transition frequency
    om_c  : cavity resonance frequency
    Om    : vacuum Rabi frequency (coupling strength)
    gamma : atomic decay rate
    alpha : amplitude of the initial coherent state of the cavity 
    t     : array of time values at which to evaluate the state evolution
    """
    om_a = pi2 * om_a
    om_c = pi2 * om_c
    Om = pi2 * Om
    gamma = pi2 * gamma
    H = om_a/2 * sz + om_c * ad*a + Om/2 * (sp * a + sm * ad)
    rho0 = q.tensor(q.basis(2,0), q.coherent(N, alpha))

    if return_states:
        e_ops = None
    else:
        e_ops = [sp*sm, ad*a]
    
    if gamma > 0:
        c_ops = [np.sqrt(gamma) * sm]
    else:
        c_ops = None
        
    result = q.mesolve(H, rho0, t, c_ops, e_ops, options={"nsteps":5000})
    
    return result

# In[]
N = 5
sz = q.tensor(q.sigmaz(), q.identity(N))
sm = q.tensor(q.sigmam(), q.identity(N))
sp = q.tensor(q.sigmap(), q.identity(N))
a = q.tensor(q.identity(2), q.destroy(N))
ad = q.tensor(q.identity(2), q.create(N))
pi2 = 2*np.pi
result = solve_qed(om_a=1, om_c=1, Om=.05, gamma=0, alpha=3, t=np.linspace(0,300,500), return_states=False)
plt.subplots()
plt.plot(result.times, result.expect[0])

# In[]
alphas = np.linspace(0, 4, 4)
Oms = [.025, .05, .1]
om_a = 1
om_c = 1
t = np.linspace(0, 200, 200)

fig, axs = plt.subplots(len(alphas), len(Oms), 
                        sharex=True, sharey=True, 
                        figsize=(9,9), gridspec_kw=dict(wspace=0, hspace=0))
for i, alpha in enumerate(alphas):
    for j, Om in enumerate(Oms):
        result = solve_qed(om_a, om_c, Om, gamma=0, alpha=alpha, t=t, return_states=False)
        axs[i,j].plot(t, result.expect[0])
        axs[i,j].text(.5, .9, f'Ω = 2π × {Om:.3f}\nα = 2π × {alpha:.1f}', 
                      size=8, ha='left', va='top',
                      bbox=dict(ec=(.5,.5,.5), fc=(.8,.8,1,.8)), transform=axs[i,j].transAxes)
        if i==len(alphas)-1:
            axs[i,j].set_xlabel('t')
    axs[i,0].set_ylabel('population')
# In[]
from matplotlib.animation import ArtistAnimation
from IPython.display import HTML
N = 30
H = q.num(N)
rho0 = q.coherent(N, 2)
tlist = np.linspace(0,4*np.pi)
sz = q.tensor(q.sigmaz(), q.identity(N))
sm = q.tensor(q.sigmam(), q.identity(N))
sp = q.tensor(q.sigmap(), q.identity(N))
a = q.tensor(q.identity(2), q.destroy(N))
ad = q.tensor(q.identity(2), q.create(N))
pi2 = 2*np.pi
sim_out = q.mesolve(H, rho0, tlist) 
xvec = np.linspace(-5,5,100)
ims = []
fig, ax = plt.subplots()
ax.set_aspect(1)

for s in sim_out.states:
    W = q.wigner(s, xvec, xvec)
    im = plt.contourf(xvec, xvec, W,
                      vmin=-1/np.pi, vmax=1/np.pi)
    ims.append([im])

ani = ArtistAnimation(fig, ims, interval=50, blit=True)
HTML(ani.to_html5_video())