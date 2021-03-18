import numpy as np
from scipy import sparse
from scipy.sparse import linalg
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 17})

sigma = 1
gfactor = 1/np.sqrt(2*np.pi*sigma*sigma)

PDE = {
       'domain': [(0, 1), (-1, 1)], # time x space
       'num_steps': [10*50*50, 50], # time, space
       'initial_function': lambda x: gfactor * np.exp(-0.5*(x**2)/(sigma**2)),
}

def build_pde(pde):
    t0, t1 = pde['domain'][0]
    x0, x1 = pde['domain'][1]
    initial_function = pde['initial_function']
    num_time_steps, num_space_steps = pde['num_steps']

    pde.update({
        'h': (x1 - x0) / num_space_steps,
        'tau': (t1 - t0) / num_time_steps,
        'space_mesh': np.linspace(x0, x1, num_space_steps+2),
        'boundary_values': [initial_function(x0), initial_function(x1)]
        })

def solve_heat_forward(pde):
    h, tau = pde['h'], pde['tau']
    time_steps, space_steps = pde['num_steps']

    p = np.zeros((time_steps, space_steps+2))
    p[0, :] = np.vectorize(pde['initial_function'])(pde['space_mesh'])
    p0, p1 = p[0, 0], p[0, -1]

    for i in range(1, time_steps):
        prev = p[i-1]
        p[i,1:-1] = prev[1:-1] + (tau/(h**2)) * (prev[2:] - 2*prev[1:-1] + prev[:-2])
        p[i, 0] = p0
        p[i, -1] = p1

    # No velocity at the right-most node
    vel = np.zeros((time_steps, space_steps+1))
    for i in range(time_steps):
        vel[i] = (p[i,1:] - p[i,:-1])/(h*p[i,:-1])

    return p, vel

def solve_transport_backwards(pde, vel, p_final):
    h, tau = pde['h'], pde['tau']
    time_steps, space_steps = pde['num_steps']

    q = np.zeros((time_steps, space_steps + 2))
    q[-1] = p_final

    for i in reversed(range(time_steps-1)):
        #if i % (time_steps//10) == 0:
        #    print("Time step {}".format(i))

        main_diag = np.ones((space_steps+2,))
        main_diag[1:-1] += (tau/h)*vel[i][1:]

        bottom_diag = np.zeros((space_steps+1,))
        bottom_diag[1:-1] = -(tau/h)*vel[i][1:-1]

        f = np.array(q[i+1])
        f[1] += (tau/h)*vel[i][0]*q[i+1, 0]

        A = sparse.diags([main_diag, bottom_diag], [0, -1], format="csr")
        q[i] = linalg.spsolve(A, f)

    return q

def solve_transport_backwards_blackbox(pde, vel, p_final, method='RK45'):
    h, tau = pde['h'], pde['tau']
    time_steps, space_steps = pde['num_steps']
    t0, t1 = pde['domain'][0]
    def time_idx(t):
        idx = int((t1 - t)//tau)
        rem = ((t1 - t) % tau) / tau
        return max(0, min(time_steps-1, idx)), rem
    def func(t, y):
        idx, rem = time_idx(t)
        if rem < 0.0005 or idx == len(vel)-1:
            v = vel[idx]
        else:
            v = (1-rem)*vel[idx] + rem*vel[idx+1]
        der = np.zeros_like(y)
        der[1:-1] = -(v[1:]*y[1:-1] - v[:-1]*y[:-2])/h
        return der

    sol = solve_ivp(func,
                     [t0, t1],
                     y0 = p_final,
                     t_eval = np.linspace(t0, t1, time_steps),
                     method = method,
                     max_step = tau)

    return np.flip(np.transpose(sol.y), axis=0)

def solve_transport_backwards_FE(pde, vel, p_final):
    h, tau = pde['h'], pde['tau']
    time_steps, space_steps = pde['num_steps']
    t0, t1 = pde['domain'][0]

    q = np.zeros((time_steps, space_steps + 2))
    q[-1] = p_final

    for i in reversed(range(time_steps-1)):
        q[i] = q[i+1]
        q[i, 1:-1] += -(tau/h) * (vel[i+1, 1:]*q[i+1, 1:-1] - vel[i+1, :-1]*q[i+1, :-2])

    return q

#%%
print("Preparing data...")
build_pde(PDE)

print("Solving the heat equation forward...")
p, vel = solve_heat_forward(PDE)

print("Solving the backwards transport equation using BE...")
q = solve_transport_backwards(PDE, vel, p[-1])

print("Solving the backwards transport equation using black box...")
#q2 = solve_transport_backwards_blackbox(PDE, vel, p[-1], method='RK23')
q21 = solve_transport_backwards_blackbox(PDE, vel, p[-1], method='RK45')
#q22 = solve_transport_backwards_blackbox(PDE, vel, p[-1], method='Radau')
#q23 = solve_transport_backwards_blackbox(PDE, vel, p[-1], method='DOP853')

print("Solving the backwards transport equation using FE...")
q3 = solve_transport_backwards_FE(PDE, vel, p[-1])


#%%
# =======================================================
# ================== Plot Solution ======================
# =======================================================

mesh = PDE['space_mesh']
T = PDE['num_steps'][0]
plt.figure()
plt.title("True Solution p")
plt.plot(mesh, p[0], label="T=0")
plt.plot(mesh, p[T//4], label='T=0.25')
plt.plot(mesh, p[2*T//4], label='T=0.5')
plt.plot(mesh, p[3*T//4], label='T=0.75')
plt.legend()
plt.show()

#%%
# =======================================================
# =================== Plot Errors =======================
# =======================================================


T = np.array(range(len(p)))
err = np.max(np.abs(p-q), axis=1)[:len(T)]
#err2 = np.max(np.abs(p-q2), axis=1)[:len(T)]
err21 = np.max(np.abs(p-q21), axis=1)[:len(T)]
#err22 = np.max(np.abs(p-q22), axis=1)[:len(T)]
#err23 = np.max(np.abs(p-q23), axis=1)[:len(T)]
err3 = np.max(np.abs(p-q3), axis=1)[:len(T)]


#L=20
#factor = 1/np.min(err[:L])
#Y = np.log(factor*err[:L])
#B, logA = np.polyfit(T[:L], Y, 1)

plt.figure()
plt.title("Reconstruction Errors for sigma={:.1f}".format(sigma))
plt.plot(T, err, label='Absolute error for BE')
#plt.plot(T, err2, label='Absolute error for RK23')
plt.plot(T, err21, label='Absolute error for RK45')
#plt.plot(T, err22, label='Absolute error for Radau')
#plt.plot(T, err23, label='Absolute error for DOP853')
plt.plot(T, err3, label='Absolute error for FE')
plt.yscale('log')


#plt.plot(T, np.exp(logA - np.log(factor) + B*T), label='Exponential Fit')
plt.xlabel('Time Iteration')
plt.ylabel('Maximum Error')
plt.legend()
plt.show()

plt.savefig('toy_project.png')
