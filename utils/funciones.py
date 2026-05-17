import numpy as np
from numba import njit

# Hamiltonian Function
@njit
def Hamiltonian_potential(r, p, mass, G_const, conditional="all"):   
    n_batch = r.shape[0]
    n_bodies = r.shape[1]
    
    Ec = np.zeros(n_batch)
    if conditional == "cm" or conditional == "all":
        for i in range(n_batch):
            sum_ec = 0.0
            for j in range(n_bodies):
                p_norm_sq = p[i, j, 0]**2 + p[i, j, 1]**2 + p[i, j, 2]**2
                sum_ec += p_norm_sq / (2.0 * mass[j])
            Ec[i] = sum_ec
            
    Ep = np.zeros(n_batch)
    if conditional == "pos" or conditional == "all":
        for i in range(n_batch):
            sum_ep = 0.0
            for g in range(n_bodies - 1):
                for m in range(g + 1, n_bodies):
                    dx = r[i, g, 0] - r[i, m, 0]
                    dy = r[i, g, 1] - r[i, m, 1]
                    dz = r[i, g, 2] - r[i, m, 2]
                    r_scalar = (dx**2 + dy**2 + dz**2)**0.5
                    sum_ep -= G_const * mass[g] * mass[m] / r_scalar
            Ep[i] = sum_ep
                
    return Ec, Ep

@njit
def force_fun(data, G, mass):
    force = np.zeros((data.shape[0],3))
    for i in range(data.shape[0]):
        for j in range(data.shape[0]):
            if i == j: 
                continue
            force[i,:] += G * mass[i] * mass[j] * (data[j,:] - data[i,:]) / ((np.linalg.norm(data[j,:] - data[i,:]))**3)
    return force

coef_c = np.array([0.6756035959798289,-0.1756035959798288,-0.1756035959798288,0.6756035959798289], dtype=np.float64)
coef_d = np.array([1.3512071919596578,-1.7024143839193153,1.3512071919596578], dtype=np.float64)

@njit
def Yoshida(r, p, mass, G_const, dt_const, coef_c=coef_c, coef_d=coef_d):

    num_stages_p = len(coef_d)

    for i in range(num_stages_p):
        r = r + coef_c[i] * dt_const * p / mass[:, None]
        p = p + coef_d[i] * dt_const * force_fun(r, G_const, mass)

    r = r + coef_c[-1] * dt_const * p / mass[:, None]

    return r, p

def get_yoshida_coefficients(degree):
    """
    Generates the c and d coefficients for an order 'degree' (must be even).
    Based on the recursion S_{2n+2} = S_{2n}(z1) S_{2n}(z0) S_{2n}(z1)
    """
    # We start with order 2 (Basic Leapfrog/Verlet)
    c = np.array([0.5, 0.5], dtype=np.float64)
    d = np.array([1.0], dtype=np.float64)
    
    # We iterate in steps of 2 until reaching the desired degree
    for current_n in range(2, degree, 2):
        # n in the Yoshida formula is (current_order / 2)
        formula_n = current_n / 2
        
        # Calculation of the x1 and x0 factors according to Yoshida
        power = 1.0 / (current_n + 1) # It is 1/(2n+1)
        x1 = 1.0 / (2.0 - 2.0**power)
        x0 = 1.0 - 2.0 * x1
        
        # Apply the triple composition: blocks of x1, then x0, then x1
        # This expands the c and d arrays
        new_d = np.concatenate([x1 * d, x0 * d, x1 * d])
        
        # For c, we remember the fusion of the extremes (final_c + initial_c) 
        c_center_1 = x1 * c[-1] + x0 * c[0]
        c_center_2 = x0 * c[-1] + x1 * c[0]
        
        new_c = np.concatenate([
            x1 * c[:-1], 
            [c_center_1], 
            x0 * c[1:-1], 
            [c_center_2], 
            x1 * c[1:]
        ])
        
        c, d = new_c, new_d
        
    return c, d