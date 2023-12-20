import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
import matplotlib as mpl

import matplotlib
matplotlib.rc('figure', figsize=(10, 5))
plt.style.use('ggplot')

# fonctions auxilliaires

def get_compressibility(G,i):
    """Retourne la compressibilité de la chambre i"""
    return G.nodes[i]["compressibility"]

def get_conductivity(G,i,j):
    """Retourne la conductivité du conduit entre i et j"""
    try :
        # print(i,j,G.get_edge_data(i,j)["conductivity"])
        return G.get_edge_data(i,j)["conductivity"]
    except:
        # print(i,j,0)
        return 0.
    
    
# Construction de la matrice du système A

def build_matrix_A(G):
    """ 
    Construit à partir de la description du système sous forme de graph la matrice A du système dp/dt = Ap
    """
    
    # Initiation matrice
    N = len(G) - 1 # N'inclut pas la source 
    A = np.zeros((N,N))
    
    # Remplissage des coefficients 
    for i in range(1,N+1):
        for j in range(1,N+1):
            
            # Terme en i,i
            if j == i: 
                # print("i=j",i)
                A[i-1,i-1] = - sum([get_conductivity(G,i,k) for k in range(N+1) if k != i]) / get_compressibility(G,i)
                
            # Terme en i,j (j!=i)
            else : 
                # print("i!=j",i,j)
                A[i-1,j-1] = get_conductivity(G,i,j)/get_compressibility(G,i)
        
    return A 

# Construction du vecteur de pondération du terme source
def build_vector_B(G):
    
    # Initiation matrice
    N = len(G) - 1 # N'inclut pas la source 
    B = np.zeros(N)
    
    for i in range(1,N+1):
        B[i-1] = get_conductivity(G,0,i)/get_compressibility(G,i)
        
    return B
        
def compute_pressure_time_serie(G, source, t_max, p0):
    
    # Domaine temporel de résolution (système adimentionalisé : temps caractéristique ~ 1)
    t_space = np.linspace(0.,t_max,1000)
    
    # Construction matrice du système 
    A = build_matrix_A(G)
    B = build_vector_B(G)
        
    source_B = lambda t : B * source(t)

    # Résolution du système
    def system(p, t):
        dpdt = np.dot(A, p) + source_B(t)
        return dpdt 

    p = odeint(system, p0, t_space)
    
    return p


# ---------- PRESSURE DEPENDANT CONDUCTIVITIES ----------------


def get_conductivity_p_dep(G,i,j,p,p_ts=1e6):
    """Retourne la conductivité du conduit entre i et j"""
    if p < p_ts : return 0.
    try :
        # print(i,j,G.get_edge_data(i,j)["conductivity"])
        return G.get_edge_data(i,j)["conductivity"]
    except:
        # print(i,j,0)
        return 0.

def build_matrix_A_p_dependant(G,p,p_ts=1e5):
    """ 
    Construit à partir de la description du système sous forme de graph la matrice A du système dp/dt = Ap
    """
    
    # Initiation matrice
    N = len(G) - 1 # N'inclut pas la source 
    A = np.zeros((N,N))
    
    # Remplissage des coefficients 
    for i in range(1,N+1):
        for j in range(1,N+1):
            
            # Terme en i,i
            if j == i: 
                # print("i=j",i)
                A[i-1,i-1] = - sum([get_conductivity_p_dep(G,i,k,np.abs(p[i-1]-p[k-1]),p_ts=p_ts) for k in range(N+1) if k != i]) / get_compressibility(G,i)
                
            # Terme en i,j (j!=i)
            else : 
                # print("i!=j",i,j)
                A[i-1,j-1] = get_conductivity_p_dep(G,i,j,np.abs(p[i-1]-p[j-1]),p_ts=p_ts)/get_compressibility(G,i)
        
    return A 

def compute_pressure_time_serie_p_dep(G, source, t_max, p0, p_ts=1e5):
    
    # Domaine temporel de résolution (système adimentionalisé : temps caractéristique ~ 1)
    t_space = np.linspace(0.,t_max,1000)
    
    # Construction matrice du système 
    B = build_vector_B(G)
        
    source_B = lambda t : B * source(t)

    # Résolution du système
    def system(p, t):
        A = build_matrix_A_p_dependant(G,p,p_ts=p_ts)
        dpdt = np.dot(A, p) + source_B(t)
        return dpdt 

    p = odeint(system, p0, t_space)
    
    return p