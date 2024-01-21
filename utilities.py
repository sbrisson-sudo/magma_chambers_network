import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib 
from mpl_toolkits.mplot3d import art3d
import networkx as nx


def _transform_zdir(zdir):
    zdir = art3d.get_dir_vector(zdir)
    zn = zdir / np.linalg.norm(zdir)

    cos_angle = zn[2]
    sin_angle = np.linalg.norm(zn[:2])
    if sin_angle == 0:
        return np.sign(cos_angle) * np.eye(3)

    d = np.array((zn[1], -zn[0], 0))
    d /= sin_angle
    ddt = np.outer(d, d)
    skew = np.array([[0, 0, -d[1]], [0, 0, d[0]], [d[1], -d[0], 0]], dtype=np.float64)
    return ddt + cos_angle * (np.eye(3) - ddt) + sin_angle * skew


def set_3d_properties(self, verts, zs=0, zdir="z"):
    zs = np.broadcast_to(zs, len(verts))
    self._segment3d = np.asarray(
        [
            np.dot(_transform_zdir(zdir), (x, y, 0)) + (0, 0, z)
            for ((x, y), z) in zip(verts, zs)
        ]
    )

def pathpatch_translate(pathpatch, delta):
    pathpatch._segment3d += np.asarray(delta)

art3d.Patch3D.set_3d_properties = set_3d_properties
art3d.Patch3D.translate = pathpatch_translate

def to_3d(pathpatch, z=0.0, zdir="z", delta=(0, 0, 0)):
    if not hasattr(pathpatch.axes, "get_zlim"):
        raise ValueError("Axes projection must be 3D")
    art3d.pathpatch_2d_to_3d(pathpatch, z=z, zdir=zdir)
    pathpatch.translate(delta)
    return pathpatch

matplotlib.patches.Patch.to_3d = to_3d

def plot_3D_geometry(sills_xyzR, conduits_xyzz, volume_depth_top, volume_depth_bottom, volume_radius):
    
    n_sills = sills_xyzR.shape[0]
    
    # Plot de la géometrie 
    zmax = volume_depth_top * 0.5
    zmin = volume_depth_bottom * 1.3

    xmin = -np.max(sills_xyzR[:,3])*1.4
    xmax = -1*xmin
    ymin = -np.max(sills_xyzR[:,3])*1.1
    ymax = -1*ymin

    #fig, axs = plt.subplots(2, 2, figsize=(14,11))
    fig = plt.figure(figsize=(14,11))
    
    ax0 = fig.add_subplot(2, 2, 1)
    ax0.set_title('Vue du dessus')
    circle = plt.Circle((0,0), volume_radius, color='orange', alpha=0.1)
    ax0.add_patch(circle)
    ax0.plot(sills_xyzR[:,0], sills_xyzR[:,1], 'og', ms=12, alpha=0.2)
    for i in range(n_sills):
        circle = plt.Circle((sills_xyzR[i,0], sills_xyzR[i,1]), sills_xyzR[i,3], color='red', alpha=0.1)
        ax0.add_patch(circle)
        ax0.text(sills_xyzR[i,0], sills_xyzR[i,1], s="%d" % i, horizontalalignment='center', verticalalignment='center')
    ax0.plot(conduits_xyzz[:,2], conduits_xyzz[:,3], 'x', color='grey', alpha=0.5)
    #axs[0,0].axis('equal')
    ax0.set_xlim(xmin, xmax)
    ax0.set_ylim(ymin, ymax)


    ax1 = fig.add_subplot(2, 2, 2)
    ax1.set_title('Vue selon Y')
    ax1.axvline(0, color='grey', lw=1)
    #axs[0,1].set_ylim(zmin, zmax)
    #axs[0,1].invert_yaxis()
    rectangle = plt.Rectangle((volume_depth_bottom, -volume_radius), volume_depth_top-volume_depth_bottom, 2*volume_radius, color='orange', alpha=0.1)
    ax1.add_patch(rectangle)
    ax1.plot(sills_xyzR[:,2], sills_xyzR[:,1], 'og', ms=12, alpha=0.2)
    for i in range(n_sills):
        ax1.plot([sills_xyzR[:,2], sills_xyzR[:,2]], [sills_xyzR[:,1]-sills_xyzR[:,3], sills_xyzR[:,1]+sills_xyzR[:,3]], color='red', alpha=0.1)
        ax1.text(sills_xyzR[i,2], sills_xyzR[i,1], s="%d" % i, horizontalalignment='center', verticalalignment='center')
    #for i in range(n_sills):
    #    circle = plt.Circle((sills_xyzR[i,0], sills_xyzR[i,1]), sills_xyzR[i,3], color='red', alpha=0.1)
    #    axs[0,1].add_patch(circle)
    ax1.plot(conduits_xyzz[:,4], conduits_xyzz[:,3], 'x', color='grey', alpha=0.5)
    ax1.plot(conduits_xyzz[:,4], conduits_xyzz[:,3], 'x', color='grey', alpha=0.5)
    for j, conduit in enumerate(conduits_xyzz):
        ax1.plot([conduit[5], conduit[4]], [conduit[3], conduit[3]], color='blue', alpha=0.5)
        # ax1.text((conduit[4]+conduit[5])/2, conduit[3], s="%s" % list_chars_alpha[j], ha='center', va='center')
    #axs[0,1].axis('equal')
    ax1.set_xlim(zmax, zmin)
    ax1.set_ylim(ymin, ymax)
    #axs[0,1].set_xlim(ymin, ymax)


    ax2 = fig.add_subplot(2, 2, 3)
    ax2.set_title('Vue selon X')
    ax2.axhline(0, color='grey', lw=1)
    #axs[0,1].set_ylim(zmin, zmax)
    #axs[1,0].invert_yaxis()
    rectangle = plt.Rectangle((-volume_radius, volume_depth_bottom), 2*volume_radius, volume_depth_top-volume_depth_bottom, color='orange', alpha=0.1)
    ax2.add_patch(rectangle)
    ax2.plot(sills_xyzR[:,0], sills_xyzR[:,2], 'og', ms=12, alpha=0.2)
    for i in range(n_sills):
        ax2.plot([sills_xyzR[:,0]-sills_xyzR[:,3], sills_xyzR[:,0]+sills_xyzR[:,3]],[sills_xyzR[:,2], sills_xyzR[:,2]], color='red', alpha=0.1)
        ax2.text(sills_xyzR[i,0], sills_xyzR[i,2], s="%d" % i, horizontalalignment='center', verticalalignment='center')
    ax2.plot(conduits_xyzz[:,2], conduits_xyzz[:,4], 'x', color='grey', alpha=0.5)
    ax2.plot(conduits_xyzz[:,2], conduits_xyzz[:,5], 'x', color='grey', alpha=0.5)
    for j, conduit in enumerate(conduits_xyzz):
        ax2.plot([conduit[2], conduit[2]],[conduit[5], conduit[4]], color='blue', alpha=0.5)
        # ax2.text(conduit[2], (conduit[4]+conduit[5])/2, s="%s" % list_chars_alpha[j], va='center', ha='center')
    #axs[1,0].axis('equal')
    ax2.set_ylim(zmin, zmax)
    ax2.set_xlim(xmin, xmax)
    #axs[1,0].set_ylim(ymin, ymax)

    ax3 = fig.add_subplot(2, 2, 4, projection='3d',computed_zorder=False)
    ax3.set_title('Vue 3D')
    for i in range(n_sills):
        ax3.add_patch(matplotlib.patches.Circle((sills_xyzR[i,0], sills_xyzR[i,1]), sills_xyzR[i,3], facecolor="red", alpha=0.3, zorder=-1*sills_xyzR[i,2])).to_3d(z=sills_xyzR[i,2], zdir='z', delta=(0,0,0))
        ax3.plot3D(sills_xyzR[i,0], sills_xyzR[i,1], sills_xyzR[i,2], 'og', ms=12, alpha=0.2)
        ax3.text(sills_xyzR[i,0], sills_xyzR[i,1], sills_xyzR[i,2], s="%d" % i, horizontalalignment='center', verticalalignment='center')
    for conduit in conduits_xyzz:
        ax3.plot3D([conduit[2], conduit[2]], [conduit[3], conduit[3]], [conduit[4], conduit[5]], color='blue', alpha=0.5, zorder=-1*(conduit[4]+ conduit[5])/2)
    ax3.auto_scale_xyz([xmin, xmax], [xmin, xmax], [zmin, zmax])
    #ax3.view_init(elev=1., azim=0.) # Same as Y view
    #ax3.view_init(elev=1., azim=-90.) # Same as X view
    #ax3.view_init(elev=90., azim=-90.) # Same as map view
    ax3.view_init(elev=10., azim=30.) # Oblique view
    ax3.invert_zaxis()

    return fig,[ax0,ax1,ax2,ax3]

def build_graph(sills_xyzR, conduits_xyzz, G_hostrock=2.1e9, nu_hostrock=0.25, mu_melt=1.):
    
    n_sills = sills_xyzR.shape[0]

    G = nx.Graph()

    # 1. ajout des noeuds
    
    gamma_hostrock = 8*(1-nu_hostrock)/(3*np.pi)
    
    for i in range(n_sills):
        G.add_node(i, compressibility = np.pi*(sills_xyzR[i,3]*1e3)**3*gamma_hostrock/G_hostrock)
        
    # 2. ajout des arrêtes
    
    radius_conduits = 1.
    
    for i,j,_,_,zi,zj in conduits_xyzz:
        i,j = int(i),int(j)
        height_conduit = np.abs(zj-zi)*1e3
        G.add_edge(i,j,conductivity = np.pi*radius_conduits**4/(8*mu_melt*height_conduit))

    return G

def barycenter(x1, R1, x2, R2):
    return (R1*x2 + R2*x1)/(R1+R2)

def make_sill_connections(sills_xyzR):
    
    conduit_xyzz = []
    n_sills = sills_xyzR.shape[0]
    
    for i in range(n_sills): # loop over sills, assuming that sills are sorted in decreasing order of depth
        xi, yi, zi, Ri = sills_xyzR[i,:]
        
        for j in range(i+1, n_sills): # loop over other sills above
            xj, yj, zj, Rj = sills_xyzR[j,:]
            
            dij = np.sqrt((xj-xi)**2+(yj-yi)**2) # distance between sill centers
            if dij < Ri + Rj: # the two sills have an intersection
                xa, ya = barycenter(np.array([xi,yi]), Ri, np.array([xj,yj]), Rj) # coordinates of connection between the sills
                
                test_intersect = False
                for k in range(i+1, j): # check if connection intersects a third sill
                    xk, yk, zk, Rk = sills_xyzR[k,:]
                    dka = np.sqrt((xk-xa)**2+(yk-ya)**2) # distance between connection and third sill
                    #print("  Check: i=%d (%.2f) ; j=%d (%.2f) ; k=%d (%.2f)" % (i,Ri,j,Rj,k,Rk))
                    if dka < Rk: # connection intersects the third sill
                        test_intersect = True # intersection: ignore connection
                        break
                    
                if not(test_intersect): # no intersection: append connection
                    conduit_xyzz.append([i, j, xa, ya, zi, zj])
                    
    return np.array(conduit_xyzz)

def get_compressibility(G,i):
    """Retourne la compressibilité de la chambre i"""
    return G.nodes[i]["compressibility"]

def get_conductivity(G,i,j):
    """Retourne la conductivité du conduit entre i et j"""
    try :
        return G.get_edge_data(i,j)["conductivity"]
    except:
        return 0.

def build_state_space_matrix_free(G):
    """ 
    Construit à partir de la description du système sous forme de graph la matrice A telle que dx/dt = Ax
    """
    
    N = len(G)
    A = np.zeros((N,N))
    
    for i in range(N):
        for j in range(N):
            # Terme en i,i
            if j == i: 
                A[i,i] = - sum([get_conductivity(G,i,k) for k in range(N+1) if k != i]) / get_compressibility(G,i)
            # Terme en i,j (j!=i)
            else : 
                A[i,j] = get_conductivity(G,i,j)/get_compressibility(G,i)
        
    return A 

def build_state_space_matrix_forced(G,sources):
    """ 
    Construit à partir de la description du système sous forme de graph G les matrices A et B telles que dx/dt = Ax + Bu
    Avec u les termes de forçage 
    sources : vecteur des connectivités au terme source 
    """
    
    N = len(G)
    
    # Matrice A
    A = build_state_space_matrix_free(G)
    # Ajout du terme source 
    for i in range(N):
        A[i,i] -= sources[i]/get_compressibility(G,i)
        
    # Matrice B 
    B = np.zeros((N,N))
    B = np.diag(sources/np.array([get_compressibility(G,i) for i in range(N)]))
    
    return A,B

def exponential_compliance_time(A, t):
    n_nodes = A.shape[0]
    # eigendecomposition of A
    D, T = np.linalg.eig(A) # /!\ we need to use eig (instead of eigh) because A is no longer symmetric /!\
    tol = 1e-9
    D[np.where(np.abs(D)<tol)]=0.
    # compute matrix exponential using : e^{At}=T*e^{Dt}*T^{-1}
    Tinv = np.linalg.inv(T)
    eDt = np.exp(np.array((t[:][:, np.newaxis, np.newaxis]*D), dtype=float))*np.eye(n_nodes) # use broadcasting
    eDtTinv = np.dot(eDt, Tinv)
    eAt = np.dot(eDtTinv.transpose((0,2,1)),(T.T)).transpose((0,2,1)) 
    return eAt

def solve_analytically_free(G, p0, time_):
    
    n_sills = len(G)
    
    A = build_state_space_matrix_free(G)
    
    # Résolution p(t)
    eAt = exponential_compliance_time(A, time_)
    p = np.dot(eAt,p0.T).reshape(-1, n_sills)
    
    # Calcul des pressions à t_inf
    D, T = np.linalg.eig(A) # /!\ now we need to use eig (instead of eigh) because A is no longer symmetric /!\
    Tinv = np.linalg.inv(T)
    
    # Index corresponding to null eigenvalue
    index_null_eigenvalue = np.where(np.abs(D)<1e-14)[0]
    
    # e^{Dt} for t -> infty should have zeros everywhere, except for the null eigenvalue
    eDtinf = np.zeros((n_sills, n_sills))
    eDtinf[index_null_eigenvalue, index_null_eigenvalue] = 1
    
    # Compute pressure accordingly
    p_inf = np.dot( np.dot(T, np.dot(eDtinf, Tinv)), p0.T).reshape(n_sills)
    
    return p,p_inf
    
def solve_analytically_forced(G, p0, li, p_source, time_):
    
    n_sills = len(G)
    
    A,B = build_state_space_matrix_forced(G,li)
    
    p_sources = np.array([1 if li[i]!=0 else 0 for i in range(len(G))])*p_source
    u = p_sources
    
    ### Time step
    h = time_[1]-time_[0]
    k_stop = int(((time_[-1]-time_[0])/h)+2)
    k = np.array(range(time_.shape[0]))

    #### Build matrix $A_d=e^{Ah}$
    Ad = exponential_compliance_time(A, np.array([h])).reshape(n_sills,n_sills) # = e^{hA}
    
    #### Build matrix $B_d=A^{-1}.(A_d-I).B$ 
    Ainv = np.linalg.inv(A) # we will need this one too
    Bd = np.dot(np.dot(Ainv, (Ad - np.eye(A.shape[0]))), B) # = A^{-1}*(e^{hA}-I)*B
    
    #### Iterative solution: $x_d(k+1)=A_d.x_d(k)+B_d.u_d(k)$
    
    xditer = np.zeros((len(k), A.shape[0]))
    
    # Initilisation
    xditer[0,:] = p0
    
    # Itération
    for _k in k[1:]:
        xditer[_k,:] = (np.dot(Ad, xditer[_k-1,:]).T + np.dot(Bd, u)).reshape(-1)
    
    p = xditer
    return p 