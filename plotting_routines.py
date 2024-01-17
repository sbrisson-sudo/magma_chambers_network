import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib 
from mpl_toolkits.mplot3d import art3d

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
    fig.suptitle('Sills and conduits')

    ax0 = fig.add_subplot(2, 2, 1)
    ax0.set_title('Map view')
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
    ax1.set_title('Y view')
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
    ax2.set_title('X view')
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
    ax3.set_title('3D view')
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