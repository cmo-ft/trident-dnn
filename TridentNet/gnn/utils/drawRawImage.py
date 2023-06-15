import numpy as np
import pandas as pd
import torch
import argparse
import math
# For plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        return np.min(zs)



def plot_track_3d(hits: pd.DataFrame, primary):
    """
    `hits: pd.DataFrame`
        Contains following columns:
        - "DomId": dom id
        - "x0", "y0", "z0": position of DOM being hit
        - "t0": time of hits, can be earlist time or average time
    
    `paras_true: np.ndarray`
        The simulation truth of the primary particle. It is composed as [x0, y0, z0, px, py, pz]
    """    
    # Telescope size and edges
    xmin, xmax = 0, 400
    ymin, ymax = 0, 400
    zmin, zmax = 0, 60
    # xmin, xmax = 0, 120
    # ymin, ymax = 0, 120
    # zmin, zmax = 0, 48
    space = 0
    
    # fig = plt.figure(figsize=(12, 9), dpi=200)
    fig = plt.figure(dpi=200)
    # fig.add_axes([left, bottom, width, height])
    ax = fig.add_axes([0.07, 0.2, 0.9, 0.75], projection='3d')
    # ax.set_box_aspect(aspect = (1500./160, 1500./310, 600./20.))   
    # primary[0] = primary[0] * 160./1500
    # primary[1] = primary[1] * 310./1500
    # primary[2] = primary[2] * 1500./600
    # primary /= math.sqrt((primary*primary).sum())
    print(primary)

    # Draw primary particles
    length = np.linspace(-100, 100, 10)
    # firstHit = hits.sort_values('t1st').iloc[0]
    firstHit = hits.sort_values('nhits').iloc[-1]
    trackX = firstHit['idX'] + primary[0] * length
    trackY = firstHit['idY'] + primary[1] * length
    trackZ = firstHit['idZ'] + primary[2] * length
    # print(f"trackX:{trackX}\ntrackY:{trackY}\ntrackZ:{trackZ}")
    ax.plot(trackX, trackY, trackZ, '-', c='xkcd:steel blue')
    arrow_prop_dict = dict(
        mutation_scale=30, arrowstyle='-|>', shrinkA=0, shrinkB=0)
    arrow = Arrow3D([trackX[0], trackX[-1]], [trackY[0], trackY[-1]], [trackZ[0], trackZ[-1]],
                    **arrow_prop_dict, color='steelblue')
    ax.add_artist(arrow)

    # Modify domhit into hits
    hits['tmean'] = np.log(hits['tmean'] + 1)
    ax.scatter(hits["idX"], hits["idY"], hits["idZ"],
               s=np.power(hits["nhits"], .7) * 6.,
               c=hits["tmean"],
               marker='.',
               alpha=1.,
               cmap='rainbow_r')
    # config axis
    ax.set_xlim(xmin-space, xmax+space)
    ax.set_ylim(ymin-space, ymax+space)
    ax.set_zlim(zmin-space, zmax+space)
    # ax.set_xticks(np.linspace(xmin, xmax, 6))
    # ax.set_yticks(np.linspace(ymin, ymax, 6))
    # ax.set_zticks(np.linspace(zmin, zmax, 6))
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.set_zlabel("Z coordinate")
    # ax.grid(False)
    
    # axes title
    # if title:
    #     ax.set_title(title)


    # plot color bar
    cmap_ = plt.get_cmap('rainbow_r')
    # fig.add_axes([left, bottom, width, height])
    cax = fig.add_axes([0.2, 0.1, 0.6, 0.05])
    length = int(len(hits) / 20)
    hits.sort_values("tmean", inplace=True)
    minT = hits["tmean"].iloc[length]
    maxT = hits["tmean"].iloc[-length-1]
    norm = mpl.colors.Normalize(vmin=minT, vmax=maxT)
    cb1 = mpl.colorbar.ColorbarBase(cax, cmap=cmap_,
                                    norm=norm,
                                    orientation='horizontal')
    cax.set_xlabel('Time [ns]', size=12)
    plt.show()
    # plt.savefig("./rawImage.png")



if __name__ == '__main__':
    # Input arguments
    parser = argparse.ArgumentParser(description="Draw 3D raw image by Cen Mo")
    parser.add_argument('--sliceId', type=int, default=0, help='Data slice id')
    parser.add_argument('--entryId', type=int, default=0, help='Entry id of file')
    args = parser.parse_args()

    dataDir = "/lustre/collider/mocen/project/hailing/machineLearning/data/signal_16Jan2023/xyz_view/"
    slicePrefix="xyz_"
    slicePath = f"{dataDir}{slicePrefix}{args.sliceId}.csv"
    data = pd.read_csv(slicePath).set_index('id')
    primary = np.array(torch.load(f"{dataDir}drct_{args.sliceId}.pt"))
    plot_track_3d(data.loc[[args.entryId]], primary[args.entryId])

