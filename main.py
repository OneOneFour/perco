import pylab as pl
import numpy.random as npr
import numpy as np
import time
from mpl_toolkits.mplot3d import Axes3D
radius = 0.1
startTime = time.time()


class Cluster:
    __discs = None
    __discPos = None
    maxx = 0
    minx = 1
    maxy = 0
    miny = 1
    minz = 1
    maxz = 0

    def __init__(self, discPos):
        self.__discPos = discPos
        self.__discs = []

    def hasDisc(self, disc):
        return disc in self.__discs

    def checkIfComplete(self):
        return (self.maxx >= 1 - radius and self.minx <= radius) or (self.maxy >= 1 - radius and self.miny <= radius) or (self.maxz >= 1-radius and self.minz <= radius)

    def addDiscs(self, discs):
        self.__discs.extend(discs)
        if (self.__discPos[discs][:, 0] > self.maxx).any():
            self.maxx = np.amax(self.__discPos[discs][:, 0])
        if (self.__discPos[discs][:, 1] > self.maxy).any():
            self.maxy = np.amax(self.__discPos[discs][:, 1])
        if (self.__discPos[discs][:, 0] < self.minx).any():
            self.minx = np.amin(self.__discPos[discs][:, 0])
        if (self.__discPos[discs][:, 1] < self.miny).any():
            self.miny = np.amin(self.__discPos[discs][:, 1])
        if (self.__discPos[discs][:,2] < self.minz).any():
            self.minz = np.amin(self.__discPos[discs][:,2])
        if (self.__discPos[discs][:,2] > self.maxz).any():
            self.maxz = np.amax(self.__discPos[discs][:,2])
        # scan the added discs to expand cluster
        for d in discs:
            nextScan = getOverlap(self.__discPos[d], self.__discPos)[0].tolist()
            approvedScan = []
            for point in nextScan:
                if point not in self.__discs:
                    approvedScan.append(point)
            if len(approvedScan) < 1:
                continue
            self.addDiscs(np.asarray(approvedScan))


def getOverlap(pos, arr):
    distance = np.sqrt((pos[0] - arr[:, 0]) ** 2 + (pos[1] - arr[:, 1]) ** 2 + (pos[2] - arr[:,2])**2)
    return np.where(distance <= 2 * radius)


def getDiscColor(disc, clusters):
    for cluster in clusters:
        if cluster.hasDisc(disc) and cluster.checkIfComplete():
            return 'r'
    return 'b'


def getNext(discPos, clusters):
    for i in range(len(discPos)):
        if getCluster(i, clusters) is None:
            return discPos[i]
    return None


def getCluster(disc, clusters):
    for cluster in clusters:
        if cluster.hasDisc(disc):
            return cluster
    return None


def plotDiscPos(discPos, clusters):
    f = pl.figure(figsize=(6, 6))
    ax = f.add_subplot(111,projection = '3d') 
    #pl.axis([0, 1, 0, 1]
    ax.set_xlim3d(0,1)
    ax.set_ylim3d(0,1)
    ax.set_zlim3d(0,1)
    for i in range(len(discPos)):
        phi,theta = np.mgrid[0.0:2*np.pi:50j,0.0:np.pi:50j]
        x = (radius * np.sin(theta) * np.cos(phi) )+ discPos[i,0]
        y = (radius * np.sin(theta) * np.sin(phi) )+ discPos[i,1]
        z = (radius * np.cos(theta))+ discPos[i,2]
        ax.plot_surface(x,y,z,color=getDiscColor(i,clusters),alpha = 0.5)
    pl.show()


def runIter(n, show=False):
    clusters = []
    discPos = npr.uniform(size=(n, 3))
    while getNext(discPos, clusters) is not None:
        scan = getOverlap(getNext(discPos, clusters), discPos)[0]
        scCluster = Cluster(discPos)
        clusters.append(scCluster)
        scCluster.addDiscs(scan)
        for cluster in clusters:
            if cluster.checkIfComplete():
                if show:
                    plotDiscPos(discPos, clusters)
                return True
    if show:
        plotDiscPos(discPos, clusters)
    return False


def runfor(n, count):
    x = 0
    show = False
    for i in range(count):
        if i == 0 and n % 10 == 0:
            show = False
        else:
            show = False
        if runIter(n,show):
            x += 1
    return x
def getThreshold(val):
    for i in range(len(val)):
        if val[i] >= 50:
            return (4/3) * (i*5 + 20) * np.pi * (radius**3)


nPos = np.arange(20,150,1)
val = [runfor(n,1000) for n in nPos]
print "Percolation threshold = ",getThreshold(val)
pl.plot(nPos,val,"b-")
pl.show()