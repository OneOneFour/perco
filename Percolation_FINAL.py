# coding=utf-8
import pylab as pl
import numpy.random as npr
import numpy as np
import time
import sys
import scipy.stats as sps
from mpl_toolkits.mplot3d import Axes3D
'''
Inital setup,
Ask user for inital input of number of trials,radius to simulate and whether
to run the simulation in 3D
'''

trials = int(raw_input("enter number of trials "))
r= float(raw_input("enter disc radius "))
runthreedee = raw_input("run in 3D? (t/f) ")
threeDee = False
if runthreedee.lower() == "t":
    threeDee = True
startTime = time.time()


class Cluster:
    '''
    Cluster Class,
    Stores a copy of the indices of the disks in the discPos array in the main loop
    that form the cluster.

    Also stores the current maximum and minimum bounds of the cluster. All maximum
    bounds are initalized with 0 so that any disk added to the array (as  0<x,y,z<1)
    will overwrite the default value. Vice versa for minimum bounds.

    Class initalized by creating the empty array of disks as well as by passing
    the value for the radius of the disks that will form the cluster

    Function hasDiscs returns True if a disc is present in the clusters disk array,
    false if not

    Function checkIfComplete returns true if the cluster is touching an edge

    Function addDisc takes a indice of a disk in diskPos as well as the discPos
    array and adds it to the clusters disk array as well as checking if the
    disk moves the boundaries of the cluster closer to an edge.

    Function joinCluster takes another cluster as an argument and copies that
    clusters disc array to the this clusters personal array
    the function also checks if adding these disks expand
    the boundaries of the cluster
    '''
    __discs = None
    maxx = 0
    minx = 1
    maxy = 0
    miny = 1
    minz = 1
    maxz = 0
    __radius = 0
    def __init__(self,radius):
        self.__discs = []
        self.__radius = radius
    def hasDisc(self, disc):
        return disc in self.__discs

    def checkIfComplete(self):
        return (self.maxx >= 1 - self.__radius and self.minx <= self.__radius)\
               or (self.maxy >= 1 - self.__radius and self.miny <= self.__radius)\
               or(self.maxz >= 1 - self.__radius and self.minz <= self.__radius)

    def addDiscs(self, disc,discPos):
        self.__discs.append(disc)
        if discPos[disc,0] > self.maxx:
            self.maxx = discPos[disc,0]
        if discPos[disc,1]> self.maxy:
            self.maxy = discPos[disc,1]
        if discPos[disc,0] < self.minx:
            self.minx = discPos[disc,0]
        if discPos[disc,1] < self.miny:
            self.miny = discPos[disc, 1]
        if threeDee:
            if discPos[disc,2] < self.minz:
                self.minz = discPos[disc,2]
            if discPos[disc,2] > self.maxz:
                self.maxz = discPos[disc,2]
    def getDiscs(self):
        return self.__discs
    def joinCluster(self,cluster):
        self.__discs.extend(cluster.getDiscs())
        if self.maxx < cluster.maxx:
            self.maxx = cluster.maxx
        if self.minx > cluster.minx:
            self.minx = cluster.minx
        if self.maxy < cluster.maxy:
            self.maxy = cluster.maxy
        if self.miny > cluster.miny:
            self.miny = cluster.miny
        if threeDee:
            if self.maxz < cluster.maxz:
                self.maxz = cluster.maxz
            if self.minz > cluster.minz:
                self.minz = cluster.minz
'''
Function getOverlap takes a position, an array of positions to check against and the
radius of the disk. The function finds if the position inputted using pos has any
overlap with any other disk in arr.If there are overlaps, (seperation distance less
than 2r), then the function returns the indices in arr where the overlaps are.
'''
def getOverlap(pos, arr,radius):
    if len(arr) < 1:
        return []
    distance = (pos[0]- arr[:,0]) ** 2 + (pos[1] - arr[:,1]) ** 2
    if threeDee:
        distance += (pos[2] - arr[:,2])**2
    return np.where((distance <= 4 * radius**2))
'''
Function takes a disc and the array of clusters and determines whether the disk is
part of a cluster which spans the area If so, its color is set to be red
Else its colour is blue
'''
def getDiscColour(disc, clusters):
    for cluster in clusters:
        if cluster.hasDisc(disc) and cluster.checkIfComplete():
            return 'r'
    return 'b'
'''
Function takes in the array of clusters and checks if any are completed
Returns True if any are completed
False otherwise
'''
def getIfComplete(clusters):
    for c in clusters:
        if c.checkIfComplete():
            return True
    return False
'''Function takes in a single disc and the cluster array and returns the
cluster if one exists that the disc is contained in Returns cluster
that disc is in if it exists
Else Returns None
'''
def getCluster(disc, clusters):
    for cluster in clusters:
        if cluster.hasDisc(disc):
            return cluster
    return None

#function draws the discs that form percolating system.
def plotDiscPos(discPos, clusters,radius,j):
    '''
    Function creates a square 6 inch by 6 inch figure.
    Depending on whether the program is in 2D or 3D mode, the function will
    either draw a spherical surface using spherical polar coordinates or it will
    draw circles of radius, "radius". The function calls the getDiscColour
    function in order to work out whether a specified disk is in a completed
    cluster or not so that it can be coloured accordingly.In both cases the
    axes are manually set to stretch from 1 to 0 on all relevant coordinate
    axis to ensure shapes appear correctly.
    The function also takes in "j", the current disc density of the system
    to be drawn which is displayed in the top right hand corner.
    '''
    f = pl.figure(figsize=(6, 6))
    if threeDee:
        ax = f.add_subplot(111, projection='3d')
        ax.set_xlim3d(0, 1)
        ax.set_ylim3d(0, 1)
        ax.set_zlim3d(0, 1)
        for i in range(len(discPos)):
            phi, theta = np.mgrid[0.0:2 * np.pi:50j, 0.0:np.pi:50j]
            x = (radius * np.sin(theta) * np.cos(phi)) + discPos[i, 0]
            y = (radius * np.sin(theta) * np.sin(phi)) + discPos[i, 1]
            z = (radius * np.cos(theta)) + discPos[i, 2]
            ax.plot_surface(x, y, z, color=getDiscColour(i, clusters), alpha=0.5)
    else:
        pl.axis([0, 1, 0, 1])
        for i in range(len(discPos)):
            circ = pl.Circle(discPos[i],
                             radius=radius,
                             color = getDiscColour(i,clusters)
                             ,alpha =0.5)
            pl.gca().add_artist(circ)

    pl.gca().set_xlabel("x")
    pl.gca().set_ylabel("y")
    pl.gca().annotate(j,xy=(1,1))
    pl.show()


def runIter(n,r,show=False):
    clusters = []
    '''
    Generate all the random numbers at the start of the loop as resizing
    numpy arrays is a slow operation. By using the varible i to track
    the current number of discs, discPos[:i] will give the positions
    of discs that have already been spawned
    '''
    i = 0
    if threeDee:
        discPos = npr.uniform(size=(n,3))
    else:
        discPos = npr.uniform(size=(n,2))
    while i < n and not getIfComplete(clusters):
        '''
        Check for collisions using the new disc to be created at
        discPos[i], the discs that already exist in discPos[:i],
        and the fixed radius for this iteration.

        Then create a new cluster, append to the cluster array and
        add the new position which is at the "i" index in the discPos
        array
        '''
        colDisc = getOverlap(discPos[i],discPos[:i],r)
        newCluster = Cluster(r)
        clusters.append(newCluster)
        newCluster.addDiscs(i,discPos)
        '''
        if there are any disks in the colDisc varible, which contains
        the indices in discPos of discs which overlap with the disc at
        position "i":
        check if that disk is in a cluster and if so join the disk and
        its cluster to newCluster ensuring to remove the old cluster
        from the clusters array.
        '''
        if len(colDisc) > 0:
            for disc in colDisc[0]:
                tmpCluster = getCluster(disc,clusters)
                if tmpCluster is not None and tmpCluster is not newCluster:
                    clusters.remove(tmpCluster)
                    newCluster.joinCluster(tmpCluster)
                else:
                    newCluster.addDiscs(disc,discPos)
        i += 1
        #plotDiscPos(discPos[:i],clusters,r,i)
        #uncomment the line above to draw each time a disc is added
    if getIfComplete(clusters):
        #plotDiscPos(discPos[:i],clusters,r,i)
        #uncomment above line if you want to draw succesful runs only
        return True,i
    else:
        return False,

def runRadiusIter(n,r,count):
    x=[]#array storing succesful number disk density that cause percolation.
    startTime = time.time()
    for i in range(count):
        a = runIter(n,r)
        if len(a) > 1:
            x.append(a[1])
        '''
        if runIter returns a tuple with a length greater than 1, then it returned
        successful, the 2nd element is the disk count that enabled percolation
        '''

        print '\r',i, " out of ",count, " iterations complete",
        sys.stdout.flush()
    duration = time.time() - startTime
    #caculate time to run iteration, needed to draw time against r plot.
    return x,duration
'''
runNIter simpler versino of runRadiusIter except only checks if runIter
returns true and if so adds 1 to a counter variable.
'''
def runNIter(n,r,count):
    x=0
    for i in range(count):
        if runIter(n,r)[0]:
            x+=1
    return x
def getPercolationConstant(r):#gets percolation constant at a fixed r
    '''
    Run number of iterations of runIter equal to the number of trials defined by
    the user input at the start of the program. Allow for the system to simulate
    up to 10,000 discs in each iteration.
    '''
    val = np.array(runRadiusIter(10000,r,trials)[0])
    pl.hist(val,np.amax(val) - np.amin(val))
    pl.gca().set_xlabel("Disc number density (N)")
    pl.gca().set_ylabel("Frequency")
    pl.show()
    #calculate percolation threshold correctly depending on 2D/3D
    if threeDee:
        print "eta_c = ",np.mean(val,dtype=np.float64) * np.pi * (r**3)\
                       * (4/3) ,"plus-minus",sps.sem(val*np.pi*(r**3)*(4/3))
    else:
        print "eta_c = ",np.mean(val,dtype=np.float64) * np.pi * (r**2)\
            ,"plus-minus",sps.sem(val*np.pi*(r**2))

#get percolation constant for a range of r, produces sigmoid curve
def getPercolationConstantN(nMin,nTo,nStep,r):
    '''
    Create variable nRange of each maximum n value that will be fed into runNIter.
    This then for each value of n in nRange, trials number of repeats and returns
    the number of succesful percolations with the specified n value. The get
    probability of successful percolation divide this result by trials.
    '''
    nRange = np.arange(nMin,nTo,nStep)
    val = [float(runNIter(n,r,trials))/float(trials) for n in nRange]
    pl.figure()
    pl.plot(nRange,val,"b-")
    pl.gca().set_xlabel("Disc number density (N)")
    pl.gca().set_ylabel("Probability of forming connection")
    pl.show()
'''
Function takes a range of r and shows how the percolation threshold
is affected by this r.
'''
def getPercolationConstantR(rMin,rTo,rStep):
    '''
    rRange is an array of radius values from rMin to rTo going up in steps of rStep.
    For each value of r, run trials iterations of runIter to gain an array of number
    disk density that cause percolation. runRadiusIter also returns the time it took
    for all the trials at a specified r to complete as the 2nd element in the returned
    tuple. Both the time taken and the calculate percolation threshold can then be
    plotted against the radius.
    '''
    rRange = np.arange(rMin,rTo,rStep)
    v1 = []
    v2 = []
    for r in rRange:
        result = runRadiusIter(10000,r,trials)
        v1.append(np.mean(result[0])*np.pi*(r**2))
        v2.append(result[1])
    fig,ax1 = pl.subplots()
    ax1.plot(rRange,v2,"b-")
    ax1.set_xlabel("Radius of disk")
    ax1.set_ylabel("Time (s)",color = "b")
    ax1.tick_params('y', colors='b')

    ax2 = ax1.twinx()
    ax2.plot(rRange,v1,"r-")
    ax2.set_ylabel("Percolation Threshold",color = "r")
    ax2.tick_params('y',colors="r")

    fig.tight_layout()
    pl.show()

getPercolationConstant(r)
print "Time taken for program ", time.time() - startTime," seconds"