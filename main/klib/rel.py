# -*- coding: utf-8 -*-

import numpy as np
import copy, pdb
from collections import defaultdict


class Rel(object):
    """ reliability class.
        In this class, we can evaluate the joints' reliability 
        according to their behavior( spatio & temporal), kinemetic
        ( physical) and tacking( Kinect) feature.
    """
    def __init__(self):
        """initialize parameters
        """
        # kinematic segment length (unit:cm)
        self.kinseg    = {}
        self.kinseg[0] = 13.4   # head2neck
        self.kinseg[1] = 8.3    # neck2spins
        self.kinseg[2] = 15.4   # spins2spinm
        self.kinseg[3] = 32.5   # spinm2spinb
        self.kinseg[4] = 16.65  # spins2shlder
        self.kinseg[5] = 33.2   # shlder2elbow
        self.kinseg[6] = 27.1   # elbow2wrist

        # target joint order
        self.trg_jorder = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 20]
        self.jointary   = defaultdict(list)  # joint array

        self.Rb = defaultdict(list)
        self.Rt = defaultdict(list)
        self.Rk = defaultdict(list)
        # gaussian parameter
        sigma = 0.65
        self.gw = 1/(2*np.pi)**2/sigma*np.exp(-0.5*np.arange(2)**2/sigma**2)
        self.gw = self.gw*(1/sum(self.gw))

    def rel_behav(self, joint, th=0.03, fsize=3):  # behavior term
        """ according to the joint's position in frame t, t-1 and t-2
            calculate the behavior reliability term
            joint : 3D joint position in [..., f-4, f-3, f-2, f-1, f]
            th   : threshold (uint: m)
        """
        r = 1
        if len(joint) >= fsize:
            for k in xrange(1):
                dist2   = joint[-(k+1)]-joint[-(k+2)]
                dist1   = joint[-(k+1)]-joint[-(k+3)]
                n_dist2 = np.linalg.norm(dist2)
                n_dist1 = np.linalg.norm(dist1)

                if (n_dist1 < th):
                    r = 1
                else:
                    if (n_dist2 > th):
                        r = max(1-4*(n_dist2-th)/th, 0)
                    else:
                        r = 1
        return r

    def rel_kin(self, joints):  # kinematic term
        """ according to the segment length of each joint pair
            calculate the kinematic reliability term
        """
        order1 = [9, 5, 20, 1, 2]
        order2 = [8, 6, 4, 20, 3]  # joints' order
        order3 = [10, 4, 8, 0, 20]
        refer1 = [5, 6, 4, 2, 0]   # kinseg's order
        refer2 = [6, 5, 4, 3, 1]

        segrel = defaultdict(lambda: int(0))
        result = []
        cnts = np.zeros(21)

        for i in xrange(len(order1)):
            A = np.array([joints[order1[i]].Position.x, joints[order1[i]].Position.y, joints[order1[i]].Position.z])
            B = np.array([joints[order2[i]].Position.x, joints[order2[i]].Position.y, joints[order2[i]].Position.z])
            C = np.array([joints[order3[i]].Position.x, joints[order3[i]].Position.y, joints[order3[i]].Position.z])

            tmp = min(np.abs(np.linalg.norm(A-B)*100-self.kinseg[refer1[i]])/self.kinseg[refer1[i]], 1)
            segrel[order1[i]] += tmp
            segrel[order2[i]] += tmp

            tmp = min(np.abs(np.linalg.norm(A-C)*100-self.kinseg[refer2[i]])/self.kinseg[refer2[i]], 1)
            segrel[order1[i]] += tmp
            segrel[order3[i]] += tmp

            cnts[order1[i]] += 2
            cnts[order2[i]] += 1
            cnts[order3[i]] += 1

        for i in self.trg_jorder:
            result.append(1-(segrel[i]/cnts[i]))

        return result

    def rel_trk(self, joints):  # tracking term
        """ Kinect sensor's tracking state of each joint
        """
        trkrel = []
        for i in self.trg_jorder:
            if joints[i].TrackingState == 2:
                trkrel.append(1.0)
            elif joints[i].TrackingState == 1:
                trkrel.append(1.0)
            else:
                trkrel.append(0.0)

        return trkrel

    def rel_overall(self, Rb, Rk, Rt, order, flen=2):
        """combine the behavior, kinematic and tracking reliability
           calculate overall reliability score
        """
        Relary = np.zeros(21)
        Rel = defaultdict(int)
        if (len(Rb[0]) >= flen) & (len(Rk[0]) >= flen) & (len(Rt[0]) >= flen):
            if order == self.trg_jorder:
                for j in order:
                    for i in xrange(flen):
                        Rel[j] += self.gw[i]*min(Rb[j][-(i+1)], Rk[j][-(i+1)], Rt[j][-(i+1)])
                        Relary[j] += self.gw[i]*min(Rb[j][-(i+1)], Rk[j][-(i+1)], Rt[j][-(i+1)])
            else:
                raise ImportError('joints order not match !!')
        else:
            return Rel, np.array([])
        return Rel, Relary

    def run(self, jdic, order):
        """calculate joints' relability for each frame
        """
        rt = self.rel_trk(jdic)
        rk = self.rel_kin(jdic)
        for jj, ii in enumerate(self.trg_jorder):
            self.jointary[ii].append(np.array([jdic[ii].Position.x, jdic[ii].Position.y, jdic[ii].Position.z]))

            self.Rb[ii].append(self.rel_behav(self.jointary[ii]))
            self.Rt[ii].append(rt[jj])
            self.Rk[ii].append(rk[jj])

        return self.rel_overall(self.Rb, self.Rk, self.Rt, order)
