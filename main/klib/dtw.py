# -*- coding: utf-8 -*-
from scipy.spatial.distance import _validate_vector
from scipy.ndimage.filters import gaussian_filter1d as gf
from scipy.ndimage.filters import gaussian_filter as gf_2D
from scipy.linalg import norm
from collections import defaultdict
from w_fastdtw import fastdtw
import numpy as np
from scipy.signal import argrelextrema
from scipy import signal
from dataoutput import Dataoutput
import matplotlib.pyplot as plt
import pdb


class Dtw(object):
    """ Dynamic time warping class.
        in this clas we can basically find several subsequences
        from the total sequence.
        Way to extend the exercise database: 1. in initial function,
        add the exercise order ie order[M][N]. Here M represents the
        new exercise and N is equal to 2+ #segments in your exercise.
        2. setting the weight for each joint's coordinates
    """
    def __init__(self):
        """ initailize the order and weight for each exercise
            initailize dtw parameters
        """
        # dtw parameters initialization
        self.io          = Dataoutput()
        self._done       = False
        self.do_once     = False
        self.decTh       = 1800
        self.cnt         = 0
        self.distp_prev  = 0
        self.distp_cmp   = np.inf
        self.oidx        = 0  # initail
        self.gt_idx      = 0
        self.Ani_idx     = 0
        self.presv_size  = 0
        self.idxlist     = []
        self.idx_cmp     = 0
        self.fcnt        = 0
        # self.finishcnt   = 0
        self.srchfw      = 10  # forward search range
        self.srchbw      = 20  # backward search range
        self.error       = []
        # exe1 parameters
        self.cntdown     = 90
        # exe2 parameters
        # self.hcnt        = 0
        self.hstate      = np.array([])
        self.rawhstate   = np.array([0,0])
        # self.hstate_cnt  = np.array([0,0])
        # self.hopen_list  = []
        # self.hclose_list = []
        self.holdstate   = True
        self.holdlist    = np.array([])
        self.ref_dmap    = None
        self.ref_bdry    = np.array([])
        self.breath_list = []
        # updatable parameters
        self.dpfirst     = {}
        self.dist_p      = {}
        self.deflag_mul  = defaultdict(lambda: (bool(False)))
        self.seqlist     = np.array([])
        self.seqlist_reg = np.array([])
        self.seqlist_gf  = np.array([])
        self.dcnt        = 0
        self.chk_flag    = False
        self.deflag      = False  # decreasing flag
        self.onedeflag   = False
        self.segini      = True
        self.evalstr     = ''
        #self.segend      = False
        # exercise order
        self.order = defaultdict(dict)
        # exercise 1
        self.order[1][0] = [1]
        self.order[1][1] = [2]
        self.order[1][2] = 'end'          
        # exercise 2
        self.order[2][0] = [1]
        self.order[2][1] = [2]
        self.order[2][2] = 'end'        
        # exercise 3
        self.order[3][0] = [1]
        self.order[3][1] = [3]
        self.order[3][2] = 'end'
        self.order[3][3] = [4]
        self.order[3][4] = [2, 3]
        self.order[3][5] = 'end'
        # exercise 4
        self.order[4][0] = [1]
        self.order[4][1] = [3]
        self.order[4][2] = 'end'
        self.order[4][3] = [4]
        self.order[4][4] = [2, 3]
        self.order[4][5] = 'end'
        # weight
        self.jweight = {}
        self.jweight[1] = np.array([0., 0., 0., 3., 3., 3., 9., 9., 9.,
                                    0., 0., 0., 3., 3., 3., 9., 9., 9.,
                                    0., 0., 0.])        
        self.jweight[2] = np.array([0., 0., 0., 3., 3., 3., 9., 9., 9.,
                                    0., 0., 0., 3., 3., 3., 9., 9., 9.,
                                    0., 0., 0.])
        self.jweight[3] = np.array([0., 0., 0., 9., 9., 9., 9., 9., 9.,
                                    0., 0., 0., 9., 9., 9., 9., 9., 9.,
                                    0., 0., 0.])
        self.jweight[4] = np.array([0., 0., 0., 3., 3., 3., 9., 9., 9.,
                                    0., 0., 0., 3., 3., 3., 9., 9., 9.,
                                    0., 0., 0.])

        for ii in self.jweight.keys():
                self.jweight[ii] = self.jweight[ii]/sum(self.jweight[ii])*1.5

    def wt_euclidean(self, u, v, w):
        """ normal euclidean dist with the weighting
        """
        u = _validate_vector(u)
        v = _validate_vector(v)
        dist = norm(w*(u-v))
        return dist

    def clip(self, seqlist, exeno):
        """ try find the subsequence from current sequence
        """
        tgrad = 0
        for ii in [3, 4, 5, 6, 7, 8, 12, 13, 14, 15, 16, 17]:
            tgrad += (np.gradient(gf(seqlist[:, ii], 1))**2)*self.jweight[exeno][ii]
        tgrad = tgrad**0.5
        lcalminm = argrelextrema(tgrad, np.less, order=5)[0]
        foo = np.where(((tgrad < 1)*1) == 0)[0]
        if (len(foo) == 0) | (len(lcalminm) == []):
            return []
        else:
            lb = max(foo[0], 50)
            minm = []
            for ii in lcalminm[lcalminm > lb]:
                if tgrad[ii] < 1:
                    minm.append(ii)
            return minm

    def seg_update(self, endidx):
        """ update the dictionary content then reset the parameters
        """
        self.seqlist_reg = self.seqlist_reg[endidx+1:, :]  # update the seqlist
        self.presv_size  = self.seqlist_reg.shape[0]
        self.oidx        = self.gt_idx
        self.deflag_mul  = defaultdict(lambda: (bool(False)))
        self.cnt         = 0
        self.dpfirst     = {}
        self.dist_p      = {}
        self.deflag      = False
        self.onedeflag   = False
        self.segini      = True

    def run(self, reconJ, gt_data, exeno, surface, lhs=0, rhs=0, dmap=[], bdry=[], lowpass=True):
        """ according to different exercise, doing different processing
        """
        if not self.order[exeno][self.oidx] == 'end':
            if exeno == 1:
                if self.cntdown < 0:
                    if len(self.holdlist) == 0:  # hand in the holding state or not
                        self.holdlist = reconJ
                    else:
                        self.holdlist = np.vstack([self.holdlist, reconJ]) 
                        # print np.sum(np.abs(self.holdlist[0]-self.holdlist[-1])[self.jweight[2] != 0])
                        if np.sum(np.abs(self.holdlist[0]-self.holdlist[-1])[self.jweight[2] != 0]) > 400:
                            self.holdstate = False
                    if self.holdstate:
                        self.io.typetext(surface,'Starting breath in/out' ,(100,100),(255,0,0))
                        self.breathIO(bdry, dmap)
                    else:
                        if not self.do_once:
                            self.breath_analyze(self.fcnt)
                            self.do_once = True
                            self._done = True
                else:
                    self.io.typetext(surface,'will Starting at '+str(np.round(self.cntdown/30.,2))+' second' ,(100,100),(255,0,0))
                    self.cntdown -= 1
            elif exeno == 2:
                if self.order[exeno][self.oidx] == [2]:
                    if len(self.holdlist) == 0:  # hand in the holding state or not
                       self.holdlist = reconJ
                    else:
                        self.holdlist = np.vstack([self.holdlist, reconJ]) 
                        # print np.sum(np.abs(self.holdlist[0]-self.holdlist[-1])[self.jweight[2] != 0])
                        if np.sum(np.abs(self.holdlist[0]-self.holdlist[-1])[self.jweight[2] != 0]) > 1000:
                            self.holdstate = False
                    if self.holdstate:
                        self.io.typetext(surface,'Starting breath in/out with hand close/open ' ,(100,100),(255,0,0))
                        self.handstate(lhs, rhs)
                        self.breathIO(bdry, dmap)
                    else:
                        if not self.do_once:
                            self.breath_analyze(self.fcnt)
                            self.hand_analyze(self.fcnt)
                            self.do_once = True                        
                        self.matching(reconJ, gt_data, exeno)
                else:
                    self.matching(reconJ, gt_data, exeno)    
            elif exeno == 3:
                self.matching(reconJ, gt_data, exeno)
            elif exeno == 4:
                self.matching(reconJ, gt_data, exeno)
            else:
                raise ImportError('No such exercise !!')
        else:
            print('================= exe END ======================')
            self._done = True

    def matching(self, reconJ, gt_data, exeno, lowpass=True):
        """the main part of dtw matching algorithm
        """
        self.fcnt += 1
        if self.segini:  # new segement/movement start
            self.segini = False
            # self.Ani_idx = self.aniorder[exeno][self.Ani_idx]
            if len(self.order[exeno][self.oidx]) == 1:
                self.gt_idx = self.order[exeno][self.oidx][0]
                self.idxlist.append(self.gt_idx)
        if len(self.seqlist_reg) == 0:  # build sequence list
            self.seqlist_reg = reconJ
            # print self.seqlist_reg.shape
            self.seqlist_reg = self.seqlist_reg.reshape(-1, 21)
            self.seqlist = self.seqlist_reg
        else:
            self.seqlist_reg = np.vstack([self.seqlist_reg, reconJ])
            self.seqlist_gf = gf(self.seqlist_reg, 3, axis=0)
            if not lowpass:
                self.seqlist = self.seqlist_reg
            else:
                self.seqlist = self.seqlist_gf
        if not self.deflag:  # Not yet decreasing
            if np.mod(self.seqlist.shape[0]-self.presv_size-1, 10) == 0:
                # check every 10 frames
                if len(self.order[exeno][self.oidx]) > 1:
                    if self.seqlist.shape[0] > 1:
                        result = self.clip(self.seqlist, exeno)
                        if result != []:
                            endidx = result[0]
                            if self.seqlist[endidx, 7] < 150:
                                minidx = 2
                            else:
                                minidx = 3
                            self.gt_idx = minidx
                            self.idxlist.append(self.gt_idx)
                            self.seg_update(endidx)
                else:
                    test_data_p = self.seqlist + np.atleast_2d((gt_data[self.gt_idx][0, :]-self.seqlist[0, :]))
                    self.dist_p, _ = fastdtw(gt_data[self.gt_idx], test_data_p, self.jweight[exeno], dist=self.wt_euclidean)
                    if (self.seqlist.shape[0] == 1+self.presv_size):  # new movement initail setting
                        self.dpfirst, _ = fastdtw(gt_data[self.gt_idx], test_data_p[:2], self.jweight[exeno], dist=self.wt_euclidean)
                        print('dpfirst is : %f' % self.dpfirst)
                    else:
                        print('de diff is :%f' % (self.dpfirst - self.dist_p))
                        if (self.dpfirst - self.dist_p) > self.decTh:
                            print('=========')
                            print('deflag on')
                            print('=========')
                            self.deflag = True
                            self.distp_prev = self.dist_p
        else:  # already start decreasing
            test_data_p = self.seqlist + np.atleast_2d((gt_data[self.gt_idx][0, :] - self.seqlist[0, :]))
            self.dist_p, path_p = fastdtw(gt_data[self.gt_idx], test_data_p, self.jweight[exeno], dist=self.wt_euclidean)
            if self.chk_flag:  # in check global min status
                self.cnt += 1
                if self.dist_p < self.distp_cmp:  # find smaller value
                    self.cnt = 1
                    self.distp_cmp = self.dist_p
                    self.idx_cmp = self.seqlist.shape[0]
                    print(' ==== reset ====')
                elif self.cnt == self.srchfw:
                    self.evalstr = 'Well done'
                    self.chk_flag = False
                    tgrad = 0
                    for ii in xrange(self.seqlist.shape[1]):  # maybe can include jweight
                        tgrad += (np.gradient(gf(self.seqlist[:, ii], 1))**2)*self.jweight[exeno][ii]
                    tgrad = tgrad**0.5
                    endidx = np.argmin(tgrad[self.idx_cmp-self.srchbw:self.idx_cmp+self.srchfw-1])\
                                + (self.idx_cmp-self.srchbw)
                    self.seg_update(endidx)
            else:
                if (self.dist_p - self.distp_prev) > 0:  # turning point
                    print (' ==============  large ====================')
                    self.distp_cmp = self.distp_prev
                    self.idx_cmp = self.seqlist.shape[0]
                    self.chk_flag = True
            self.distp_prev = self.dist_p


    def handstate(self, lhs, rhs):
        """ check the hand state and analyze it.
            the value of the lhs and rhs represent the tracking
            state given foem Kinect sensor. 
            0: unknown
            1: not tracked
            2: open
            3: closed
            4: lasso
        """
        self.rawhstate = np.vstack([self.rawhstate, np.array([lhs,rhs]).reshape(-1, 2)])
        if lhs == 4:
            lhs = 0
        if rhs == 4:
            rhs = 0
        if (lhs == 0 | lhs == 1 ) and (rhs == 2 or rhs == 3): # if left hand is no tracking , using right
            lhs = rhs
        elif (rhs == 0 | rhs == 1 ) and (lhs == 2 or lhs == 3): # if right hand is no tracking , using left
            rhs = lhs
        # if hand state unknown, assign defalut state
        if lhs == 0:
            if len(self.hstate) == 0:
                lhs = 2
            else:
                lhs = self.hstate[-1, 0]
        if rhs == 0:
            if len(self.hstate) == 0:
                rhs = 2
            else:
                rhs = self.hstate[-1, 1]

        if len(self.hstate) == 0:
            self.hstate = np.array([lhs,rhs]).reshape(-1, 2)
            self.hstate = np.vstack([self.hstate, self.hstate])  # duplicate the data 
        else:
            self.hstate = np.vstack([self.hstate, np.array([lhs,rhs]).reshape(-1, 2)])

        # if np.mod(self.hcnt, 10) == 0:
        #     hstate_diff = self.hstate - np.roll(self.hstate, -1, axis=0)
        #     hand_chg = hstate_diff[-11:-1, 0]+hstate_diff[-11:-1, 1]
        #     if np.sum(hand_chg) == 2:  # hand close -> open
        #         self.hopen_list.append(self.hcnt)
        #         self.hstate_cnt[0] += 1
        #     elif np.sum(hand_chg) == -2:  # hand open -> close
        #         self.hclose_list.append(self.hcnt)
        #         self.hstate_cnt[1] += 1
        #     # two hand state are not the same    
        #     elif np.sum(hand_chg) == 1:  # open state, one hand not open
        #         if np.sum(hand_chg[0]) == 1:
        #             print('please open your left hand')
        #         else:
        #             print('please open your right hand')
        #     elif np.sum(hand_chg) == -1:  # close state, one hand not close
        #         if np.sum(hand_chg[0]) == -1:
        #             print('please close your left hand')
        #         else:
        #             print('please close your right hand')

    def breathIO(self, bdry, dmap):
        """according to the depth map in the chest region,
           detect breath in and breath out.
        """
        cur_bdry = np.array([bdry[0][1], bdry[3][1], bdry[1][0], bdry[2][0]])
        if len(self.ref_bdry) == 0:
            # setup reference frame's boundary (up, down, left and right)
            self.ref_bdry = cur_bdry
            self.ref_dmap = dmap
        else:
            ubdry = np.array([int(min(cur_bdry[0], self.ref_bdry[0])),
                              int(max(cur_bdry[1], self.ref_bdry[1])),
                              int(max(cur_bdry[2], self.ref_bdry[2])),
                              int(min(cur_bdry[3], self.ref_bdry[3]))])
            blk_diff = gf_2D(abs(dmap-self.ref_dmap)[ubdry[1]:ubdry[0], ubdry[2]:ubdry[3]], 5)
            self.breath_list.append(np.mean(blk_diff))

    def find_pair_within(self, l1, l2, dist=10):
        """ from list 1 and list 2 find pairs
        """
        b = 0
        e = 0
        ans = []
        for idx,a in enumerate(l1):
            while b < len(l2) and a - l2[b] > dist:
                b += 1
            while e < len(l2) and l2[e] - a <= dist:
                e += 1
            ans.extend([(idx,b) for x in l2[b:e]])
        return ans
    
    def breath_analyze(self, offset=0, th=10):
        """ Analyze the human and breath in/out behavior
            
        """
        # breath part
        breath_gd = np.gradient(gf_2D(self.breath_list, 10))
        breath_gd[breath_gd > 0] = 0
        breath_gd[breath_gd < 0] = 1
        breath_pulse = breath_gd[:-1]-np.roll(breath_gd, -1)[:-1]
        breath_in = argrelextrema(breath_pulse, np.less, order=10)[0]+offset
        breath_out = argrelextrema(breath_pulse, np.greater, order=10)[0]+offset
        breath = np.sort(np.hstack([breath_in, breath_out]))
        if len(breath) != 0:
            if breath_in[0] == breath[0]:  # breath in happen first
                string = ['in', 'out']
            else:
                string = ['out', 'in']
            print('\n')
            for idx in xrange(len(breath)-1):
                print('breath '+string[np.mod(idx, 2)]+' from frame '+str(breath[idx])
                    +' to frame '+str(breath[idx+1]-1))
            diff = (np.roll(breath,-1)-breath)[:-1]
            print('\naverage breath '+string[0]+' freq is: '+str(np.round(30./np.mean(diff[0::2]),2))+' Hz')
            print('average breath '+string[1]+' freq is: '+str(np.round(30./np.mean(diff[1::2]),2))+' Hz\n')
        else:
            raise ImportError('Doing too fast !! please redo again !!')    

    def hand_analyze(self, offset=0, th=10):
        """Analyze the human and hand open/close behavior
        """
        # === hand close/open part ===
        foo = signal.medfilt(self.hstate, kernel_size=3)
        self.hstate[1:-1] = foo[1:-1]
        if np.sum(self.hstate[0]) != 4:
            self.error.append('two hand must open when you rise you hands')
        if np.sum(self.hstate[-1]) != 4:
            self.error.append('two hand must open when you put down your hands')
        hand_pulse = (self.hstate - np.roll(self.hstate, -1, axis=0))[:-1]
        lh         = np.where(hand_pulse[:, 0] != 0)[0]
        lh_open    = np.where(hand_pulse[:, 0] == 1)[0]
        lh_close   = np.where(hand_pulse[:, 0] == -1)[0]
        rh         = np.where(hand_pulse[:, 1] != 0)[0]
        rh_open    = np.where(hand_pulse[:, 1] == 1)[0]
        rh_close   = np.where(hand_pulse[:, 1] == -1)[0]
        # open test
        pair = self.find_pair_within(lh_open, rh_open)
        if len(lh_open) != len(rh_open):
            foo = np.array(pair)
            res = list(set(foo[:,0])-set(foo[:,1]))
            if len(lh_open) > len(rh_open):
                string = 'right hand'
            else:
                string = 'left hand'
            for i in res:
                self.error.append(string+' did not open at '+str(i+1)+' time')
            print('hand open '+str(max(len(lh_open), len(rh_open)))+' times,')
        else:
            print('hand open '+str(len(lh_open))+' times')
        # close test
        pair = self.find_pair_within(lh_open, rh_open)
        if len(lh_close) != len(rh_close):
            foo = np.array(pair)
            res = list(set(foo[:,0])-set(foo[:,1]))
            if len(lh_close) > len(rh_close):
                string = 'right hand'
            else:
                string = 'left hand'
            for i in res:
                self.error.append(string+' did not close at '+str(i+1)+' time')       
            print('hand close '+str(max(len(lh_close), len(rh_close)))+' times,')
        else:
            print('hand close '+str(len(lh_close))+ ' times\n')

    def evaluation(self, exeno, err=[]):
        """ exercise performance evaluation
        """
        fig = plt.figure(1)
        if exeno == 1:       
            plt.plot(gf_2D(self.breath_list, 10)/self.breath_list[0]*2, color='g')
            plt.title('Breath in and out')
            fig.savefig('output/bio.jpg')
            pdb.set_trace()
        elif exeno == 2:
            plt.plot(self.hstate[:,0], color='b')
            plt.plot(self.hstate[:,1]-5, color='r')
            plt.plot(gf_2D(self.breath_list, 10)/self.breath_list[0]*2, color='g')
            plt.title('Breath in and out & hands open and close')
            fig.savefig('output/biohoc.jpg')
        plt.close(fig)

        print('\nevaluation:')
        if len(self.error) != 0:
            for i in self.error:
                print i
            print('\n')
        else:
            print('perfect !!\n')
