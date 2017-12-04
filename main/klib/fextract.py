import cv2,operator
import pygame
import numpy as np
import scipy.ndimage.morphology as ndm
from scipy.ndimage.morphology import distance_transform_edt as dte
from numpy.linalg import norm as nlnorm
from dataoutput import Dataoutput

class Finger_extract(object):
    """ 
    """
    def __init__(self):
        self.io  = Dataoutput()

    def handseg(self, frame_in, hw, hc, w=1920, h=1080):
        """ segment hand region
            input frame : binary array
            hc : hand center
        """
        rad = int(((hw.x-hc.x)**2+(hw.y-hc.y)**2)**0.5)*2
        xul = max(int(hc.x)-rad, 0)
        yul = max(int(hc.y)-rad, 0)
        xlr = min(int(hc.x)+rad, w)
        ylr = min(int(hc.y)+rad, h)
        hand = ndm.binary_opening(frame_in[yul:ylr, xul:xlr], structure=np.ones((5, 5))).astype(np.uint8)  # hand closing

        return hand, (xul, yul), rad

    def hand_contour_find(self, contours):
        """
        """
        max_area = 0
        largest_contour = -1
        for i in xrange(len(contours)):
            cont = contours[i]
            area = cv2.contourArea(cont)
            if(area > max_area):
                max_area = area
                largest_contour = i

        if(largest_contour == -1):
            return False, 0
        else:
            h_contour = contours[largest_contour]
            return True, h_contour

    def find_hand_center(self, thresh):
        """
        """
        thresh[thresh > 0] = 1
        th, tw = thresh.shape
        dist = dte(thresh).flatten()  # find dist between hand pts and hand contour (can speed up)
        distidx = dist.argmax()
        return (np.mod(distidx, tw), distidx//tw)

    def find_angle(self, fpts, midpt):
        """
        """
        v = fpts - midpt
        vdist = nlnorm(v, axis=1)
        vlen = len(v)
        indot = [sum(v[i]*v[i+1]) for i in xrange(0, vlen, 2)]
        costheta = [indot[i//2]/(vdist[i//2*2]*vdist[i//2*2+1]) for i in xrange(vlen) ]
        return np.arccos(costheta)/np.pi*180

    def find_fingers(self, cnt, center, wrist, offset, rad):
        """
        """
        bx, by, bw, bh = cv2.boundingRect(cnt)
        hull = cv2.convexHull(cnt, returnPoints=False)
        defects = cv2.convexityDefects(cnt, hull)
        if defects is not None:
            cor = []
            mid = []
            for i in xrange(len(defects)):
                s, e, f, d = defects[i, 0]
                cor.append(tuple(cnt[s][0]))  # append start pt
                cor.append(tuple(cnt[e][0]))  # append end pt
                # append twice
                mid.append(tuple(cnt[f][0])) 
                mid.append(tuple(cnt[f][0]))
            # select real finger points
            cmtx = np.array(cor)
            mmtx = np.array(mid)
            distchk = ((cmtx.T[0]-center[0])**2+(cmtx.T[1]-center[1])**2)**0.5 > max(bw, bh)/3.  # finger len check
            wristchk = ((mmtx.T[0]-wrist.x+offset[0])**2+(mmtx.T[1]-wrist.y+offset[1])**2)**0.5 > ((center[0]-wrist.x+offset[0])**2+(center[1]-wrist.y+offset[1])**2)**0.5
            aglchk = self.find_angle(cmtx, np.array(mid)) < 90  # finger angle check

            chkidx = np.where((aglchk & distchk & wristchk) == True)[0]

            cor = list(set([cor[i] for i in chkidx]))  # remove duplicate pts 
            cormtx = np.array(cor)
            chk = len(cor)
            X = np.array([])
            Y = np.array([])

            if chk > 1 :  # more than 5 finger points # merger close pt pairs
                # calculate distance
                XX1 = np.tile(cormtx.T[0], (chk, 1))
                XX2 = np.tile(np.array([cormtx.T[0]]).T, (1, chk))
                YY1 = np.tile(cormtx.T[1], (chk, 1))
                YY2 = np.tile(np.array([cormtx.T[1]]).T, (1, chk))

                distpt = ((XX1-XX2)**2+(YY1-YY2)**2)**0.5   #pt dist mtx 
    
                th = rad/5.
                distpt[distpt > th] = 0
                # find shortest dist in dist matrix (in upper triangle)
                temp = np.nonzero(np.triu(distpt))  # points coordinate
                dup = (np.bincount(temp[0], minlength=chk) > 1) | (np.bincount(temp[1], minlength=chk) > 1)
                duppts = np.where(dup)[0]  # duplicat pts: one pt close to multi pts
                rmidx = np.array([])

                for i in duppts:
                    dupidx = np.where((temp[0] == i) | (temp[1] == i))[0]
                    minidx = distpt[temp[0][dupidx],temp[1][dupidx]].argmin()
                    delidx = np.delete(dupidx, minidx)
                    rmidx  = np.append(rmidx, delidx)

                X = np.delete(temp[0], rmidx)
                Y = np.delete(temp[1], rmidx)

            #-- merge points
                fingeridx = np.delete(np.arange(chk), np.append(X, Y))
                finger = [cor[i] for i in fingeridx]
                for i,j in zip(X, Y):
                    finger.append(tuple(np.add(cor[i], cor[j])//2))   
                return finger 
            elif chk == 1:
                return cor
            else:
                return False
        else:            
            return False
                    
    def draw_hand(self, thresh, frame, offset, rad, wrist, color, frame_surface): 
        """
        """
        try:
            contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        except:
            return []
        if len(contours) == 1:  # if has muiltiple contours
            cnt = contours[0]
            found = 1
        else:
            found, cnt = self.hand_contour_find(contours)
        if found:
            center = self.find_hand_center(thresh)
            fingers = self.find_fingers(cnt, center, wrist, offset, rad)
            if fingers :
                for i in fingers:
                    pts = tuple(map(operator.add, i, offset))
                    pygame.draw.circle(frame_surface, color, pts, 10, 8)
                    pygame.draw.line(frame_surface, color, pts, (center[0]+offset[0],center[1]+offset[1]), 8)  
                return fingers
            else:
                return[]
        else:
                return[]

    def run(self, frame, bkimg, body, bddic, joint_points, color, frame_surface):
        """
        """
        hsvimg = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)    
        tmp = np.abs(hsvimg[:, :, 2] - bkimg[:, :, 2] )                                           
        fgimg = np.zeros([1080, 1920])                  
        fgimg[tmp > 80] = 1 
                    
        if (body.hand_left_state == 2) | (body.hand_left_state == 0):  # hand open
            Lhand, bddic['Loffset'], Lrad = self.handseg(fgimg, joint_points[6], joint_points[7])                  
            bddic['Lhand'] = self.draw_hand(Lhand, frame, bddic['Loffset'], Lrad, joint_points[6], color, frame_surface)
            self.io.typetext(frame_surface, 'Lhand :'+repr(len(bddic['Lhand'])) +' fingers ', (100, 100)) 
            if  (body.hand_left_state == 2) & ( len(bddic['Lhand']) <= 3):
                self.io.typetext(frame_surface, 'Open your left hand more !!', (1000, 100), (255, 0, 0), 60, True)
            else:
                self.io.typetext(frame_surface, 'nice job !!', (1600, 100), (0, 255, 0))
        elif body.hand_left_state == 4:  # Lasso
            Lhand, bddic['Loffset'], Lrad = self.handseg(fgimg, joint_points[6], joint_points[7])                  
            bddic['Lhand'] = self.draw_hand(Lhand, frame,bddic['Loffset'], Lrad,joint_points[6], color, frame_surface)
            self.io.typetext(frame_surface, 'Lhand :'+repr(len(bddic['Lhand'])) +' fingers ', (100, 100))
        elif body.hand_left_state == 3 :  # closed
            self.io.typetext(frame_surface, 'Lhand : closed', (100, 100))
        else:
            self.io.typetext(frame_surface, 'Lhand : Not detect', (100, 100))
            
        self.io.typetext(frame_surface, 'Rhand :'+repr(body.hand_right_state), (100, 200))     
        if (body.hand_right_state == 2) | (body.hand_right_state == 0):
            Rhand, bddic['Roffset'], Rrad = self.handseg(fgimg, joint_points[10], joint_points[11])
            bddic['Rhand'] = self.draw_hand(Rhand, frame, bddic['Roffset'],Rrad, joint_points[10], color, frame_surface) 
            self.io.typetext(frame_surface, 'Rhand :'+repr(len(bddic['Rhand'])) +' fingers ', (100, 150))
            if  (body.hand_right_state == 2) & (len(bddic['Rhand']) <= 3):
                self.io.typetext(frame_surface, 'Open your right hand more !!', (1000, 150), (255, 0, 0), 60, True)
            else:
                self.io.typetext(frame_surface, 'nice job !!', (1600, 150), (0, 255, 0))
  
        elif body.hand_right_state == 4:
            Rhand, bddic['Roffset'], Rrad = self.handseg(fgimg,joint_points[10],joint_points[11])
            bddic['Rhand'] = self.draw_hand(Rhand, frame, bddic['Roffset'], Rrad, joint_points[10], color, frame_surface) 
            self.io.typetext(frame_surface, 'Rhand :'+repr(len(bddic['Rhand'])) +' fingers ', (100, 150))                                                              
        elif body.hand_right_state == 3:
            self.io.typetext(frame_surface, 'Rhand : closed', (100, 150))
        else:
            self.io.typetext(frame_surface, 'Rhand : Not detect', (100, 150))