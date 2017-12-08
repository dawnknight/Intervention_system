# -*- coding: utf-8 -*-
from .pykinect2 import PyKinectV2
from .pykinect2.PyKinectV2 import *
from .pykinect2 import PyKinectRuntime
import ctypes, os, datetime
import pygame, h5py, sys
import pdb, time, cv2, cPickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.externals import joblib
from collections import defaultdict

# import class
import movie
from dtw         import Dtw
from denoise     import Denoise
from initial_param.kparam      import Kparam
from rel         import Rel
from dataoutput  import Dataoutput
from human_model import Human_model
from skeleton    import Skeleton
from fextract    import Finger_extract

fps = 30
bkimg = np.zeros([1080, 1920])
username = 'Andy_'  # user name

# colors for drawing different bodies
SKELETON_COLORS = [pygame.color.THECOLORS["red"],
                   pygame.color.THECOLORS["blue"],
                   pygame.color.THECOLORS["green"],
                   pygame.color.THECOLORS["orange"],
                   pygame.color.THECOLORS["purple"],
                   pygame.color.THECOLORS["yellow"],
                   pygame.color.THECOLORS["violet"]]
# GPR
limbidx = np.array([4, 5, 6, 8, 9, 10, 20])
# DTW
data    = h5py.File('data/GT_V_data_mod_EX4.h5', 'r')
gt_data = defaultdict(dict)
gt_data[4][1] = data['GT_1'][:]
gt_data[4][2] = data['GT_2'][:]
gt_data[4][3] = data['GT_3'][:]
gt_data[4][4] = data['GT_4'][:]
data.close()
data = h5py.File('data/GT_V_data_mod_EX3.h5', 'r')
gt_data[3][1] = data['GT_1'][:]
gt_data[3][2] = data['GT_2'][:]
gt_data[3][3] = data['GT_3'][:]
gt_data[3][4] = data['GT_4'][:]
data.close()
data = h5py.File('data/GT_V_data_mod_EX2.h5', 'r')
gt_data[2][1] = data['GT_1'][:]
gt_data[2][2] = data['GT_2'][:]
data.close()


class BodyGameRuntime(object):

    def __init__(self):
        global bkimg
        pygame.init()
        # Used to manage how fast the screen updates
        self._clock = pygame.time.Clock()
        # Set the width and height of the screen [width, height]
        self._infoObject = pygame.display.Info()
        self._screen = pygame.display.set_mode((self._infoObject.current_w >> 1, self._infoObject.current_h >> 1),
                                               pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.RESIZABLE, 32)

        pygame.display.set_caption("Kinect Body detection")

        # Kinect runtime object, we want only color and body frames
        self._kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color |
                                                       PyKinectV2.FrameSourceTypes_Body |
                                                       PyKinectV2.FrameSourceTypes_Depth |
                                                       PyKinectV2.FrameSourceTypes_BodyIndex)
        # back buffer surface for getting Kinect color frames, 32bit color, width and height equal to the Kinect color frame size
        self.default_h = self._infoObject.current_h
        self.default_w = self._infoObject.current_w

        self._frame_surface = pygame.Surface((self.default_w, self.default_h), 0, 32).convert()
        self.h_to_w = float(self.default_h) / self.default_w
        # here we will store skeleton data
        self._bodies = None
        self.jorder  = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 20]  # upper body joints' order
        time.sleep(5)

        if self._kinect.has_new_color_frame():
            frame = self._kinect.get_last_color_frame().reshape([1080, 1920, 4])[:, :, :3]
            bkimg = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            print ('extract bg .....')
        else:
            print 'failed to extract .....'

        self.exeno = 4  # exercise number
        self.__param_init__()

    def __param_init__(self, clean=False):
        try:
            self.dataset.close()
            print('save h5py ....')
            if clean:
                os.remove(self.kp.dstr+'.h5')
                print('remove h5py ....')
        except:
            pass
        self.fig = None
        self.kp = Kparam(self.exeno, username)
        self.movie = movie.Movie(self.exeno)
        self.dtw = Dtw()
        self.denoise = Denoise()
        self.rel = Rel()
        self.io  = Dataoutput()
        self.h_mod = Human_model()
        self.skel = Skeleton()
        self.fextr = Finger_extract()

    def draw_color_frame(self, frame, target_surface):
        target_surface.lock()
        address = self._kinect.surface_as_array(target_surface.get_buffer())
        ctypes.memmove(address, frame.ctypes.data, frame.size)
        del address
        target_surface.unlock()

    def reset(self, clean=False, change=False):
        self.movie.stop(True)
        del self.movie
        self.__param_init__(clean)

    def getcoord(self, data, order=[1, 4, 8, 20]):
        foo = []
        for i in order:
            if i == 1:
                foo = np.array([data[i].x, data[i].y])
            else:
                foo = np.vstack([foo, np.array([data[i].x, data[i].y])])
        return foo

    def press_event(self, press):
        """ According to the button which is pressed by the user
            doing correspond action
        """
        if press[pygame.K_ESCAPE]:
            self.kp._done = True
            self.movie.stop()

        if press[pygame.K_h]:  # use 'h' to open, 'ctrl+h' to close finger detection
            if press[pygame.K_LCTRL] or press[pygame.K_RCTRL]:
                print('finger detection disable .....')
                self.kp.handmode = False
            else:
                print('finger detection enable .....')
                self.kp.handmode = True

        if press[pygame.K_m]:  # use 'm' to open, 'ctrl+m' to close human model
            if press[pygame.K_LCTRL] or press[pygame.K_RCTRL]:
                print('human model disable .....')
                plt.close(self.fig)
                self.kp.model_draw = False
                self.kp.model_frame = False
            else:
                print('human model enable .....')
                self.kp.model_draw = True

        if press[pygame.K_r]:  # use 'r' to start video recording
            if press[pygame.K_LCTRL] or press[pygame.K_RCTRL]:
                print('stop recording .....')
                self.kp.vid_rcd = False
            else:
                if self.kp.clipNo == 0:
                    self.dataset = h5py.File(self.kp.dstr+'.h5', 'w')
                    self.dataset = h5py.File(self.kp.dstr+'.h5', 'r')
                    # img group
                    self.imgs = self.dataset.create_group('imgs')
    #                        self.cimgs = self.imgs.create_group('cimgs')
                    self.dimgs = self.imgs.create_group('dimgs')
                    self.bdimgs = self.imgs.create_group('bdimgs')
                print('recording .....')
                self.kp.vid_rcd = True
                self.kp.clipNo += 1

        if press[pygame.K_g]:  # use 'g' to to open, 'ctrl+g' to close gpr denoise
            if press[pygame.K_LCTRL] or press[pygame.K_RCTRL]:
                print('close denoising process .....')
                self.denoise._done = True
            else:
                print('start denoising process .....')
                self.denoise._done = False

        if press[pygame.K_d]:  # use 'd' to to open, 'ctrl+d' to close dtw
            if press[pygame.K_LCTRL] or press[pygame.K_RCTRL]:
                print('disable human behavior analyze .....')
                self.dtw._done = True
                self.kp.finish = True
            else:
                print('enable human behavior analyze .....')
                self.dtw._done = False
                self.kp.finish = False

        if press[pygame.K_i]:  # use 'i' to reset every parameter
            print('Reseting ............................')
            self.reset()
        if press[pygame.K_u]:  # use 'u' to reset every parameter and remove the save data
            print('Reseting & removing the saved file ................')
            self.reset(True)

        if press[pygame.K_b]:  # use 'b' to lager the scale
            self.kp.scale = min(self.kp.scale*1.1, 1.8)
        if press[pygame.K_s]:  # use 's' to smaller the scale
            self.kp.scale = max(self.kp.scale/1.1, 1)

        if press[pygame.K_1]:  # use '1' to change to execise 1
            self.exeno = 1
            print('====  doing exercise 1 ====')
            self.reset(change=True)
        if press[pygame.K_2]:  # use '2' to change to execise 2
            self.exeno = 2
            print('====  doing exercise 2 ====')
            self.reset(change=True)
        if press[pygame.K_3]:  # use '3' to change to execise 3
            self.exeno = 3
            print('====  doing exercise 3 ====')
            self.reset(change=True)
        if press[pygame.K_4]:  # use '4' to change to execise 4
            self.exeno = 4
            print('====  doing exercise 4 ====')
            self.reset(change=True)

    def run(self):
        wait_key_cnt = 3

        while not self.kp._done:
            bddic = {}
            jdic  = {}

            # === key pressing ===
            if(wait_key_cnt < 3):
                wait_key_cnt += 1
            if(pygame.key.get_focused() and wait_key_cnt >= 3):
                press = pygame.key.get_pressed()
                self.press_event(press)
                wait_key_cnt = 0

            # === Main event loop ===
            for event in pygame.event.get():  # User did something
                if event.type == pygame.QUIT:  # If user clicked close
                    self._done = True  # Flag that we are done so we exit this loop
                    self.movie.stop()
                elif event.type == pygame.VIDEORESIZE:  # window resized
                    self._screen = pygame.display.set_mode(event.dict['size'],
                                   pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.RESIZABLE, 32)

            # === extract data from kinect ===
            if self._kinect.has_new_color_frame():
                frame = self._kinect.get_last_color_frame()
                self.draw_color_frame(frame, self._frame_surface)
                frame = frame.reshape(1080, 1920, 4)[:, :, :3]
            if self._kinect.has_new_body_frame():
                self._bodies = self._kinect.get_last_body_frame()
                timestamp = datetime.datetime.now()
            if self._kinect.has_new_body_index_frame():
                bodyidx = self._kinect.get_last_body_index_frame()
                bodyidx = bodyidx.reshape((424, 512))
            if self._kinect.has_new_depth_frame():
                dframe, oridframe = self._kinect.get_last_depth_frame()
                dframe = dframe.reshape((424, 512))

            # === when user is detected ===
            if self._bodies is not None:
                closest_ID = -1
                cdist      = np.inf
                for i in range(0, self._kinect.max_body_count):
                    body = self._bodies.bodies[i]
                    if not body.is_tracked:
                        continue
                    if body.joints[20].Position.z <= cdist:  # find the closest body
                        closest_ID = i
                        cdist = body.joints[20].Position.z
                if (closest_ID != -1):
                    body   = self._bodies.bodies[closest_ID]
                    joints = body.joints
                    rec_joints = body.joints
                    for ii in xrange(25):
                        jdic[ii] = joints[ii]
                    jps  = self._kinect.body_joints_to_color_space(joints)  # joint points in color domain
                    djps = self._kinect.body_joints_to_depth_space(joints)  # joint points in depth domain

                    # === fingers detection ===
                    if self.kp.handmode:  # finger detect and draw
                        self.fextr.run(frame, bkimg, body, bddic, jps, SKELETON_COLORS[i], self._frame_surface)

                    # === joint reliability ===
                    Rel, Relary = self.rel.run(jdic,self.jorder)
                    # joint's reliability visulization
                    # self.skel.draw_Rel_joints(jps, Rel, self._frame_surface)

                    # === dtw analyze & denoising process ===
                    if not self.dtw._done:
                        modJary = self.h_mod.human_mod_pts(joints)  # modJary is 7*3 array
                        modJary = modJary.flatten().reshape(-1, 21)  # change shape to 1*21 array
                        if not self.denoise._done:
                            if not len(Relary) == 0:
                                # === GPR denoising ===
                                if all(ii > 0.6 for ii in Relary[limbidx]):  # all joints are reliable
                                    reconJ = modJary  # reconJ is 1*21 array
                                else:  # contains unreliable joints
                                    reconJ, unrelidx = self.denoise.run(modJary, Relary)
                                    #  === recon 2D joints in color domain ===
                                    JJ = self.h_mod.reconj2joints(rec_joints, reconJ.reshape(7, 3))
                                    for ii in [4, 5, 6, 8, 9, 10]:
                                        rec_joints[ii].Position.x = JJ[i][0]
                                        rec_joints[ii].Position.y = JJ[i][1]
                                        rec_joints[ii].Position.z = JJ[i][2]
                                    tmp_jps = self._kinect.body_joints_to_color_space(rec_joints)  # joints in color domain
                                    rec_jps = jps
                                    for ii in unrelidx:
                                        rec_jps[ii].x = tmp_jps[ii].x
                                        rec_jps[ii].y = tmp_jps[ii].y
                                    self.skel.draw_body(rec_joints, rec_jps, SKELETON_COLORS[-1], self._frame_surface, 15)
                            else:
                                reconJ = modJary
                        else:
                            reconJ = modJary

                        # === DTW matching ===
                        if self.exeno in [1, 2]:
                            bdry = self.getcoord(djps)
                            self.dtw.run(reconJ, gt_data[self.exeno], self.exeno,\
                                         self._frame_surface, body.hand_left_state,
                                         body.hand_right_state, dframe, bdry, self.kp.framecnt)                        
                        else:
                            self.dtw.run(reconJ, gt_data[self.exeno], self.exeno,\
                                         body.hand_left_state, body.hand_right_state)

                        if (body.hand_left_state == 2): #Lhand open
                            Lhstatus = 'open'
                        elif body.hand_left_state == 0:
                            Lhstatus = 'unknown'
                        elif body.hand_left_state == 3:
                            Lhstatus = 'closed'
                        elif body.hand_left_state == 4:
                            Lhstatus = 'Lasso'
                        else:
                            Lhstatus = 'Not be detected'                            

                        if (body.hand_right_state == 2): #Lhand open
                            Rhstatus = 'open'
                        elif body.hand_right_state == 0:
                            Rhstatus = 'unknown'
                        elif body.hand_right_state ==3:
                            Rhstatus = 'closed'
                        elif body.hand_right_state == 4:
                            Rhstatus = 'Lasso'
                        else:
                            Rhstatus = 'Not be detected'
                        
                        self.io.typetext(self._frame_surface, 'Lhand : '+Lhstatus, (100, 800), (255, 69, 0), fontsize=60, bold=True)        
                        self.io.typetext(self._frame_surface, 'Rhand : '+Rhstatus, (100, 900), (255, 69, 0), fontsize=60, bold=True) 

                        if self.dtw.evalstr != '':
                            self.io.typetext(self._frame_surface, self.dtw.evalstr, (100, 300),(255, 0, 0), fontsize=100)
                            self.dtw.fcnt += 1
                            if self.dtw.fcnt > 30 :
                                self.dtw.evalstr = ''
                                self.dtw.fcnt  = 0
                    else:
                        self.io.typetext(self._frame_surface, 'finish exercise '+str(self.exeno), (100, 300),(255, 0, 0), fontsize=100)
                        if not self.kp.finish:
                            self.dtw.evaluation(self.exeno)
                            print self.dtw.idxlist
                            self.kp.finish = True
                    # draw skel
                    self.skel.draw_body(joints, jps, SKELETON_COLORS[i], self._frame_surface, 8)

                    # === draw unify human model ===
                    if self.kp.model_draw:
                        modJoints = self.h_mod.human_mod_pts(joints, limb=False)
                        if not self.kp.model_frame:
                            self.fig = plt.figure(1)
                            ax = self.fig.add_subplot(111, projection='3d')
                            # keys = modJoints.keys()
                            self.kp.model_frame = True
                        else:
                            plt.cla()
                        self.h_mod.draw_human_mod_pts(modJoints, ax)
                    # === save data ===
                    bddic['timestamp'] = timestamp
                    bddic['jointspts'] = jps   # joints' coordinate in color space (2D)
                    bddic['depth_jointspts'] = djps  # joints' coordinate in depth space (2D)
                    bddic['joints'] = jdic  # joints' coordinate in camera space (3D)
                    bddic['vidclip'] = self.kp.clipNo
                    bddic['Rel'] = Rel
                    bddic['LHS'] = body.hand_left_state
                    bddic['RHS'] = body.hand_right_state

                self.kp.framecnt += 1  # frame no
            else:
                self.io.typetext(self._frame_surface, 'No human be detected ', (100, 100))

            # === text infomation on the surface ===
            if self.kp.vid_rcd:  # video recoding text
                self.io.typetext(self._frame_surface, 'Video Recording', (1550, 20), (255, 0, 0))
#                self.cimgs.create_dataset('img_'+repr(self.kp.fno).zfill(4), data = frame)
                self.bdimgs.create_dataset('bd_' + repr(self.kp.fno).zfill(4), data=np.dstack((bodyidx, bodyidx, bodyidx)))
                self.dimgs.create_dataset('d_' + repr(self.kp.fno).zfill(4), data=np.dstack((dframe, dframe, dframe)))
                self.kp.fno += 1
                self.kp.bdjoints.append(bddic)
            else:
                self.io.typetext(self._frame_surface, 'Not Recording', (1550, 20), (0, 255, 0))

            # if size of the display window is changed
            if (float(self._screen.get_height())/self._screen.get_width()) > self.h_to_w:
                target_height = int(self.h_to_w * self._screen.get_width())
                self.default_w = self._screen.get_width()
                self.default_h = target_height
            elif (float(self._screen.get_height())/self._screen.get_width()) < self.h_to_w:
                target_width = int(self._screen.get_height()/self.h_to_w)
                self.default_w = target_width
                self.default_h = self._screen.get_height()
            else:  # ration not change, then check the whether the H or W change
                if self._screen.get_width() != self.default_w:
                    target_height = int(self.h_to_w * self._screen.get_width())
                    self.default_w = self._screen.get_width()
                    self.default_h = target_height
            surface_to_draw = pygame.transform.scale(self._frame_surface, (self.default_w, self.default_h))
            self._screen.blit(surface_to_draw, (0, 0))

            # user change the avatar's sacle
            if self.kp.scale != self.kp.pre_scale:
                self.movie.draw(self._screen, self.default_w, self.default_h, self.kp.scale, True)
                self.kp.pre_scale = self.kp.scale
            else:
                self.movie.draw(self._screen, self.default_w, self.default_h, self.kp.scale)
            # update
            surface_to_draw = None
            pygame.display.update()
            # limit frames per second
            self._clock.tick(fps)
        # user end the programe
        self.movie.stop(True)   # close avatar
        self._kinect.close()    # close Kinect sensor
        print self.dtw.idxlist  # show the analyzed result
        # save the recording data

        if self.kp.bdjoints != []:
            cPickle.dump(self.kp.bdjoints, file(self.kp.dstr+'.pkl', 'wb'))
        try:
            self.dataset.close()
        except:
            pass
        pygame.quit()  # quit


