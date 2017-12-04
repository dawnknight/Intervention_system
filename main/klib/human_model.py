import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from initial_param.kinect_para import Kinect_para


class Human_model(Kinect_para):
    """ Human model class.
        1. create human model(has online and offline version)
        2. reconstruct the joints position based on the denosing process
    """
    def __init__(self):
        Kinect_para.__init__(self)
        self.Jlen = {}
        self.Jlen['0203'] = 13.4   # head2neck
        self.Jlen['2002'] = 8.3    # neck2spinshoulder
        self.Jlen['0120'] = 15.4   # spinshoulder2spinmiddle
        self.Jlen['0001'] = 32.5   # spinmiddle2spinbase
        self.Jlen['2008'] = 16.65  # spinshoulder2Rshoulder
        self.Jlen['0809'] = 33.2   # Rshoulder2Relbow
        self.Jlen['0910'] = 27.1   # Relbow2Rwrist
        self.Jlen['2004'] = 16.65  # spinshoulder2Lshoulder
        self.Jlen['0405'] = 33.2   # Lshoulder2Lelbow
        self.Jlen['0506'] = 27.1   # Lelbow2Lwrist
        # setting initial spine bottom position
        self.oripos = np.array([80, 100, 0])
        self.factor = 5

    def uni_vec(self, Body, start, end):
        """ (offline version) calculate the body segment
            between two joints.
            Body : dictionary-like/array-like object.
                   dict => the dimension of each key's content is (3, #frame) array.
            return : array-like object
        """
        if isinstance(Body, dict):
            tmp = Body[start]-Body[end]
            vlen = sum(tmp**2)**.5
            vlen[vlen == 0] = 10**-6
        elif isinstance(Body, np.ndarray):
            tmp = Body[start, :]-Body[end, :]
            vlen = sum(tmp**2)**.5
            if vlen == 0:
                vlen = 10**-6
        else:
            raise ImportError('Only support dict and array object !!')

        return tmp/vlen

    def uni_vec_pts(self, Body, start, end):
        """ (online version) calculate the body segment
            between two joints.
            Body : dictionary-like object.
                   Each key's content is a pykinect object.
            return : scalar object
        """
        tmp = np.array([Body[start].Position.x-Body[end].Position.x,
                        Body[start].Position.y-Body[end].Position.y,
                        Body[start].Position.z-Body[end].Position.z])

        vlen = sum(tmp**2)**.5
        if vlen == 0:
            vlen = 10**-6
        return tmp/vlen

    def human_mod(self, Body, J={}):
        """ (offline version) calaulate each joints pair's uni-vector
            and convert it to a unified model domain.
            Body : dictionary-like object. The keys in Body represent corresponding
                   joint's order.
            return a dictionary-like object
        """
        Vec0001 = self.uni_vec(Body, self.JointType_SpineBase    , self.JointType_SpineMid)
        Vec0120 = self.uni_vec(Body, self.JointType_SpineMid     , self.JointType_SpineShoulder)
        Vec2002 = self.uni_vec(Body, self.JointType_SpineShoulder, self.JointType_Neck)
        Vec0203 = self.uni_vec(Body, self.JointType_Neck         , self.JointType_Head)
        Vec2004 = self.uni_vec(Body, self.JointType_SpineShoulder, self.JointType_ShoulderLeft)
        Vec0405 = self.uni_vec(Body, self.JointType_ShoulderLeft , self.JointType_ElbowLeft)
        Vec0506 = self.uni_vec(Body, self.JointType_ElbowLeft    , self.JointType_WristLeft)
        Vec2008 = self.uni_vec(Body, self.JointType_SpineShoulder, self.JointType_ShoulderRight)
        Vec0809 = self.uni_vec(Body, self.JointType_ShoulderRight, self.JointType_ElbowRight)
        Vec0910 = self.uni_vec(Body, self.JointType_ElbowRight   , self.JointType_WristRight)

        J[self.JointType_SpineBase]     = np.tile(self.oripos, (Body[0].shape[1], 1)).T
        J[self.JointType_SpineMid]      = J[self.JointType_SpineBase]     - Vec0001*self.Jlen['0001']*self.factor
        J[self.JointType_SpineShoulder] = J[self.JointType_SpineMid]      - Vec0120*self.Jlen['0120']*self.factor
        J[self.JointType_Neck]          = J[self.JointType_SpineShoulder] - Vec2002*self.Jlen['2002']*self.factor
        J[self.JointType_Head]          = J[self.JointType_Neck]          - Vec0203*self.Jlen['0203']*self.factor
        J[self.JointType_ShoulderLeft]  = J[self.JointType_SpineShoulder] - Vec2004*self.Jlen['2004']*self.factor
        J[self.JointType_ElbowLeft]     = J[self.JointType_ShoulderLeft]  - Vec0405*self.Jlen['0405']*self.factor
        J[self.JointType_WristLeft]     = J[self.JointType_ElbowLeft]     - Vec0506*self.Jlen['0506']*self.factor
        J[self.JointType_ShoulderRight] = J[self.JointType_SpineShoulder] - Vec2008*self.Jlen['2008']*self.factor
        J[self.JointType_ElbowRight]    = J[self.JointType_ShoulderRight] - Vec0809*self.Jlen['0809']*self.factor
        J[self.JointType_WristRight]    = J[self.JointType_ElbowRight]    - Vec0910*self.Jlen['0910']*self.factor

        return J

    def human_mod_pts(self, Body, limb=True, J={}):
        """ (online version) calaulate each joints pair's uni-vector
            and convert it to a unified model domain.
            Body : dictionary-like object. The keys in Body represent corresponding 
                   joint's order.
            limb : True  => only process limb part L/R wrist, elbow and shoulder
                   Flase => process all upper body part
            output : array-like object
        """
        Vec0001 = self.uni_vec_pts(Body, self.JointType_SpineBase    , self.JointType_SpineMid)
        Vec0120 = self.uni_vec_pts(Body, self.JointType_SpineMid     , self.JointType_SpineShoulder)
        Vec2002 = self.uni_vec_pts(Body, self.JointType_SpineShoulder, self.JointType_Neck)
        Vec0203 = self.uni_vec_pts(Body, self.JointType_Neck         , self.JointType_Head)
        Vec2004 = self.uni_vec_pts(Body, self.JointType_SpineShoulder, self.JointType_ShoulderLeft)
        Vec0405 = self.uni_vec_pts(Body, self.JointType_ShoulderLeft , self.JointType_ElbowLeft)
        Vec0506 = self.uni_vec_pts(Body, self.JointType_ElbowLeft    , self.JointType_WristLeft)
        Vec2008 = self.uni_vec_pts(Body, self.JointType_SpineShoulder, self.JointType_ShoulderRight)
        Vec0809 = self.uni_vec_pts(Body, self.JointType_ShoulderRight, self.JointType_ElbowRight)
        Vec0910 = self.uni_vec_pts(Body, self.JointType_ElbowRight   , self.JointType_WristRight)

        J[self.JointType_SpineBase]     = self.oripos
        J[self.JointType_SpineMid]      = J[self.JointType_SpineBase]     - Vec0001*self.Jlen['0001']*self.factor
        J[self.JointType_SpineShoulder] = J[self.JointType_SpineMid]      - Vec0120*self.Jlen['0120']*self.factor
        J[self.JointType_Neck]          = J[self.JointType_SpineShoulder] - Vec2002*self.Jlen['2002']*self.factor
        J[self.JointType_Head]          = J[self.JointType_Neck]          - Vec0203*self.Jlen['0203']*self.factor
        J[self.JointType_ShoulderLeft]  = J[self.JointType_SpineShoulder] - Vec2004*self.Jlen['2004']*self.factor
        J[self.JointType_ElbowLeft]     = J[self.JointType_ShoulderLeft]  - Vec0405*self.Jlen['0405']*self.factor
        J[self.JointType_WristLeft]     = J[self.JointType_ElbowLeft]     - Vec0506*self.Jlen['0506']*self.factor
        J[self.JointType_ShoulderRight] = J[self.JointType_SpineShoulder] - Vec2008*self.Jlen['2008']*self.factor
        J[self.JointType_ElbowRight]    = J[self.JointType_ShoulderRight] - Vec0809*self.Jlen['0809']*self.factor
        J[self.JointType_WristRight]    = J[self.JointType_ElbowRight]    - Vec0910*self.Jlen['0910']*self.factor

        if limb:
            joint_order = [4, 5, 6, 8, 9, 10, 20]
        else:
            joint_order = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 20]
        for idx, i in enumerate(joint_order):
            if idx == 0:
                Jary = J[i]
            else:
                Jary = np.vstack([Jary, J[i]])
        return Jary

    def draw_human_mod(self, joints):
        """ (offline version) draw an unified model
            joints : dictionary-like object
        """
        keys = joints.keys()
        nframe = joints[keys[0]].shape[1]  # total number of the frames
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for fno in xrange(len(nframe)):
            plt.cla()
            x = []
            y = []
            z = []
            for i in  xrange(len(keys)):
                x.append(joints[keys[i]][0][fno])
                y.append(joints[keys[i]][1][fno])
                z.append(-1*joints[keys[i]][2][fno])
            ax.scatter(z, x, y, c='red', s=100)

            ax.set_xlim(-200, 200)
            ax.set_ylim(-400, 400)
            ax.set_zlim(100, 500)
            ax.set_title(fno)

            plt.draw()
            plt.pause(1.0/120)

    def draw_human_mod_pts(self, joints, surface):
        """ (online version) draw an unified model.
            joints : array-like object
        """
        x = joints[:, 0]
        y = joints[:, 1]
        z = joints[:, 2]

        surface.scatter(z, x, y, c='red', s=100)
        surface.set_xlim(-200, 200)
        surface.set_ylim(-400, 400)
        surface.set_zlim(100, 500)
        plt.draw()
        plt.pause(1.0/120)

    def reconj2joints(self, joints, recon_body, joint_order=[4, 5, 6, 8, 9, 10, 20]):
        """ accoding to the denoising process, reconstruct the joints
            joints : dictionary-like object.
                     Each key's content is a pykinect object.
            recon_body : array-like object. dimension is 7-by-3.
            joint_order : limb joints' order
            output : dictionary-like object
        """
        ori_body = {}
        for i in joint_order:
            ori_body[i] = np.array([joints[i].Position.x, joints[i].Position.y, joints[i].Position.z])
        # unified vector for each body segment
        Vec45 = self.uni_vec(recon_body, 1, 0)  # 1 => joint_order(5), 0 => joint_order(4)
        Vec56 = self.uni_vec(recon_body, 2, 1)
        Vec89 = self.uni_vec(recon_body, 4, 3)
        Vec90 = self.uni_vec(recon_body, 5, 4)
        # real-length for each body segment
        Len45 = np.mean(np.sum((ori_body[5] - ori_body[4])**2, axis=0)**0.5)
        Len56 = np.mean(np.sum((ori_body[6] - ori_body[5])**2, axis=0)**0.5)
        Len89 = np.mean(np.sum((ori_body[9] - ori_body[8])**2, axis=0)**0.5)
        Len90 = np.mean(np.sum((ori_body[10]- ori_body[9])**2, axis=0)**0.5)
        # reconstruct the joint
        J = {}
        J[self.JointType_ShoulderLeft] = ori_body[4]  # use real Lshoulder position as reference
        J[self.JointType_ElbowLeft]    = Vec45*Len45+ori_body[4]
        J[self.JointType_WristLeft]    = Vec56*Len56+J[self.JointType_ElbowLeft]

        J[self.JointType_ShoulderRight] = ori_body[8]  # use real Rshoulder position as reference
        J[self.JointType_ElbowRight]    = Vec89*Len89+ori_body[8]
        J[self.JointType_WristRight]    = Vec90*Len90+J[self.JointType_ElbowRight]

        return J
