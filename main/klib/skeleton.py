import pygame
import numpy as np
from initial_param.kinect_para import Kinect_para


class Skeleton(Kinect_para):
    """ visualize the skeleton and the joint on the users
    """
    def __init__(self):
        Kinect_para.__init__(self)

    def draw_body_bone(self, joints, jointPoints, color, joint0, joint1, surface, linewidth=8):
        """ draw line between two joints
        """
        joint0State = joints[joint0].TrackingState;
        joint1State = joints[joint1].TrackingState;
        # both joints are not tracked
        if (joint0State == self.TrackingState_NotTracked) or (joint1State == self.TrackingState_NotTracked): 
            return
        # both joints are not *really* tracked
        if (joint0State == self.TrackingState_Inferred) and (joint1State == self.TrackingState_Inferred):
            return
        # ok, at least one is good
        start = (jointPoints[joint0].x, jointPoints[joint0].y)
        end = (jointPoints[joint1].x, jointPoints[joint1].y)

        try:
            pygame.draw.line(surface, color, start, end ,linewidth)
        except:  #coordinate contains invalid positions (i.e.inf)
            pass

    def draw_body(self, joints, jointPoints, color , surface, linewidth = 8):
        """draw body skeleton
        """
        # Torso
        self.draw_body_bone(joints, jointPoints, color, self.JointType_Head, self.JointType_Neck, surface, linewidth)
        self.draw_body_bone(joints, jointPoints, color, self.JointType_Neck, self.JointType_SpineShoulder, surface, linewidth)
        self.draw_body_bone(joints, jointPoints, color, self.JointType_SpineShoulder, self.JointType_SpineMid, surface, linewidth)
        self.draw_body_bone(joints, jointPoints, color, self.JointType_SpineMid, self.JointType_SpineBase, surface, linewidth)
        self.draw_body_bone(joints, jointPoints, color, self.JointType_SpineShoulder, self.JointType_ShoulderRight, surface, linewidth)
        self.draw_body_bone(joints, jointPoints, color, self.JointType_SpineShoulder, self.JointType_ShoulderLeft, surface, linewidth)
        self.draw_body_bone(joints, jointPoints, color, self.JointType_SpineBase, self.JointType_HipRight, surface, linewidth)
        self.draw_body_bone(joints, jointPoints, color, self.JointType_SpineBase, self.JointType_HipLeft, surface, linewidth)
        # Right Arm    
        self.draw_body_bone(joints, jointPoints, color, self.JointType_ShoulderRight, self.JointType_ElbowRight, surface, linewidth)
        self.draw_body_bone(joints, jointPoints, color, self.JointType_ElbowRight, self.JointType_WristRight, surface, linewidth)
        #draw_body_bone(joints, jointPoints, color, self.JointType_WristRight, self.JointType_HandRight, surface, linewidth)
        # Left Arm
        self.draw_body_bone(joints, jointPoints, color, self.JointType_ShoulderLeft, self.JointType_ElbowLeft, surface, linewidth)
        self.draw_body_bone(joints, jointPoints, color, self.JointType_ElbowLeft, self.JointType_WristLeft, surface, linewidth)
        #draw_body_bone(joints, jointPoints, color, self.JointType_WristLeft, self.JointType_HandLeft, surface, linewidth)

    def draw_Rel_joints(self, jointPoints, Rel, surface):
        """ accoding to the joint's reliability assign different size scatter to the joint
        """
        for i in Rel.keys():
            try:
                pygame.draw.circle(surface, (255, 0, 0), (int(jointPoints[i].x), int(jointPoints[i].y)), np.int((1-Rel[i])*30))
            except:
                pass
