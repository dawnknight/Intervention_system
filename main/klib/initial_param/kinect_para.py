

class Kinect_para(object):
    """ define some KinectV2 pre-assign parameters
    """
    def __init__(self):
        # joint order in kinect
        self.JointType_SpineBase     = 0
        self.JointType_SpineMid      = 1
        self.JointType_Neck          = 2
        self.JointType_Head          = 3
        self.JointType_ShoulderLeft  = 4
        self.JointType_ElbowLeft     = 5
        self.JointType_WristLeft     = 6
        self.JointType_HandLeft      = 7
        self.JointType_ShoulderRight = 8
        self.JointType_ElbowRight    = 9
        self.JointType_WristRight    = 10
        self.JointType_HandRight     = 11
        self.JointType_HipLeft       = 12
        self.JointType_KneeLeft      = 13
        self.JointType_AnkleLeft     = 14
        self.JointType_FootLeft      = 15
        self.JointType_HipRight      = 16
        self.JointType_KneeRight     = 17
        self.JointType_AnkleRight    = 18
        self.JointType_FootRight     = 19
        self.JointType_SpineShoulder = 20
        self.JointType_HandTipLeft   = 21
        self.JointType_ThumbLeft     = 22
        self.JointType_HandTipRight  = 23
        self.JointType_ThumbRight    = 24
        self.JointType_Count         = 25

        # values for enumeration '_TrackingState'
        self.TrackingState_NotTracked = 0
        self.TrackingState_Inferred   = 1
        self.TrackingState_Tracked    = 2


