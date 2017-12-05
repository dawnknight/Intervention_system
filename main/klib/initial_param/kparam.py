import datetime


class Kparam(object):
    """ Kinect intervention project parameters' initialization 
    """

    def __init__(self,exeno, username):

        self.bdjoints   = []
        self.now  = datetime.datetime.now() 
        self.dstr = './output/'+username+'data'+repr(self.now.year)+repr(self.now.month).zfill(2)\
                               +repr(self.now.day).zfill(2)+repr(self.now.hour).zfill(2)\
                               +repr(self.now.minute).zfill(2)+repr(self.now.second).zfill(2)+str(exeno)
        self.scale       = 1.0
        self.pre_scale   = 1.0  
        self._done       = False
        self.finish      = False
        self.handmode    = False
        self.vid_rcd     = False
        self.model_draw  = False
        self.model_frame = False
        self.clipNo      = 0
        self.fno         = 0 
        self.framecnt    = 0
                  




