# -*- coding: utf-8 -*-
import os
import cv2
import h5py
import glob
import pygame
import cPickle
import win32com.client
import numpy as np


class Dataoutput(object):
    """
    """
    def typetext(self, frame_surface, string, pos, color=(255, 255, 0), fontsize=60, bold=False):
        """showing the text information on the surface
        """
        myfont = pygame.font.SysFont("Arial", fontsize, bold)
        label = myfont.render(string, 1, color)
        frame_surface.blit(label, pos)

    def folder_retarget(self, src_path, shortcut):
        """redirct the shortcut folder to the real path
        """
        shell = win32com.client.Dispatch("WScript.Shell")
        return str(shell.CreateShortCut(src_path+shortcut).Targetpath)

    def makevid(self, src_path, dst_path, fps=30):
        """ convert the saved file into video
        """
        for subfolder in os.listdir(src_path):
            if '.lnk' in subfolder:
                path = folder_retarget(src_path, subfolder)
            else:
                path = src_path

            filelist = glob.glob(os.path.join(path, '*.h5'))  # find all h5 files

            for infile in filelist:
                print infile
                filename = infile.split('\\')[-1][:-3]
                f = h5py.File(infile, "r")

                if len(f['imgs'].keys()) == 2:
                    for j in xrange(2):
                        size = (512, 424)
                        if j == 0:
                            video = cv.CreateVideoWriter(savepath+filename+'_bdidx.avi', cv.CV_FOURCC('X', 'V', 'I', 'D'), fps, size, True)
                            cimg = f['imgs']['bdimgs']
                        else:
                            video = cv.CreateVideoWriter(savepath+filename+'_d.avi', cv.CV_FOURCC('X', 'V', 'I', 'D'), fps, size, True)
                            cimg = f['imgs']['dimgs']
                        for i in cimg.keys():
                            bitmap = cv.CreateImageHeader(size, cv.IPL_DEPTH_8U, 3)
                            if j == 1:
                                cv.SetData(bitmap, np.uint8(cimg[i][:]/256.).tostring(), np.uint8(cimg[i][:]/256.).dtype.itemsize*3*cimg[i][:].shape[1])
                            else:
                                cv.SetData(bitmap, cimg[i][:].tostring(), cimg[i][:].dtype.itemsize*3*cimg[i][:].shape[1])
                            cv.WriteFrame(video, bitmap)
                        del video
                elif len(f['imgs'].keys()) == 3:
                    for j in xrange(3):
                        if j == 0:
                            size = (1920, 1080)
                            video = cv.CreateVideoWriter(savepath+subfolder+'_'+filename+'.avi', cv.CV_FOURCC('X', 'V', 'I', 'D'), fps, size, True)
                            cimg = f['imgs']['cimgs']
                        elif j == 1:
                            size = (512, 424)
                            video = cv.CreateVideoWriter(savepath+subfolder+'_'+filename+'_bdidx.avi', cv.CV_FOURCC('X', 'V', 'I', 'D'), fps, size, True)
                            cimg = f['imgs']['bdimgs']
                        else:
                            size = (512, 424)
                            video = cv.CreateVideoWriter(savepath+subfolder+'_'+filename+'_d.avi', cv.CV_FOURCC('X', 'V', 'I', 'D'), fps, size, True)
                            cimg = f['imgs']['dimgs']
                        for i in cimg.keys():
                            bitmap = cv.CreateImageHeader(size, cv.IPL_DEPTH_8U, 3)
                            if j == 2:
                                cv.SetData(bitmap, np.uint8(cimg[i][:]/256.).tostring(), np.uint8(cimg[i][:]/256.).dtype.itemsize*3*cimg[i][:].shape[1])
                            else:
                                cv.SetData(bitmap, cimg[i][:].tostring(), cimg[i][:].dtype.itemsize*3*cimg[i][:].shape[1])
                            cv.WriteFrame(video, bitmap)
                        del video
                else:
                    print('Error !!')
