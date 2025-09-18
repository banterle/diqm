#
#Copyright (C) 2023 ISTI-CNR
#Licensed under the BSD 3-Clause Clear License (see license.txt)
#

import os
import re
import glob2
import argparse
import urllib.request
from model_classic import *
from util import *
import torch

#
#
#
class DIQMModel:

    #
    #
    #
    def __init__(self, run, maxClip = 1400, colorspace = 'REC709', display_referred = 'yes', ):
        url_str = 'http://www.banterle.com/francesco/projects/diqm/'
        
        bDone = False
        
        args_mode = run
        if args_mode == 'SDR':
            try:
                model = self.setup_aux('weights/diqm_sdr.pth', maxClip, colorspace, display_referred)
            except:
                model = self.setup_aux(url_str + 'diqm_sdr.pth', maxClip, colorspace, display_referred)
            bDone = True

        if args_mode == 'HDR_COMP':
            try:
                model = self.setup_aux('weights/diqm_hdrc.pth', maxClip, colorspace, display_referred)
            except:
                model = self.setup_aux(url_str + 'diqm_hdrc.pth', maxClip, colorspace, display_referred)
            bDone = True

        if args_mode == 'HDR_ITMO':
            try:
                model = self.setup_aux('weights/diqm_itmo.pth', maxClip, colorspace, display_referred)
            except:
                model = self.setup_aux(url_str + 'diqm_itmo.pth', maxClip, colorspace, display_referred)
            bDone = True

        if args_mode == 'SDR_TMO':
            try:
                model = self.setup_aux('weights/diqm_tmo.pth', maxClip, colorspace, display_referred)
            except:
                model = self.setup_aux(url_str + 'diqm_tmo.pth', maxClip, colorspace, display_referred)
            bDone = True
            
        if bDone == False:
           self.setup_aux(run, maxClip, colorspace, display_referred)
    
    #
    #
    #
    def setup_aux(self, run, maxClip = 1400, colorspace = 'REC709', display_referred = 'yes'):
        self.run = run
        ext = os.path.splitext(run)[1]
        
        if ext == '':
            ckpt_dir = os.path.join(run, 'ckpt')
            ckpts = glob2.glob(os.path.join(ckpt_dir, '*.pth'))
            assert ckpts, "No checkpoints to resume from!"

            def get_epoch(ckpt_url):
                s = re.findall("ckpt_e(\\d+).pth", ckpt_url)
                epoch = int(s[0]) if s else -1
                return epoch, ckpt_url

            start_epoch, ckpt = max(get_epoch(c) for c in ckpts)
            print('Checkpoint:', ckpt)
        else:
            if 'http://' in run:
                cache_dir = os.path.expanduser('./weights')
                os.makedirs(cache_dir, exist_ok=True)

                filename = os.path.basename(run)

                cached_path = os.path.join(cache_dir, filename)

                if not os.path.exists(cached_path):
                    urllib.request.urlretrieve(run, cached_path)

                ckpt = cached_path

            else:
                ckpt = run

        bLoad = True

        if ckpt == 'none.pth':
            bLoad = False
                        
        if bLoad:
            if torch.cuda.is_available():
                ckpt = torch.load(ckpt, weights_only=True)
            else:
                ckpt = torch.load(ckpt, weights_only=True, map_location=torch.device('cpu'))

        model = QNet(6, 1)

        model.load_state_dict(ckpt['model'])

        if(torch.cuda.is_available()):
            model = model.cuda()

        model.eval()
        
        self.model = model

        self.colorspace = colorspace
        self.maxClip = maxClip
        self.display_referred = (display_referred == 'yes')
    
    #
    #
    #
    def getModel(self):
        return self.model

    #
    #
    #
    def load(self, fn):
        stim = read_img_cv2(fn, maxClip = self.maxClip, grayscale = False, colorspace = self.colorspace, display_referred = self.display_referred)
        stim = stim.unsqueeze(0)

        if torch.cuda.is_available():
            stim = stim.cuda()

        return stim

    #
    #
    #
    def predict(self, fn_src, fn_dst):

        img_src = self.load(fn_src)
        img_dst = self.load(fn_dst)

        stim = torch.cat((img_src, img_dst), dim=1)

        with torch.no_grad():
            out = self.model(stim)

        out = out.data.cpu().numpy().squeeze()

        return out

    #
    #
    #
    def predict_t(self, stim):
        with torch.no_grad():
             out = self.model(stim)

        return out
