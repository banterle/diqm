#
#Copyright (C) 2023 ISTI-CNR
#Licensed under the BSD 3-Clause Clear License (see license.txt)
#

import os
import sys
import argparse
from model import DIQMModel
from util import read_img_cv2

if __name__ == '__main__':
    os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

    parser = argparse.ArgumentParser(description='Eval Q regressor', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', type=str, help='HDR_COMP (JPEG-XT compression), HDR_ITMO (inverse tone mapping), SDR (distortions for 8-bit images), and and SDR_TMO (tone mapping distortions).')
    parser.add_argument('-src', type=str, help='Reference image')
    parser.add_argument('-dst', type=str, help='Distorted image')
    parser.add_argument('-dr', '--display_referred', type=str, default='yes', help='Do we need to apply the display? (yes/no)')
    parser.add_argument('-cs', '--colorspace', type=str, default='REC709', help='Color space of the input images')

    args = parser.parse_args()
        
    model = DIQMModel(args.mode, colorspace = args.colorspace, display_referred = args.display_referred)
        
    if (args.mode != 'SDR') and (args.mode != 'HDR_COMP') and (args.mode != 'HDR_ITMO') and (args.mode != 'SDR_TMO'):
        print('The mode ' + args.mode + ' selected is not supported.')
        print('Supported modes: HDR_ITMO, HDR_COMP, SDR, and SDR_TMO.')
        sys.exit()

    p_model = float(model.predict(args.src, args.dst))
    print(args.dst + " Q: " + str(round(p_model * 10000)/100))

    del model