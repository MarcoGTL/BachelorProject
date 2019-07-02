"""
Author: Marco

Compares whether the ground truth pixel and result pixel are the same(True) or different(False)
"""
def compare_pixel(gtpixel,resultpixel):
    if (gtpixel[0] == 255 and  gtpixel[1] == 255 and gtpixel[2] == 255 and resultpixel == 1) or (gtpixel[0] == 0 and gtpixel[1] == 0 and gtpixel[2] == 0 and  resultpixel == 0):
        return True
    else:
        return False
