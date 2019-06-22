def compare_pixel(gtpixel,resultpixel):
    if (gtpixel[0] == gtpixel[1] == gtpixel[2] == 255 and resultpixel) or (gtpixel[0] == gtpixel[1] == gtpixel[2] == 0 and not resultpixel):
        return True
    else:
        return False
