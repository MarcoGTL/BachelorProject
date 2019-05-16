
        # save super-pixel representation or plot it
        if args["save"]:
            slic.io.imsave('super-pixels.png', slic.mark_boundaries(img_float64, segmented_pixels))
        else:
            slic.show_plot(img_float64, segmented_pixels)

        img_uint8 = sift.read_image_as_uint8(args["image"])
        gray_uint8 = sift.to_gray_value(img_uint8)

        # Create mask if labels are given, otherwise mask=None
        mask = None
        if args["labels"]:
            mask = slic.get_mask(segmented_pixels, args["labels"])
            slic.io.imsave('mask.png', mask)

        # SIFT
        kp = sift.get_keypoints(gray_uint8, mask)
        out = copy.deepcopy(img_uint8)
        sift.draw_keypoints(img_uint8, kp, out)
        sift.draw_keypoints(img_uint8, kp, out, detailed=True)

        # Histograms
        hist = histogram.get_histogram(img_uint8, mask)
        hist_gray = histogram.get_histogram(gray_uint8, mask, colors='k')
        histogram.plot_histogram(hist)
        histogram.plot_histogram(hist_gray)

        ###########################
        def plot_cosegmentations(self, images, markings):
            plt.subplot(1, 2, 2), plt.xticks([]), plt.yticks([])
            plt.title('segmentation')
            cv2.imwrite("output_segmentation.png", segmask)
            plt.imshow(segmask)

            plt.subplot(1, 2, 1), plt.xticks([]), plt.yticks([])
            img = mark_boundaries(img, segments)
            io.imsave('superpixel.png', img)
            img[img_marking[:, :, 0] != 255] = (1, 0, 0)
            img[img_marking[:, :, 2] != 255] = (0, 0, 1)
            plt.imshow(img)
            plt.title("SLIC + markings")
            plt.savefig("segmentation.png", bbox_inches='tight', dpi=96)