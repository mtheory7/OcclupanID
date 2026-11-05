package com.mtheory7.utils;

import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.opencv.opencv_core.Mat;
import org.datavec.image.data.ImageWritable;
import org.datavec.image.transform.BaseImageTransform;

import java.util.Random;

import static org.bytedeco.opencv.global.opencv_imgproc.COLOR_BGR2GRAY;
import static org.bytedeco.opencv.global.opencv_imgproc.COLOR_GRAY2BGR;
import static org.bytedeco.opencv.global.opencv_imgproc.cvtColor;

public class Greyscale3ChannelTransform extends BaseImageTransform<Mat> {

    private static final OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();

    public Greyscale3ChannelTransform() {
        super(null);
    }

    @Override
    protected ImageWritable doTransform(ImageWritable image, Random random) {
        if (image == null) {
            return null;
        }

        Mat mat = converter.convert(image.getFrame());
        Mat grayMat = new Mat();
        Mat bgrGrayMat = new Mat();

        // Convert to single channel gray, then expand back to 3 channels (R=G=B)
        cvtColor(mat, grayMat, COLOR_BGR2GRAY);
        cvtColor(grayMat, bgrGrayMat, COLOR_GRAY2BGR);

        return new ImageWritable(converter.convert(bgrGrayMat));
    }
}