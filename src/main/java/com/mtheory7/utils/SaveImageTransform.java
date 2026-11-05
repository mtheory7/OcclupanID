package com.mtheory7.utils;

import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.opencv.opencv_core.Mat;
import org.datavec.image.data.ImageWritable;
import org.datavec.image.transform.BaseImageTransform;

import java.io.File;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;

import static org.bytedeco.opencv.global.opencv_imgcodecs.imwrite;

public class SaveImageTransform extends BaseImageTransform<Mat> {

    private static final OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();
    private static final AtomicInteger COUNTER = new AtomicInteger(0);

    private final String outputDirectory;
    private final String prefix;
    private final int saveLimit;

    public SaveImageTransform(String outputDirectory, String prefix, int saveLimit) {
        super(null);
        this.outputDirectory = outputDirectory;
        this.prefix = prefix;
        this.saveLimit = saveLimit;

        // Ensure the directory exists
        new File(outputDirectory).mkdirs();
    }

    @Override
    protected ImageWritable doTransform(ImageWritable image, Random random) {
        if (image == null) {
            return null;
        }

        int id = COUNTER.getAndIncrement();

        // Only save up to the limit to prevent filling up the hard drive during training
        if (id < saveLimit) {
            Mat mat = converter.convert(image.getFrame());
            String filepath = outputDirectory + File.separator + prefix + "_" + id + ".png";
            imwrite(filepath, mat);
        }

        // Return the image completely unmodified for the network to train on
        return image;
    }
}