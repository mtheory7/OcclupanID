package com.mtheory7.config;

import java.util.List;

public class OcclupanIDConfig {
    public int batchSize;
    public int batchSizeValidation;
    public int channels;
    public List<String> classNames;
    public String dataPath;
    public String dataPathTransformed;
    public int epochEarlyStop;
    public int epochMax;
    public int imageHeight;
    public int imageWidth;
    public double learningRate;
    //public double learningWeightDecay; was set to 1e-3, training works better without it
    public String modelSaveFile;
    public String modelSavePath;
    public long seed;
    public boolean uiStatsEnabled;
}