package com.mtheory7;

import com.mtheory7.model.OcclupanidModelTester;
import com.mtheory7.model.OcclupanidModelTrainer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.time.Duration;
import java.time.Instant;
import java.util.Arrays;
import java.util.List;

public class Main {
    protected static final Logger log = LoggerFactory.getLogger(Main.class);
    public static final Instant START_TIME = Instant.now();
    public static final List<String> CLASS_LABELS = Arrays.asList(
            "AdmordorsumFloridensis",
            "AdmordorsumGrongingensis",
            "ConcavicepsScobifolium",
            "NeotesseraVitripalpus",
            "PalpatophoraUtiliformis",
            "PorrectofrontusMechadeus");
    public static final String MODEL_NAME = "deeplearning4jOcclupanidModel1.2.0.zip";
    private static final String MODEL_PATH = "C:/Users/thesp/Code/deeplearning4j-occlupanid/models";
    private static final String DATA_PATH = "C:/Users/thesp/Code/deeplearning4j-occlupanid/trainingData/data";

    public static void main(String[] args) throws Exception {
        log.info("Starting trainModelAndSave function...");
        OcclupanidModelTrainer occlupanidModelTrainer = new OcclupanidModelTrainer();
        occlupanidModelTrainer.trainModelAndSave(MODEL_PATH, DATA_PATH);
        log.info("Finished trainModelAndSave function...");

        log.info("Starting model tests...");
        OcclupanidModelTester occlupanidModelTester = new OcclupanidModelTester();
        occlupanidModelTester.predictImages((MODEL_PATH + File.separator + MODEL_NAME), DATA_PATH);
        log.info("Finished testing model");

        log.info("Program finished in {} hours", ((float) (Duration.between(START_TIME, Instant.now()).toMillis()) / 1000 / 3600));
        System.exit(0);
    }
}