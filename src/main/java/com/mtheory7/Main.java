package com.mtheory7;

import com.fasterxml.jackson.databind.ObjectMapper;
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
    private static OcclupanIDConfig config;

    public static void main(String[] args) throws Exception {
        String configPath = (args.length > 0) ? args[1] : "config.json";
        File jsonFile = new File(configPath);

        if (!jsonFile.exists()) {
            System.err.println("Configuration file not found: " + jsonFile.getAbsolutePath());
            return;
        }

        try {
            ObjectMapper mapper = new ObjectMapper();
            config = mapper.readValue(jsonFile, OcclupanIDConfig.class);
            // Map JSON to your Java POJO configuration class
            // MyConfig config = mapper.readValue(jsonFile, MyConfig.class);
        } catch (Exception e) {
            e.printStackTrace();
        }

        log.info("Starting trainModelAndSave function...");
        OcclupanidModelTrainer occlupanidModelTrainer = new OcclupanidModelTrainer();
        occlupanidModelTrainer.trainModelAndSave(config);
        log.info("Finished trainModelAndSave function...");

        log.info("Starting model tests...");
        OcclupanidModelTester occlupanidModelTester = new OcclupanidModelTester();
        occlupanidModelTester.predictImages((config.modelSavePath + File.separator + config.modelSaveFile), config.dataPath, config.classNames);
        log.info("Finished testing model");

        log.info("Program finished in {} hours", ((float) (Duration.between(START_TIME, Instant.now()).toMillis()) / 1000 / 3600));
        System.exit(0);
    }
}