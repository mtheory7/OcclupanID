package com.mtheory7;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.mtheory7.config.OcclupanIDConfig;
import com.mtheory7.model.OcclupanidModelTester;
import com.mtheory7.model.OcclupanidModelTrainer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.time.Duration;
import java.time.Instant;

public class Main {
    protected static final Logger log = LoggerFactory.getLogger(Main.class);
    public static final Instant START_TIME = Instant.now();

    public static void main(String[] args) throws Exception {
        if (args.length != 2 || !args[0].equals("--config")) {
            log.error("Unexpected or missing argument(s). Please use the following argument setup: --config <config.json file path>");
            return;
        }

        File jsonFile = new File(args[1]);
        if (!jsonFile.exists()) {
            log.error("Configuration file not found: {}", jsonFile.getAbsolutePath());
            return;
        }

        OcclupanIDConfig config;
        try {
            ObjectMapper mapper = new ObjectMapper();
            config = mapper.readValue(jsonFile, OcclupanIDConfig.class);
        } catch (Exception e) {
            log.error("There was an error parsing the configuration file: {}", e.getMessage());
            return;
        }

        log.info("Starting trainModelAndSave function...");
        new OcclupanidModelTrainer().trainModelAndSave(config);
        log.info("Finished trainModelAndSave function...");

        log.info("Starting model tests...");
        new OcclupanidModelTester().predictImages(config);
        log.info("Finished testing model");

        log.info("Program finished in {} hours", ((float) (Duration.between(START_TIME, Instant.now()).toMillis()) / 1000 / 3600));
        System.exit(0);
    }
}