package com.mtheory7.listener;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.BaseTrainingListener;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.time.Duration;
import java.time.Instant;

import static com.mtheory7.Main.START_TIME;
import static com.mtheory7.model.OcclupanidModelTrainer.MAX_EPOCHS;

public class ProgressEpochListener extends BaseTrainingListener {
    protected static final Logger log = LoggerFactory.getLogger(ProgressEpochListener.class);

    public int CURRENT_EPOCH = 1;

    @Override
    public void onEpochEnd(Model model) {
        float percentage = (((float) CURRENT_EPOCH / MAX_EPOCHS) * 100);
        float totalHours = ((float) (Duration.between(START_TIME, Instant.now()).toMillis()) / 1000 / (CURRENT_EPOCH) * MAX_EPOCHS) / 3600;
        float elapsedHours = (float) (Duration.between(START_TIME, Instant.now()).toMillis()) / 1000 / 3600;
        float averageEpochTime = ((float) (Duration.between(START_TIME, Instant.now()).toMillis()) / 60000) / CURRENT_EPOCH;
        log.info("{}% complete - Hours remaining to max epoch (est): {} - Average epoch duration: {} minutes", percentage, (totalHours - elapsedHours), averageEpochTime);
        CURRENT_EPOCH++;
    }
}