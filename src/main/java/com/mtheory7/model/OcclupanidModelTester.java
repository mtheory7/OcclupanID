package com.mtheory7.model;

import com.mtheory7.utils.PadToSquare;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.transform.PipelineImageTransform;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.common.primitives.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.stream.Stream;

import static com.mtheory7.Main.CLASS_LABELS;

public class OcclupanidModelTester {
    protected static final Logger log = LoggerFactory.getLogger(OcclupanidModelTester.class);

    public void predictImages(String modelPath, String imagesPath) throws Exception {
        log.info("Loading saved model from .zip...");
        ComputationGraph model = ComputationGraph.load(new File(modelPath), false);

        log.info("Configuring image preprocessor...");
        PipelineImageTransform inferenceTransform = new PipelineImageTransform(List.of(
                new Pair<>(new PadToSquare(), 1.0)
        ), true);
        NativeImageLoader loader = new NativeImageLoader(224, 224, 3, inferenceTransform);
        DataNormalization scaler = new VGG16ImagePreProcessor();

        int errorCount = 0;
        int totalCount = 0;
        int logThreshold = 80;

        log.info("Scanning directory: {}", imagesPath);
        log.info("Logging any images that are incorrect or less than {}% confident...", logThreshold);
        try (Stream<Path> paths = Files.walk(Paths.get(imagesPath))) {
            List<Path> imagePaths = paths
                    .filter(Files::isRegularFile)
                    .filter(p -> p.toString().toLowerCase().endsWith(".jpg") || p.toString().toLowerCase().endsWith(".png"))
                    .toList();
            for (Path imagePath : imagePaths) {
                totalCount++;
                String actualLabel = imagePath.getParent().getFileName().toString();
                INDArray imageMatrix = loader.asMatrix(imagePath.toFile());
                scaler.transform(imageMatrix);

                INDArray[] output = model.output(false, imageMatrix);

                INDArray predictions = output[0];
                int predictedIndex = predictions.argMax(1).getInt(0);
                String predictedLabel = CLASS_LABELS.get(predictedIndex);
                double confidence = predictions.getDouble(predictedIndex) * 100.0;

                if (!actualLabel.equals(predictedLabel)) {
                    errorCount++;
                    log.info("Incorrect: File '{}' | Actual: '{}' | Predicted: '{}' (Confidence: {}%)",
                            imagePath.getFileName(),
                            actualLabel,
                            predictedLabel,
                            String.format("%.2f", confidence));
                } else {
                    if (confidence < logThreshold) {
                        log.info("Correct: File '{}' | Actual: '{}' | Predicted: '{}' (Confidence: {}%)",
                                imagePath.getFileName(),
                                actualLabel,
                                predictedLabel,
                                String.format("%.2f", confidence));
                    }
                }
            }
        }
        log.info("Verification complete. Scanned {} images. Found {} incorrect predictions.", totalCount, errorCount);
    }
}
