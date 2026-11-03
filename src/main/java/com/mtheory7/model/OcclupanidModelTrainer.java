package com.mtheory7.model;

import com.mtheory7.listener.ProgressEpochListener;
import com.mtheory7.utils.PadToSquare;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.CropImageTransform;
import org.datavec.image.transform.FlipImageTransform;
import org.datavec.image.transform.PipelineImageTransform;
import org.datavec.image.transform.RotateImageTransform;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.saver.LocalFileGraphSaver;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.termination.ScoreImprovementEpochTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingGraphTrainer;
import org.deeplearning4j.nn.conf.layers.GlobalPoolingLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.FileStatsStorage;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.model.SqueezeNet;
import org.nd4j.common.primitives.Pair;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static com.mtheory7.Main.CLASS_LABELS;
import static com.mtheory7.Main.MODEL_NAME;

public class OcclupanidModelTrainer {
    public static final int MAX_EPOCHS = 100;
    protected static final Logger log = LoggerFactory.getLogger(OcclupanidModelTrainer.class);
    private static final int HEIGHT = 224;
    private static final int WIDTH = 224;
    private static final int CHANNELS = 3;
    private static final int BATCH_SIZE = 16;
    private static final long SEED = 12345L;

    public void trainModelAndSave(String modelPath, String dataPath) throws IOException {
        log.info("Configuring aggressive native memory garbage collection...");
        Nd4j.getMemoryManager().togglePeriodicGc(true);
        Nd4j.getMemoryManager().setAutoGcWindow(5000);

        String modelCheckpointPath = modelPath + "/checkpoints";
        if (!new File(modelCheckpointPath).exists()) //noinspection ResultOfMethodCallIgnored
            new File(modelCheckpointPath).mkdirs();
        String modelSavePath = modelPath + "/" + MODEL_NAME;

        UIServer uiServer = UIServer.getInstance();
        File statsFile = new File(modelPath, "training_stats.dl4j");
        StatsStorage statsStorage = new FileStatsStorage(statsFile);
        uiServer.attach(statsStorage);

        File trainingDataDirectory = new File(dataPath);
        FileSplit fileSplit = new FileSplit(trainingDataDirectory, NativeImageLoader.ALLOWED_FORMATS, new Random(123));
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        BalancedPathFilter pathFilter = new BalancedPathFilter(
                new Random(SEED),
                NativeImageLoader.ALLOWED_FORMATS,
                labelMaker
        );
        PipelineImageTransform trainTransform = new PipelineImageTransform(Arrays.asList(
                new Pair<>(new PadToSquare(), 1.0),
                new Pair<>(new FlipImageTransform(1), 0.5),
                new Pair<>(new RotateImageTransform(15), 0.3),
                new Pair<>(new CropImageTransform(10), 0.3)
        ), true);
        PipelineImageTransform valTransform = new PipelineImageTransform(List.of(
                new Pair<>(new PadToSquare(), 1.0)
        ), true);
        InputSplit[] splits = fileSplit.sample(pathFilter, 60, 20, 20);
        InputSplit trainSplit = splits[0];
        InputSplit valSplit = splits[1];
        InputSplit testSplit = splits[2];
        ImageRecordReader trainRR = new ImageRecordReader(HEIGHT, WIDTH, CHANNELS, labelMaker);
        ImageRecordReader valRR = new ImageRecordReader(HEIGHT, WIDTH, CHANNELS, labelMaker);
        ImageRecordReader testRR = new ImageRecordReader(HEIGHT, WIDTH, CHANNELS, labelMaker);
        trainRR.initialize(trainSplit, trainTransform);
        valRR.initialize(valSplit, valTransform);
        testRR.initialize(testSplit, valTransform);
        log.info("Detected training labels: {}", trainRR.getLabels());
        DataSetIterator trainIter = new RecordReaderDataSetIterator(trainRR, BATCH_SIZE, 1, CLASS_LABELS.size());
        DataSetIterator valIter = new RecordReaderDataSetIterator(valRR, BATCH_SIZE, 1, CLASS_LABELS.size());
        DataSetIterator testIter = new RecordReaderDataSetIterator(testRR, BATCH_SIZE, 1, CLASS_LABELS.size());

        DataNormalization scaler = new VGG16ImagePreProcessor();
        trainIter.setPreProcessor(scaler);
        valIter.setPreProcessor(scaler);
        testIter.setPreProcessor(scaler);

        log.info("Loading pre-trained SqueezeNet weights from ImageNet...");
        SqueezeNet zooModel = SqueezeNet.builder().build();
        ComputationGraph pretrainedNet = zooModel.initPretrained(PretrainedType.IMAGENET);
        log.info(pretrainedNet.summary());

        log.info("Configuring Transfer Learning...");
        FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
                .updater(new Adam(5e-4))
                .weightDecay(1e-3)
                .seed(SEED)
                .build();

        log.info("Building ComputationGraph...");
        ComputationGraph model = new TransferLearning.GraphBuilder(pretrainedNet)
                .fineTuneConfiguration(fineTuneConf)
                .setFeatureExtractor("fire7_concat")
                .removeVertexAndConnections("conv10")
                .removeVertexAndConnections("relu10")
                .removeVertexAndConnections("global_average_pooling2d_5")
                .removeVertexAndConnections("loss")
                .addLayer("gap", new GlobalPoolingLayer.Builder(PoolingType.AVG).build(), "drop9")
                .addLayer("predictions",
                        new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                .nIn(512)
                                .nOut(CLASS_LABELS.size())
                                .weightInit(org.deeplearning4j.nn.weights.WeightInit.XAVIER)
                                .activation(Activation.SOFTMAX)
                                .build(),
                        "gap")
                .setOutputs("predictions")
                .build();

        model.setListeners(
                new ScoreIterationListener(50),
                new StatsListener(statsStorage),
                new ProgressEpochListener()
        );

        log.info("Configuring Early Stopping...");
        EarlyStoppingConfiguration<ComputationGraph> esConf = new EarlyStoppingConfiguration.Builder<ComputationGraph>()
                .epochTerminationConditions(
                        new MaxEpochsTerminationCondition(MAX_EPOCHS),
                        new ScoreImprovementEpochTerminationCondition(20)
                )
                .scoreCalculator(new DataSetLossCalculator(valIter, true))
                .evaluateEveryNEpochs(1)
                .modelSaver(new LocalFileGraphSaver(modelCheckpointPath))
                .build();

        EarlyStoppingGraphTrainer trainer = new EarlyStoppingGraphTrainer(esConf, model, trainIter);

        log.info("Starting network training with Early Stopping...");
        EarlyStoppingResult<ComputationGraph> result = trainer.fit();

        log.info("Training complete.");
        log.info("Termination reason: {}", result.getTerminationReason());
        log.info("Total epochs: {}", result.getTotalEpochs());
        log.info("Best epoch number: {}", result.getBestModelEpoch());
        log.info("Score at best epoch: {}", result.getBestModelScore());

        ComputationGraph bestModel = result.getBestModel();

        log.info("Evaluating best model accuracy using testing data directory...");
        Evaluation eval = bestModel.evaluate(testIter);
        log.info(eval.stats());

        File modelSaveFile = new File(modelSavePath);
        bestModel.save(modelSaveFile, true);
        log.info("Model successfully exported to: {}", modelSaveFile.getAbsolutePath());
    }
}