package com.mtheory7.model;

import com.mtheory7.config.OcclupanIDConfig;
import com.mtheory7.listener.ProgressEpochListener;
import com.mtheory7.utils.Greyscale3ChannelTransform;
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
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class OcclupanidModelTrainer {
    protected static final Logger log = LoggerFactory.getLogger(OcclupanidModelTrainer.class);

    public void trainModelAndSave(OcclupanIDConfig config) throws IOException {
        log.info("MAX_EPOCHS:{} EPOCH_TERMINATION_AMOUNT:{} LEARNING_RATE:{} HEIGHT:{} WIDTH:{} CHANNELS:{} BATCH_SIZE:{} SEED:{}",
                config.epochMax, config.epochEarlyStop, config.learningRate, //config.learningWeightDecay,
                config.imageHeight, config.imageWidth, config.channels, config.batchSize, config.seed);

        String modelCheckpointPath = config.modelSavePath + File.separator + "checkpoints";
        String modelFileName = config.modelSavePath + File.separator + config.modelSaveFile;
        Files.createDirectories(Path.of(modelCheckpointPath));

        log.info("Initializing UIServer...");
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new FileStatsStorage(new File(config.modelSavePath, "training_stats.dl4j"));
        uiServer.attach(statsStorage);

        log.info("Preparing PipelineImageTransform objects...");
        PipelineImageTransform trainTransform = new PipelineImageTransform(Arrays.asList(
                new Pair<>(new PadToSquare(), 1.0),
                new Pair<>(new CropImageTransform(50), 1.0),
                new Pair<>(new CropImageTransform(5), 0.5),
                new Pair<>(new Greyscale3ChannelTransform(), 0.5),
                new Pair<>(new FlipImageTransform(1), 0.5),
                new Pair<>(new RotateImageTransform(90), 0.5)
                //new Pair<>(new SaveImageTransform(config.dataPathTransformed, "transformedImage", 100), 1.0)
        ), false);
        log.info("trainTransform object:{}", trainTransform);
        PipelineImageTransform valTransform = new PipelineImageTransform(List.of(
                new Pair<>(new PadToSquare(), 1.0),
                new Pair<>(new CropImageTransform(50), 1.0)
        ), false);
        log.info("valTransform object:{}", valTransform);

        log.info("Splitting and preparing training data...");
        FileSplit fileSplit = new FileSplit(new File(config.dataPath), NativeImageLoader.ALLOWED_FORMATS, new Random(config.seed));
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        BalancedPathFilter pathFilter = new BalancedPathFilter(
                new Random(config.seed),
                NativeImageLoader.ALLOWED_FORMATS,
                labelMaker
        );
        InputSplit[] splits = fileSplit.sample(pathFilter, 60, 20, 20);
        InputSplit trainSplit = splits[0];
        InputSplit valSplit = splits[1];
        InputSplit testSplit = splits[2];
        log.info("trainSplit.length():{} valSplit.length():{} testSplit.length():{}", trainSplit.length(), valSplit.length(), testSplit.length());
        ImageRecordReader trainRR = new ImageRecordReader(config.imageHeight, config.imageWidth, config.channels, labelMaker);
        ImageRecordReader valRR = new ImageRecordReader(config.imageHeight, config.imageWidth, config.channels, labelMaker);
        ImageRecordReader testRR = new ImageRecordReader(config.imageHeight, config.imageWidth, config.channels, labelMaker);
        trainRR.initialize(trainSplit, trainTransform);
        valRR.initialize(valSplit, valTransform);
        testRR.initialize(testSplit, valTransform);
        log.info("Detected training labels: {}", trainRR.getLabels());
        DataSetIterator trainIterator = new RecordReaderDataSetIterator(trainRR, config.batchSize, 1, config.classNames.size());
        DataSetIterator valIterator = new RecordReaderDataSetIterator(valRR, config.batchSize, 1, config.classNames.size());
        DataSetIterator testIterator = new RecordReaderDataSetIterator(testRR, config.batchSize, 1, config.classNames.size());

        log.info("Preparing DataNormalization scaler...");
        DataNormalization scaler = new VGG16ImagePreProcessor();
        trainIterator.setPreProcessor(scaler);
        valIterator.setPreProcessor(scaler);
        testIterator.setPreProcessor(scaler);

        log.info("Loading pre-trained SqueezeNet weights from ImageNet...");
        SqueezeNet zooModel = SqueezeNet.builder().build();
        ComputationGraph pretrainedNet = zooModel.initPretrained(PretrainedType.IMAGENET);
        log.info(pretrainedNet.summary());

        log.info("Configuring transfer learning...");
        FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
                .updater(new Adam(config.learningRate))
                //.weightDecay(config.learningWeightDecay)
                .seed(config.seed)
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
                                .nOut(config.classNames.size())
                                .weightInit(org.deeplearning4j.nn.weights.WeightInit.XAVIER)
                                .activation(Activation.SOFTMAX)
                                .build(),
                        "gap")
                .setOutputs("predictions")
                .build();

        log.info("Adding listeners to model...");
        model.setListeners(
                new ScoreIterationListener(50),
                new StatsListener(statsStorage),
                new ProgressEpochListener(config.epochMax)
        );

        log.info("Configuring Early Stopping...");
        EarlyStoppingConfiguration<ComputationGraph> esConf = new EarlyStoppingConfiguration.Builder<ComputationGraph>()
                .epochTerminationConditions(
                        new MaxEpochsTerminationCondition(config.epochMax),
                        new ScoreImprovementEpochTerminationCondition(config.epochEarlyStop)
                )
                .scoreCalculator(new DataSetLossCalculator(valIterator, true))
                .evaluateEveryNEpochs(1)
                .modelSaver(new LocalFileGraphSaver(modelCheckpointPath))
                .build();
        EarlyStoppingGraphTrainer trainer = new EarlyStoppingGraphTrainer(esConf, model, trainIterator);

        log.info("Starting network training...");
        EarlyStoppingResult<ComputationGraph> result = trainer.fit();

        log.info("Training complete!");
        log.info("Termination reason: {}", result.getTerminationReason());
        log.info("Total epochs: {}", result.getTotalEpochs());
        log.info("Best epoch number: {}", result.getBestModelEpoch());
        log.info("Score at best epoch: {}", result.getBestModel().score());

        log.info("Evaluating best model accuracy using testing data directory...");
        ComputationGraph bestModel = result.getBestModel();
        Evaluation eval = bestModel.evaluate(testIterator);
        log.info(eval.stats());

        log.info("Saving best model...");
        File modelFile = new File(modelFileName);
        bestModel.save(modelFile, true);
        log.info("Model successfully exported to: {}", modelFile.getAbsolutePath());
    }
}