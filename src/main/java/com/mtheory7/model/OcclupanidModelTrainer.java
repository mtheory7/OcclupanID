package com.mtheory7.model;

import com.mtheory7.listener.ProgressEpochListener;
import com.mtheory7.utils.Greyscale3ChannelTransform;
import com.mtheory7.utils.PadToSquare;
import com.mtheory7.utils.SaveImageTransform;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.CollectionInputSplit;
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
import java.net.URI;
import java.util.*;

import static com.mtheory7.Main.CLASS_LABELS;
import static com.mtheory7.Main.MODEL_NAME;

public class OcclupanidModelTrainer {
    public static final int MAX_EPOCHS = 50;
    protected static final Logger log = LoggerFactory.getLogger(OcclupanidModelTrainer.class);
    private static final int EPOCH_TERMINATION_AMOUNT = 20;
    private static final double LEARNING_RATE = 1e-4;
    private static final double WEIGHT_DECAY = 1e-3;
    private static final int HEIGHT = 224;
    private static final int WIDTH = 224;
    private static final int CHANNELS = 3;
    private static final int BATCH_SIZE = 16;
    private static final long SEED = 12345L;
    private static final int K_FOLDS = 5;

    public void trainModelAndSave(String modelPath, String dataPath) throws IOException {
        File trainingDataDirectory = new File(dataPath);
        List<List<URI>> partitions = createKFoldPartitions(trainingDataDirectory, K_FOLDS, SEED);

        // Aggregator to track metrics across all folds
        List<Evaluation> foldEvaluations = new ArrayList<>();

        for (int fold = 0; fold < K_FOLDS; fold++) {
            log.info("===========================================");
            log.info("STARTING FOLD {} OF {}", fold + 1, K_FOLDS);
            log.info("===========================================");

            // 1. Assign chunks to Train, Val, Test (e.g., for K=5: 3 chunks Train, 1 Val, 1 Test)
            List<URI> testURIs = partitions.get(fold);
            List<URI> valURIs = partitions.get((fold + 1) % K_FOLDS);
            List<URI> trainURIs = new ArrayList<>();
            for (int i = 0; i < K_FOLDS; i++) {
                if (i != fold && i != (fold + 1) % K_FOLDS) {
                    trainURIs.addAll(partitions.get(i));
                }
            }

            InputSplit trainSplit = new CollectionInputSplit(trainURIs);
            InputSplit valSplit = new CollectionInputSplit(valURIs);
            InputSplit testSplit = new CollectionInputSplit(testURIs);

            // 2. Initialize RecordReaders with your existing transforms
            ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
            ImageRecordReader trainRR = new ImageRecordReader(HEIGHT, WIDTH, CHANNELS, labelMaker);
            ImageRecordReader valRR = new ImageRecordReader(HEIGHT, WIDTH, CHANNELS, labelMaker);
            ImageRecordReader testRR = new ImageRecordReader(HEIGHT, WIDTH, CHANNELS, labelMaker);

            PipelineImageTransform trainTransform = new PipelineImageTransform(Arrays.asList(
                    new Pair<>(new PadToSquare(), 1.0),
                    new Pair<>(new CropImageTransform(50), 1.0),
                    new Pair<>(new CropImageTransform(5), 0.4),
                    new Pair<>(new Greyscale3ChannelTransform(), 0.4),
                    new Pair<>(new FlipImageTransform(1), 0.4),
                    new Pair<>(new RotateImageTransform(90), 0.4)
                    //new Pair<>(new SaveImageTransform("C:/Users/thesp/Code/OcclupanID/trainingData/transformed", "final_pipeline_out", 100), 1.0)
            ), false);
            PipelineImageTransform valTransform = new PipelineImageTransform(List.of(
                    new Pair<>(new PadToSquare(), 1.0),
                    new Pair<>(new CropImageTransform(50), 1.0)
            ), false);

            trainRR.initialize(trainSplit, trainTransform); // Keep your PadToSquare/Greyscale pipeline here
            valRR.initialize(valSplit, valTransform);
            testRR.initialize(testSplit, valTransform);

            DataSetIterator trainIter = new RecordReaderDataSetIterator(trainRR, BATCH_SIZE, 1, CLASS_LABELS.size());
            DataSetIterator valIter = new RecordReaderDataSetIterator(valRR, BATCH_SIZE, 1, CLASS_LABELS.size());
            DataSetIterator testIter = new RecordReaderDataSetIterator(testRR, BATCH_SIZE, 1, CLASS_LABELS.size());

            DataNormalization scaler = new VGG16ImagePreProcessor();
            trainIter.setPreProcessor(scaler);
            valIter.setPreProcessor(scaler);
            testIter.setPreProcessor(scaler);

            UIServer uiServer = UIServer.getInstance();
            File statsFile = new File(modelPath, "training_stats.dl4j");
            StatsStorage statsStorage = new FileStatsStorage(statsFile);
            uiServer.attach(statsStorage);

            // 3. Rebuild the Model from Scratch (MUST BE INSIDE THE LOOP)
            SqueezeNet zooModel = SqueezeNet.builder().build();
            ComputationGraph pretrainedNet = zooModel.initPretrained(PretrainedType.IMAGENET);

            log.info("Configuring transfer learning...");
            FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
                    .updater(new Adam(LEARNING_RATE))
                    .weightDecay(WEIGHT_DECAY)
                    .seed(SEED)
                    .build();

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
                                    .nIn(512).nOut(CLASS_LABELS.size())
                                    .weightInit(org.deeplearning4j.nn.weights.WeightInit.XAVIER)
                                    .activation(Activation.SOFTMAX).build(),
                            "gap")
                    .setOutputs("predictions")
                    .build();

            model.setListeners(
                    new ScoreIterationListener(50),
                    new StatsListener(statsStorage),
                    new ProgressEpochListener()
            );


            // 4. Train with Early Stopping
            String foldCheckpointPath = modelPath + File.separator + "checkpoints_fold_" + fold;
            EarlyStoppingConfiguration<ComputationGraph> esConf = new EarlyStoppingConfiguration.Builder<ComputationGraph>()
                    .epochTerminationConditions(
                            new MaxEpochsTerminationCondition(MAX_EPOCHS),
                            new ScoreImprovementEpochTerminationCondition(EPOCH_TERMINATION_AMOUNT))
                    .scoreCalculator(new DataSetLossCalculator(valIter, true))
                    .evaluateEveryNEpochs(1)
                    .modelSaver(new LocalFileGraphSaver(foldCheckpointPath))
                    .build();

            EarlyStoppingGraphTrainer trainer = new EarlyStoppingGraphTrainer(esConf, model, trainIter);
            EarlyStoppingResult<ComputationGraph> result = trainer.fit();
            ComputationGraph bestModel = result.getBestModel();

            // 5. Evaluate and Store Metrics
            Evaluation eval = bestModel.evaluate(testIter);
            log.info("Fold {} Evaluation: \n{}", fold + 1, eval.stats());
            foldEvaluations.add(eval);

            // Save the best overall model (optional: save all, or save based on highest fold score)
            bestModel.save(new File(modelPath + File.separator + MODEL_NAME + "_fold_" + fold), true);
        }

        // 6. Output Averaged Cross-Validation Results
        log.info("=== K-FOLD CROSS VALIDATION COMPLETE ===");
        for (int i = 0; i < foldEvaluations.size(); i++) {
            log.info("Fold {} Accuracy: {}", i + 1, foldEvaluations.get(i).accuracy());
        }
    }

    private List<List<URI>> createKFoldPartitions(File dataDir, int k, long seed) {
        File[] classDirs = dataDir.listFiles(File::isDirectory);
        List<List<URI>> folds = new ArrayList<>();
        for (int i = 0; i < k; i++) folds.add(new ArrayList<>());

        // Stratify by class to ensure equal representation in each fold
        for (File classDir : classDirs) {
            File[] images = classDir.listFiles();
            List<URI> imageURIs = new ArrayList<>();
            if (images != null) {
                for (File img : images) imageURIs.add(img.toURI());
            }

            Collections.shuffle(imageURIs, new Random(seed));

            int foldSize = imageURIs.size() / k;
            for (int i = 0; i < imageURIs.size(); i++) {
                int foldIndex = Math.min(i / foldSize, k - 1);
                folds.get(foldIndex).add(imageURIs.get(i));
            }
        }
        return folds;
    }
}