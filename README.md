# OcclupanID
OcclupanID is a Java based neural network designed to classify images of occlupanid to their genus/species. It uses [SqueezeNet](https://github.com/forresti/SqueezeNet) and [Deeplearning4j](https://deeplearning4j.konduit.ai/) along with (currently) [1800 images](https://github.com/mtheory7/OcclupanID/tree/main/trainingData/data) (150 images for 12 classes) to train.

Currently, the model is very accurate - validation during training reports >98% accuracy - running the final model on all 1800 images only misclassifies 5 images - training time for most recent version took ~7.14 hours to complete.

Genus/species names come from the amazing [HORG - The Holotypic Occlupanid Research Group](https://www.horg.com/horg/)

Genus/species currently supported, more to come:
- Admordorsum floridensis
- Admordorsum grongingensis
- Aspericardis lehmeri
- Concaviceps scobifolium
- Genevertonicus quimbyi
- Neotessera vitripalpus
- Palpatophora glyphodorsalis
- Palpatophora utiliformis
- Palpatophora utiliformis grandis
- Porrectofrontus mechadeus
- Unidentia oblongaris
- Unidentia obtusamaxilla azureus

Improvements to come...
- ~~Remove hard-coded String paths to /data, /models, etc.~~
- ~~Make JAR easily runnable using properties file or JSON configuration~~
- Support for more occlupanid genus/species
- Support for CUDA-compatible training (for AWS, local GPUs, etc.)