This is a working Java based neural network designed to classify images of occlupanid to their genus/species. It uses SqueezeNet and Deeplearning4j along with (currently) 900 images (150 images for 6 classes) to train.

Currently, the model is very accurate - validation during training reports >99% accuracy - running the final model on all 900 images only misclassifies 1 image - training time for most recent version took ~4.5 hours to complete.

Genus/Species currently supported, more to come(6):
- Admordorsum floridensis
- Admordorsum grongingensis
- Concaviceps scobifolium
- Neotessera vitripalpus
- Palpatophora utiliformis
- Porrectofrontus mechadeus

Improvements to come...
- Remove hard-coded String paths to /data, /models, etc.
- Make easily runnable using properties file or JSON configuration
- Support for more occlupanid genus/species
