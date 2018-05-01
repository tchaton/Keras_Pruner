
### IMPORTS
import sys
import os
#try:
#    del os.environ["CUDA_VISIBLE_DEVICES"]
#except:
#    pass
#os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement=True
session = tf.Session(config=config)
sys.path.append('../')
sys.path.append('../../')
import argparse
import subprocess, os, sys

### CUSTOM IMPORTS
from nn_test.data import load_data
from nn.cnn_clf import CNN_Classifier
from pruning.nvidia_pruning.transform import Transformer
from pruning.nvidia_pruning.wrappers import Conv2D_Masked
from pruning.nvidia_pruning.pruner import NVIDIA_Trainer
from pruning.nvidia_pruning.custom_logger import get_logger

def info(msg):
    print('[INFO] '+str(msg))

def core_pruning(args):
    ### LOAD PIPELINE
    train_pipe, test_pipe = load_data()

    ### CREATE MODEL
    # Define params
    model_type= args.model_type
    #nb_train = 11990
    #nb_test = 7700
    #batch_size = int(args.batch_size)
    info('CREATE MODEL')
    # Define model
    cnn = CNN_Classifier(model_type=model_type, include_top=False, weights='imagenet')
    cnn.create_clf(19, symbolic=True, is_dropout=False)
    cnn.compile('yellowfin', loss='categorical_crossentropy', metrics=['accuracy'])

    info('LOAD MODEL WEIGHTS')
    # Load weights
    cnn.load_weights('vgg16_dropout_3.h5')

    info('CREATE TRANSFORMER')
    # Convert Model Into a Wrapped Model
    transformer = Transformer(cnn._convNet.model)

    info('WRAP CONVOLUTIONAL LAYERS')
    seq = transformer.convert(Conv2D_Masked)

    info('CREATE NVIDIA_PRUNER')
    # Create pruner
    nvidia_trainer = NVIDIA_Trainer(
                        seq,
                        'vgg16_0.85_single',
                        'vgg16',
                        'adadelta',
                        19,
                        )

    info('SET NVIDIA_PRUNER WEIGHTS_SHAPE')
    # Set shapes (could be done inside the pruner)
    nvidia_trainer.set_shapes(transformer.w_shapes)

    info('SET NVIDIA_PRUNER PIPELINES')
    # Set the pipelines
    nvidia_trainer.set_pipelines(train_pipe, test_pipe)

    info('INIT PRUNING')
    # Init the pruning (LOADING TEST / EVALUATE ONCE)
    nvidia_trainer.init_pruning()

    info('LAUNCH PRUNING')
    # Launch pruning
    nvidia_trainer.start_pruning(nb_iterations=100, nb_epochs=1, incremental_pruned=1, batch_size=32, with_debug=False, limit_per_class=2)

def parse_args():
    parser = argparse.ArgumentParser(description='Process inputs.')
    parser.add_argument('--model_type', type=str, help='This will define the core model2use')
    parser.add_argument('--batch_size', help='This will define the batch_size to be used inside the generator')

    args = parser.parse_args()
    print(args)
    return args

def create_logger():
    l = len(os.listdir('logs/'))
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)
    tee = subprocess.Popen(["tee", "logs/"+ str(l+1) +".txt"], stdin=subprocess.PIPE)
    os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
    os.dup2(tee.stdin.fileno(), sys.stderr.fileno())

if __name__ == '__main__':
    # CREATE A STREAM LOGGER
    create_logger()

    args = parse_args()
    core_pruning(args)
