import copy

import tensorflow.python.keras.layers as layers
from keras import backend as K
from keras.layers import Dense, GlobalAveragePooling2D, Multiply
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.python.keras import activations, backend
from tensorflow.python.keras.applications import imagenet_utils
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.utils import data_utils, layer_utils

DEFAULT_BLOCKS_ARGS = [{
    'filters_in': 32,
    'filters_out': 16,
    'expand_ratio': 1,
    'bn_momentum': 0.9,
    'drop_ratio': 0.2,
    'use_se': False,
    'se_ratio': 0,
}, {
    'filters_in': 16,
    'filters_out': 27,
    'expand_ratio': 6,
    'bn_momentum': 0.9,
    'drop_ratio': 0.2,
    'use_se': False,
    'se_ratio': 0,
}, {
    'filters_in': 27,
    'filters_out': 38,
    'expand_ratio': 6,
    'bn_momentum': 0.9,
    'drop_ratio': 0.2,
    'use_se': False,
    'se_ratio': 0,
}, {
    'filters_in': 38,
    'filters_out': 50,
    'expand_ratio': 6,
    'bn_momentum': 0.9,
    'drop_ratio': 0.2,
    'use_se': True,
    'se_ratio': 12,
}, {
    'filters_in': 50,
    'filters_out': 61,
    'expand_ratio': 6,
    'bn_momentum': 0.9,
    'drop_ratio': 0.2,
    'use_se': True,
    'se_ratio': 12,
}, {
    'filters_in': 61,
    'filters_out': 72,
    'expand_ratio': 6,
    'bn_momentum': 0.9,
    'drop_ratio': 0.2,
    'use_se': True,
    'se_ratio': 12,
}, {
    'filters_in': 72,
    'filters_out': 84,
    'expand_ratio': 6,
    'bn_momentum': 0.9,
    'drop_ratio': 0.2,
    'use_se': True,
    'se_ratio': 12,
}, {
    'filters_in': 84,
    'filters_out': 95,
    'expand_ratio': 6,
    'bn_momentum': 0.9,
    'drop_ratio': 0.2,
    'use_se': True,
    'se_ratio': 12,
}, {
    'filters_in': 95,
    'filters_out': 106,
    'expand_ratio': 6,
    'bn_momentum': 0.9,
    'drop_ratio': 0.2,
    'use_se': True,
    'se_ratio': 12,
}, {
    'filters_in': 106,
    'filters_out': 117,
    'expand_ratio': 6,
    'bn_momentum': 0.9,
    'drop_ratio': 0.2,
    'use_se': True,
    'se_ratio': 12,
}, {
    'filters_in': 117,
    'filters_out': 120,
    'expand_ratio': 6,
    'bn_momentum': 0.9,
    'drop_ratio': 0.2,
    'use_se': True,
    'se_ratio': 12,
}, {
    'filters_in': 128,
    'filters_out': 140,
    'expand_ratio': 6,
    'bn_momentum': 0.9,
    'drop_ratio': 0.2,
    'use_se': True,
    'se_ratio': 12,
}, {
    'filters_in': 140,
    'filters_out': 151,
    'expand_ratio': 6,
    'bn_momentum': 0.9,
    'drop_ratio': 0.2,
    'use_se': True,
    'se_ratio': 12,
}, {
    'filters_in': 151,
    'filters_out': 162,
    'expand_ratio': 6,
    'bn_momentum': 0.9,
    'drop_ratio': 0.2,
    'use_se': True,
    'se_ratio': 12,
}, {
    'filters_in': 162,
    'filters_out': 174,
    'expand_ratio': 6,
    'bn_momentum': 0.9,
    'drop_ratio': 0.2,
    'use_se': True,
    'se_ratio': 12,
}, {
    'filters_in': 174,
    'filters_out': 185,
    'expand_ratio': 6,
    'bn_momentum': 0.9,
    'drop_ratio': 0.2,
    'use_se': True,
    'se_ratio': 12,
}]


def ReXNet(
        input_tensor=None,
        input_shape=(224, 224, 3),
        activation="swish",
        use_bias=False,
        alpha=1.0,
        se_ratio=4,
        blocks_args='default',
        include_top=True,
        weights=None,
        classes=4,
        classifier_activation='softmax',
        default_size=224,
        bn_momentum=0.9,
        pooling='avg'
):
    """ ReXNetV1 architecture.
     Reference:
     - [ReXNet: Diminishing Representational Bottleneck on Convolutional Neural //
    Network](
      https://arxiv.org/pdf/2007.00992v1.pdf) (CVPR 2020)
    Optionally loads weights pre-trained on ImageNet.
    Arguments:
    input_shape: optional shape tuple, only to be specified
          if `include_top` is False.
          It should have exactly 3 inputs channels.
    input_tensor: optional Keras tensor
          (i.e. output of `layers.Input()`)
          to use as image input for the model. 
    activation: A `str` or callable. The activation function to use
      on the "final" layer.
    use_bias : Boolean, whether the layer uses a bias vector.  
    alpha: Float between 0 and 1. controls the width of the network.
      This is known as the width multiplier in the ReXnet paper .
      - If `alpha` < 1.0, proportionally decreases the number
          of filters in each layer.
      - If `alpha` > 1.0, proportionally increases the number
          of filters in each layer.
      - If `alpha` = 1, default number of filters from the paper
          are used at each layer.
    se_ratio: float between 0 and 1, fraction to squeeze the input filters.
    include_top: whether to include the fully-connected
          layer at the top of the network.
    weights: one of `None` (random initialization),
            'imagenet' (pre-training on ImageNet),
            or the path to the weights file to be loaded.
    
    classes: Integer, optional number of classes to classify images
      into, only to be specified if `include_top` is True, and
      if no `weights` argument is specified.
    default_size: integer, default input image size.
    Returns:
    A `keras.Model` .
    Raises:
    ValueError: in case of invalid argument for `weights`, or invalid input //
    shape. 
    """
    input_shape = imagenet_utils.obtain_input_shape(
        input_shape,
        default_size=default_size,
        min_size=32,
        data_format=backend.image_data_format(),
        require_flatten=include_top,
        weights=None,
    )

    if blocks_args == 'default':
        blocks_args = DEFAULT_BLOCKS_ARGS
    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    channel_axis = 1 if backend.image_data_format() == "channels_first" else -1

    # ReXNet architecture :

    # stem layer :  (conv 3x3 , 2 stride , swish activation)
    # conBSwish_1 = conv2d_BN(
    # img_input, 32, 3, (2, 2), padding="valid", use_bias=use_bias ,bn_momentum= bn_momentum
    # )
    conv = layers.Conv2D(
        32,
        3,
        strides=2,
        padding='valid',
        use_bias=False,
        name='stem_conv')(img_input)

    convbn = layers.BatchNormalization(axis=channel_axis,
                                       momentum=bn_momentum, name='stem_bn')(conv)
    conBNswish = layers.Activation(activation, name='stem_activation')(convbn)
    # inverted_Bottleneck_block : (Expand + Depthwise + Squeeze Excitation //
    # Module (optional) + Project )
    blocks_args = copy.deepcopy(blocks_args)
    x = conBNswish

    for (i, args) in enumerate(blocks_args):
        x = inverted_residual_block(
            x,
            alpha=alpha,
            name='block_{}_'.format(i + 1),
            **args)

    # penultimate layer : (conv 1X1 , 1 stride , swish activation )
    #  AveragePooling2D + Fullyconnected
    pen_channels = 1280
    # block_17 = layers.Dropout(0.2)(block_16)
    conBNswish_2 = conv2d_BN(x, 1280, 1, (1, 1), bn_momentum=bn_momentum)
    Average_Pooling_layer = layers.AveragePooling2D((1, 1), padding="same")(
        conBNswish_2
    )
    block_17 = layers.Dropout(0.2)(Average_Pooling_layer)
    # # FC_layer = conv2d_BN(block_17, 1280, 1, (1, 1), name="conv3")
    FC_layer = layers.Conv2D(
        4,
        kernel_size=1,
        strides=1, use_bias=True,
    )(block_17)

    # Ensure that the model takes into account any potential predecessors \\
    #  of `input_tensor`.
    # FC_layer = layers.Flatten()(FC_layer)
    # x = layers.GlobalAveragePooling2D()(FC_layer)
    # x = layers.Dense(classes, activation=classifier_activation,
    #                  name='predictions')(x)
    x = layers.GlobalAveragePooling2D()(FC_layer)
    # if pooling == 'avg':
    #   x = layers.GlobalAveragePooling2D()(FC_layer)
    # elif pooling == 'max':
    #   x = layers.GlobalMaxPooling2D()(FC_layer)

    if input_tensor is not None:
        inputs = layer_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # create the model
    model = training.Model(inputs, x, name="ReXNet_V1")
    return model


def conv2d_BN(
        input_tensor,
        filter,
        kernel_size,
        strides,
        bn_momentum,
        padding="same",
        activation="swish",
        use_bias=False,
        name="top_conv",

):
    """A block that has a conv layer at shortcut.
    Arguments :
    input_tensor: input tensor
    filters: Integer, the dimensionality of the output space //
    (i.e. the number of output filters in the convolution).
    kernel_size: An integer or tuple/list of 2 integers, specifying the //
     height and width of the 2D convolution window.
    strides: 	An integer or tuple/list of 2 integers, specifying the //
    strides of the convolution along the height and width.
    padding : 	one of "valid" or "same" 
    activation : Activation function to use. If you don't specify anything,//
     no activation is applied (see keras.activations).
    use_bias : 	Boolean, whether the layer uses a bias vector.
    name  : layer name .
    Returns : 
    Output tensor for the block.
    """

    input_tensor = layers.Conv2D(
        filter,
        kernel_size,
        strides=strides,
        padding=padding,
        use_bias=use_bias,
        name=name,
    )(input_tensor)
    if not use_bias:
        bn_axis = 1 if backend.image_data_format() == "channels_first" else -1
        bn_name = None if name is None else name + "_bn"
        input_tensor = layers.BatchNormalization(
            axis=bn_axis, epsilon=1e-05, momentum=bn_momentum,
            scale=False, name=bn_name
        )(input_tensor)
    if activation is not None:
        an_name = None if name is None else name + "_ac"
        input_tensor = layers.Activation(activation, name=an_name)(
            input_tensor
        )
    return input_tensor


def inverted_residual_block(inputs,
                            alpha,
                            name='',
                            filters_in=32,
                            filters_out=16,
                            expand_ratio=1,
                            bn_momentum=0.9,
                            drop_ratio=0.2,
                            use_se=False,
                            se_ratio=0.):
    """An inverted residual block.
    Arguments:
        inputs: input tensor.
        drop_rate: float between 0 and 1, fraction of the input units to drop.
        name: string, block label.
        filters_in: integer, the number of input filters.
        filters_out: integer, the number of output filters.
        kernel_size: integer, the dimension of the convolution window.
        expand_ratio: integer, scaling coefficient for the input filters.
        se_ratio: float between 0 and 1, fraction to squeeze the input filters.
        use_se : boolean , for using squeeze and excitation block.
    Returns:
        output tensor for the block.
    """
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
    channel = int(filters_out * alpha)
    # Expansion phase
    filters = filters_in * expand_ratio
    if expand_ratio != 1:
        x = layers.Conv2D(
            filters,
            kernel_size=1,
            strides=1,
            padding='same',
            use_bias=False,
            name=name + 'expand_conv')(
            inputs)
        x = layers.BatchNormalization(axis=bn_axis, momentum=bn_momentum
                                      , name=name + 'expand_bn')(x)
        x = layers.Activation("swish", name=name + 'expand_activation')(x)
    else:
        x = inputs

    # Depthwise Convolution

    conv_pad = 'same'
    x = layers.DepthwiseConv2D(
        kernel_size=3,
        strides=1,
        padding=conv_pad,
        use_bias=False,
        name=name + 'dwconv')(x)
    x = layers.BatchNormalization(axis=bn_axis, momentum=bn_momentum
                                  , name=name + 'bn')(x)
    x = layers.Activation('relu', name=name + 'activation')(x)

    # Squeeze and Excitation phase
    if use_se:
        # filters_se = max(1, int(filters_in * se_ratio))
        nb_chan = K.int_shape(x)[-1]
        filters_se = nb_chan // se_ratio
        se = layers.GlobalAveragePooling2D(name=name + 'se_squeeze')(x)
        se = layers.Reshape((1, 1, filters), name=name + 'se_reshape')(se)
        se = layers.Conv2D(
            filters_se,
            1,
            padding='same',
            activation='relu',
            name=name + 'se_reduce')(
            se)
        se = layers.Conv2D(
            filters,
            1,
            padding='same',
            activation='sigmoid',
            name=name + 'se_expand')(se)
        x = layers.multiply([x, se], name=name + 'se_excite')

    # Output phase
    x = layers.Conv2D(
        channel,
        1,
        padding='same',
        use_bias=False,
        name=name + 'project_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, momentum=bn_momentum,
                                  name=name + 'project_bn')(x)
    if filters_in <= filters_out:
        y = layers.Conv2D(filters=filters_out, kernel_size=1)(inputs)
        y = layers.BatchNormalization(axis=bn_axis)(y)
        x += y
    return x


def ReXNetV1(ReXNetV1_config, task, train_config):
    """
    Rexnetv1 final architetcure add last layer according to the task in the //
    input .
    Arguments :
    ReXNetV1_config : ReXNetV1 config file
    task : string , regression / classification 
    returns :
    A `keras.Model` .
    """
    ReXNetV1 = ReXNet(include_top=ReXNetV1_config.INCLUDE_TOP,
                      weights=ReXNetV1_config.WEIGHTS,
                      input_tensor=None,
                      pooling=ReXNetV1_config.POOLING,
                      input_shape=(
                          ReXNetV1_config.INPUT_SHAPE.HEIGHT,
                          ReXNetV1_config.INPUT_SHAPE.WIDTH,
                          ReXNetV1_config.INPUT_SHAPE.CHANNELS
                      )
                      )

    base_model = ReXNetV1.output
    if ReXNetV1_config.DROPOUT.BOOL == True:
        base_model = layers.Dropout(ReXNetV1_config.DROPOUT.PROB)(base_model)

    if task == "regression":
        predection = layers.Dense(1, activation="linear")(base_model)
    elif task == "binary_classification" or task == "multiclass_classification":
        if train_config.ACTIVATION == 'sigmoid':
            predection = layers.Dense(1, activation="sigmoid")(base_model)
        elif train_config.ACTIVATION == 'softmax':
            predection = layers.Dense(
                train_config.CLASSES, activation="softmax"
            )(base_model)
        else:
            raise Exception("none of the activation functions  are true !")
    else:
        raise Exception("task not available!")
    model = Model(inputs=ReXNetV1.input, outputs=predection)
    return model
