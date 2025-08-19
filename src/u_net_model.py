"""
This module contains the functions necessary 
for building a model with a U-Net architecture.
"""
import tensorflow as tf
from keras.layers import *
from keras.models import Model 
#import keras.backend as K
from keras.losses import binary_crossentropy, dice
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

#
def conv_block(x, num_filters, block_avtivation, do_dropout, dropout_rate=0.05):
    x = Conv2D(num_filters, (3,3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation(block_avtivation)(x)

    x = Conv2D(num_filters, (3,3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation(block_avtivation)(x)
    if do_dropout:
        x = Dropout(dropout_rate)(x)

    return x
def build_model(filters, shape, block_avtivation, out_activation, do_dropout, dropout_rate):
    num_filters = filters

    inputs = Input((shape))

    skip_x = []
    x = inputs

    # encoder Unet part 
    for f in num_filters:
        x = conv_block(x, f, block_avtivation, do_dropout, dropout_rate)
        skip_x.append(x)
        x = MaxPool2D((2,2))(x)

    # bridge with 1024 filters 
    x = conv_block(x, 128, block_avtivation, do_dropout, dropout_rate)

    # prepare for the decoder
    num_filters.reverse()
    skip_x.reverse()

    # Decoder Unet part
    for i, f in enumerate(num_filters):
        x = UpSampling2D((2,2))(x)
        xs = skip_x[i]
        x = Concatenate()([x,xs])
        x = conv_block(x, f, block_avtivation, do_dropout, dropout_rate)
            
    #output
    x = Conv2D(1, (1,1), padding="same")(x)
    x = Activation(out_activation)(x) # since it is a binary classification and segmentation

    return Model(inputs, x) 

"""
def dice_loss(y_true, y_pred): #, smooth=100):
    dice = (2 * sum(y_true * y_pred)) / (sum(y_true) + sum(y_pred))
    #dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1 - dice  # Return 1 - dice loss for minimization
"""

def weighted_loss(lambda_bce=0.99, lambda_dice=0.01):
    """
    Combines binary crossentropy and dice loss into a single loss function.
    """
    def loss(y_true, y_pred):
        bce = binary_crossentropy(y_true, y_pred)
        dl = dice(y_true, y_pred)
        return lambda_bce * bce + lambda_dice * dl
    return loss
def initialize_model(parameters, show_summary=False):
    """
    Initializes the model with .
    """    
    #print(f"Filters before: {parameters['filters']}")    
    filters = parameters['filters']
    #print(f"Filters after: {filters}")
    block_activation = parameters['block activation']
    out_activation = parameters['output activation']
    do_dropout = parameters['do dropout']
    dropout_rate = parameters['dropout rate']

    shape = (512, 512, 1)    
        
    model = build_model(filters, shape, block_activation, out_activation, do_dropout, dropout_rate)
    model.summary() if show_summary else None

    return model

def display_training_process(history, model_name, lr, epochs, e_s_patience, e_s_intervention_monitor, lambda_bce, lambda_dice, optimizer_str, do_dropout, dropout_rate):
    """
    Displays the training process of the model.
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_stop = len(acc)
    epochs_arr = range(epochs_stop)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=((12), 4)) # Adjust figsize as needed
    
    # Plot accuracy on the first subplot (ax1)
    ax2.plot(epochs_arr, acc, 'm', label="Train accuracy")
    ax2.plot(epochs_arr, val_acc, 'c', label="Validation accuracy")
    ax2.set_title('Train and validation accuracy')
    ax2.set_ylim(0, 1)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend(loc='best')
    
    # Plot loss on the second subplot (ax2)
    ax1.plot(epochs_arr, loss, 'r', label="Train loss")
    ax1.plot(epochs_arr, val_loss, 'b', label="Validation loss")
    ax1.set_title('Train and validation loss')
    ax1.set_ylim(0, 1)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (%)')
    ax1.legend(loc='best')
    
    # apply %
    formatter = mtick.PercentFormatter(xmax=1.0)
    ax1.yaxis.set_major_formatter(formatter)
    ax2.yaxis.set_major_formatter(formatter)

    # title
    fig.suptitle(f"U-NET TRAINING: {model_name}" + 
                 f"\n{optimizer_str} Optimization  |  Weighted Loss: " + 
                 f"{(100*lambda_bce):.2f}% Binary Crossentropy Loss, " + 
                 f"{(100*lambda_dice):.2f}% Dice Loss" + 
                 f"\n{epochs_stop}/{epochs} Epochs " + 
                 f"(Early Stop Patience = {e_s_patience}, " + 
                 f"Monitoring '{e_s_intervention_monitor}')  |  " + 
                 f"{'with' if do_dropout else 'no'} Dropout " + 
                 f"({str(round((dropout_rate*100), 2)) + '%' if do_dropout else 'N/A'})  |  " + 
                 f"Initial Learning Rate = {lr}", fontsize=12, y=1.05)
    
    # Display the plots
    plt.tight_layout() # This helps prevent labels from overlapping
    plt.show()
def train_and_save_model(parameters, X_train, y_train, X_val, y_val, model, models_path, show_training_process=False):
    if show_training_process:
        show_t_p_verbose = 1
    else:
        show_t_p_verbose = 0

    lr = parameters['learning rate']
    batch_size = parameters['batch size']
    epochs = parameters['epochs']
    optimizer_str = parameters['optimizer title']
    lambda_bce = parameters['lambda binary crossentropy loss']
    lambda_dice = parameters['lambda dice loss']
    model_name = parameters['model name']
    r_lr_intervention_monitor = parameters['learn rate intervention monitor'] 
    plateau_patience = parameters['plateau patience']
    lr_reduction_factor = parameters['learn rate reduction factor']
    lr_floor = parameters['minimum learn rate']
    e_s_intervention_monitor = parameters['early stopping intervention monitor']
    e_s_patience = parameters['early stopping patience']

    if optimizer_str == "Adam":
        opt = tf.keras.optimizers.Adam(lr)
    elif optimizer_str == "SGD":
        opt = tf.keras.optimizers.SGD(lr)
    elif optimizer_str == "RMSprop":
        opt = tf.keras.optimizers.RMSprop(lr)

    model.compile(loss=weighted_loss(lambda_bce, lambda_dice), optimizer=opt , metrics=['accuracy'])

    stepsPerEpoch = int(np.ceil(len(X_train) / batch_size))
    validationSteps = int(np.ceil(len(X_val) / batch_size))

    best_model_file_name = model_name + '.keras' #model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`
    best_model_file_path = models_path+best_model_file_name

    callbacks = [
        ModelCheckpoint(best_model_file_path, verbose=show_t_p_verbose , save_best_only=True),
        ReduceLROnPlateau(monitor=r_lr_intervention_monitor, patience=plateau_patience, factor=lr_reduction_factor , verbose=show_t_p_verbose, min_lr=lr_floor),
        EarlyStopping(monitor=e_s_intervention_monitor, patience=e_s_patience, verbose=show_t_p_verbose)
    ]

    history = model.fit(
        X_train, 
        y_train, 
        batch_size=batch_size,
        epochs=epochs,
        verbose=show_t_p_verbose,
        validation_data = (X_val, y_val),
        validation_steps = validationSteps,
        steps_per_epoch = stepsPerEpoch,
        shuffle = True,
        callbacks = callbacks
    )

    do_dropout = parameters['do dropout']
    dropout_rate = parameters['dropout rate']

    filters = parameters['filters']
    filters.reverse()

    if show_training_process:
        print(f"Training process for model '{model_name}' completed.")
        print(f"Best model saved to: {best_model_file_path}")
        print("\n" + "="*50 + "\n")
        print(f"Filters: {filters}\n")
        print(f"Activation Function at each Convolutional Block: {parameters['block activation']}")
        print(f"Activation Function at Output: {parameters['output activation']}\n")
        print(f"Batch Size: {batch_size}")
        print(f"Steps per Epoch: {stepsPerEpoch}")
        print(f"Validation Steps: {validationSteps}\n")
        print(f"Monitored {r_lr_intervention_monitor} for plateau (Patience: {plateau_patience})")
        print(f"\t(on plateau, reduced learn rate by a factor of {lr_reduction_factor}, to a minimum of {lr_floor})\n") 
        print("Displaying Train / Validation Accuracy & Loss Plots:")
        display_training_process(
            history, 
            model_name, 
            lr, 
            epochs, 
            e_s_patience, 
            e_s_intervention_monitor, 
            lambda_bce, 
            lambda_dice, 
            optimizer_str, 
            do_dropout, 
            dropout_rate
            )
    return history



    """
    steps per epoch
    validation steps

    """

def load_model(model_path):
    """
    Loads a model from the specified path.
    """
    try:
        model = tf.keras.models.load_model(model_path, custom_objects={'weighted_loss': weighted_loss})
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None