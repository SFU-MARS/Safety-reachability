import tensorflow as tf
import tensorflow.contrib.eager as tfe
import matplotlib.pyplot as plt
import os
import utils
from tensorflow.python.util.tf_export import tf_export

import skimage.io
import skimage.segmentation
import matplotlib.pyplot as plt
import numpy as np


class TrainerHelper(object):
    
    def __init__(self, params):
        self.p = params.trainer
        self.session_dir = params.session_dir

    def train(self, model, data_source, callback_fn=None):
        """
        Train a given model.
        """
        # Create the optimizer
        self.optimizer = self.create_optimizer()
        
        # Compute the total number of training samples
        num_training_samples = self.p.batch_size * int((self.p.training_set_size * self.p.num_samples) //
                                                       self.p.batch_size)
        
        # Keep a track of the performance
        epoch_performance_training = []
        epoch_performance_validation = []

        # Instantiate one figure and axis object for plotting losses over training
        self.losses_fig = plt.figure()
        self.losses_ax = self.losses_fig.add_subplot(111)
        init = tf.global_variables_initializer()
    # with tf.Session() as sess:
    #     sess.run(init)
        # Begin the training
        for epoch in range(self.p.num_epochs):
            # Shuffle the dataset
            data_source.shuffle_datasets()

            # Define the metrics to keep a track of average loss over the epoch.
            training_loss_metric = tfe.metrics.Mean()
            validation_loss_metric = tfe.metrics.Mean()
            # epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

            # For loop over the training samples
            for j in range(0, num_training_samples, self.p.batch_size):
                training_batch1 = {}
                validation_batch1 = {}
                # Get a training and a validation batch


                training_batch = data_source.generate_training_batch(j)
                # plt.imshow(training_batch['image'][0].astype(np.int32))
                # plt.grid(False)
                # plt.show()


                validation_batch = data_source.generate_validation_batch()
                # plt.imshow(validation_batch['image'][0].astype(np.int32))
                # plt.grid(False)
                # plt.show()

                with tf.GradientTape() as tape:

                    tape.watch(model.get_trainable_vars())
                    # tape.watch(training_batch)

                    loss = model.compute_loss_function(training_batch, is_training=True, return_loss_components=False)

                    # behavior during training versus inference (e.g. Dropout).

                    # tape.watch(loss)
                # Take an optimization step

                # [var.name for var in tape.watched_variables()]
                grads = tape.gradient(loss, model.get_trainable_vars())
                # for grad in grads:
                #     if grad == None:
                #         grad = tf.constant([0])
                #     grads1.append(grad)



                self.optimizer.apply_gradients(zip(grads, model.get_trainable_vars()),
                                               global_step=tf.train.get_or_create_global_step())
                # epoch_accuracy.update_state(accuracy)
                # Record the average loss for the training and the validation batch
                self.record_average_loss_for_batch(model, training_batch, validation_batch, training_loss_metric,
                                                   validation_loss_metric)

            # Do all the things required at the end of epochs including saving the checkpoints

            # images=training_batch['image']
            # from lime import lime_image
            # explainer = lime_image.LimeImageExplainer()
            # explanation = explainer.explain_instance(
            #     images[0].astype('double'),
            #     model.predict_nn_output(is_training=True),
            #     top_labels=5,
            #     hide_color=0,
            #     num_samples=1)
            # temp, mask = explanation.get_image_and_mask(
            #     explanation.top_labels[0],
            #     positive_only=True,
            #     num_features=5,
            #     hide_rest=True)
            # plt.imshow(skimage.segmentation.mark_boundaries(temp / 2 + 0.5, mask))
            # # Select the same class explained on the figures above.
            # ind = explanation.top_labels[0]
            # # Map each explanation weight to the corresponding superpixel
            # dict_heatmap = dict(explanation.local_exp[ind])
            # heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)
            # # Plot. The visualization makes more sense if a symmetrical colorbar is used.
            # plt.imshow(heatmap, cmap='RdBu', vmin=-heatmap.max(), vmax=heatmap.max())
            # plt.colorbar()
            # plt.show()
            epoch_performance_training.append(training_loss_metric.result().numpy())
            epoch_performance_validation.append(validation_loss_metric.result().numpy())
            self.finish_epoch_processing(epoch+1, epoch_performance_training, epoch_performance_validation, model,
                                         callback_fn)
            
    def restore_checkpoint(self, model):
        """
        Load a given checkpoint.
        """
        # Create a checkpoint
        self.checkpoint = tfe.Checkpoint(optimizer=self.create_optimizer(), model=model.arch)
        
        # Restore the checkpoint
        self.checkpoint.restore(self.p.ckpt_path)
    
    def save_checkpoint(self, epoch, model):
        """
        Create and save a checkpoint.
        """
        # Create the checkpoint directory if required
        self.ckpt_dir = os.path.join(self.session_dir, 'checkpoints')
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
            self.checkpoint = tfe.Checkpoint(optimizer=self.optimizer, model=model.arch)
           
            # Note: This allows the user to specify how many checkpoints should be saved.
            # Tensorflow does not expose the parameter in tfe.Checkpoint for max_to_keep,
            # however under the hood it uses a Saver object so we can hack around this.
            from tensorflow.python.training.saver import Saver
            default_args = list(Saver.__init__.__code__.co_varnames)
            default_values = list(Saver.__init__.__defaults__)
            if 'self' in default_args:
                # Subtract one since default_values has no value for 'self'
                idx = default_args.index('max_to_keep') - 1
                default_values[idx] = self.p.max_num_ckpts_to_keep
                Saver.__init__.__defaults__ = tuple(default_values)
            else:
                assert(False)

        # Save the checkpoint
        if epoch % self.p.ckpt_save_frequency == 0:
            self.checkpoint.save(os.path.join(self.ckpt_dir, 'ckpt'))
        else:
            return
    
    def create_optimizer(self):
        """
        Create an optimizer for the training and initialize the learning rate variable.
        """
        self.lr = tfe.Variable(self.p.lr, dtype=tf.float64)
        # @tf_export(v1=["train.exponential_decay"])
        # self.lr = tfe.Variable(tf.compat.v1.train.cosine_decay(self.p.lr, 100, 100, alpha=0.0, name=None))


        return self.p.optimizer(learning_rate=self.lr)
    
    def record_average_loss_for_batch(self, model, training_batch, validation_batch, training_loss_metric,
                                      validation_loss_metric):
        """
        Record the average loss for the batch and update the metric.
        """
        regn_loss_training, prediction_loss_training, _ = model.compute_loss_function(training_batch, is_training=False,return_loss_components=True)
        regn_loss_validation, prediction_loss_validation, _ = model.compute_loss_function(validation_batch,is_training=False,return_loss_components=True)
        # Now add the loss values to the metric aggregation
        training_loss_metric(prediction_loss_training)
        validation_loss_metric(prediction_loss_validation)

    def finish_epoch_processing(self, epoch, epoch_performance_training, epoch_performance_validation, model,
                                callback_fn=None):
        """
        Finish the epoch processing for example recording the average epoch loss for the training and the validation
        sets, save the checkpoint, adjust learning rates, hand over the control to the callback function etc.
        """
        # Print the average loss for the last epoch
        print('Epoch %i: training loss %0.3f, validation loss %0.3f' % (epoch, epoch_performance_training[-1],
                                                                        epoch_performance_validation[-1]))
        
        # Plot the loss curves
        self.plot_training_and_validation_losses(epoch_performance_training, epoch_performance_validation)
        
        # Update the learning rate
        self.adjust_learning_rate(epoch)
        
        # Save the checkpoint
        self.save_checkpoint(epoch, model)
        
        # Pass the control to the callback function
        if callback_fn is not None:
            callback_fn(locals())

    def adjust_learning_rate(self, epoch):
        """
        Adjust the learning rates.
        """
        if self.p.learning_schedule == 1:
            # No adjustment is necessary
            return
        elif self.p.learning_schedule == 2:
            # Decay the learning rate by the decay factor after every few epochs
            if epoch % self.p.lr_decay_frequency == 0:
                self.lr.assign(self.lr * self.p.lr_decay_factor)
            else:
                return
        else:
            raise NotImplementedError

    def plot_training_and_validation_losses(self, training_performance, validation_performance):
        """
        Plot the loss curves for the training and the validation datasets over epochs.
        """
        fig = self.losses_fig
        ax = self.losses_ax
        ax.clear()

        ax.plot(training_performance, 'r-', label='Training')
        ax.plot(validation_performance, 'b-', label='Validation')
        ax.legend()
        fig.savefig(os.path.join(self.session_dir, 'loss_curves.pdf'))
