import abc
import logging
import pandas
from comet_ml import Optimizer
from keras import Sequential
from keras.callbacks import TensorBoard, CSVLogger, EarlyStopping

from gin_train.trainers import KerasTrainer

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class CometMLOptimizerTrainer(KerasTrainer, metaclass=abc.ABCMeta):

    optimizer: Optimizer

    def __init__(self,
                 modelfactory,
                 optconfig,
                 train_dataset,
                 valid_dataset,
                 output_dir,
                 cometml_experiment):
        """
        Args:
          modelfactory: function symbol that is responsible for model construction (esp. injected through Gin)
          optconfig: optimization configuration format needed by Comet ML
          train: training Dataset (object inheriting from kipoi.data.Dataset)
          valid: validation Dataset (object inheriting from kipoi.data.Dataset)
          output_dir: output directory where to log the training
          cometml_experiment: the name of the CometML project
        """
        self.modelfactory = modelfactory
        self.optconfig = optconfig
        self.optimizer = Optimizer(optconfig, project_name = cometml_experiment)
        self.optimal_experiment = None
        KerasTrainer.__init__(self, Sequential(), train_dataset, valid_dataset, output_dir, cometml_experiment, wandb_run = None)

    def train(self,
              batch_size=256,
              epochs=100,
              early_stop_patience=4,
              num_workers=8,
              train_epoch_frac=1.0,
              valid_epoch_frac=1.0,
              train_samples_per_epoch=None,
              validation_samples=None,
              train_batch_sampler=None,
              tensorboard=True):
        """Train various models based on optimization config from construction.
        Args:
          batch_size:
          epochs:
          patience: early stopping patience
          num_workers: how many workers to use in parallel
          train_epoch_frac: if smaller than 1, then make the epoch shorter
          valid_epoch_frac: same as train_epoch_frac for the validation dataset
          train_batch_sampler: batch Sampler for training. Useful for say Stratified sampling
          tensorboard: if True, tensorboard output will be added
        """

        if train_batch_sampler is not None:
            train_it = self.train_dataset.batch_train_iter(shuffle=False,
                                                           batch_size=1,
                                                           drop_last=None,
                                                           batch_sampler=train_batch_sampler,
                                                           num_workers=num_workers)
        else:
            train_it = self.train_dataset.batch_train_iter(batch_size=batch_size,
                                                           shuffle=True,
                                                           num_workers=num_workers)
        next(train_it)
        valid_dataset = self.valid_dataset[0][1]  # take the first one
        valid_it = valid_dataset.batch_train_iter(batch_size=batch_size,
                                                  shuffle=True,
                                                  num_workers=num_workers)
        next(valid_it)

        if tensorboard:
            tb = [TensorBoard(log_dir=self.output_dir)]
        else:
            tb = []

        if self.wandb_run is not None:
            from wandb.keras import WandbCallback
            wcp = [WandbCallback(save_model=False)]  # we save the model using ModelCheckpoint
        else:
            wcp = []

        # train the model
        if len(valid_dataset) == 0:
            raise ValueError("len(self.valid_dataset) == 0")

        if train_samples_per_epoch is None:
            train_steps_per_epoch = max(int(len(self.train_dataset) / batch_size * train_epoch_frac), 1)
        else:
            train_steps_per_epoch = max(int(train_samples_per_epoch / batch_size), 1)

        if validation_samples is None:
            # parametrize with valid_epoch_frac
            validation_steps = max(int(len(valid_dataset) / batch_size * valid_epoch_frac), 1)
        else:
            validation_steps = max(int(validation_samples / batch_size), 1)

        for experiment in self.optimizer.get_experiments():
            self.model = construct_model_from_experiment(modelfactory=self.modelfactory, experiment = experiment,
                                                         parameters_list=self.optconfig["parameters"])

            self.model.fit_generator(
                train_it,
                epochs=epochs,
                steps_per_epoch=train_steps_per_epoch,
                validation_data=valid_it,
                validation_steps=validation_steps,
                callbacks=[
                              EarlyStopping(
                                  patience=early_stop_patience,
                                  restore_best_weights=True
                              ),
                              CSVLogger(self.history_path)
                          ] + tb + wcp
            )
            # log metrics from the best epoch
            try:
                dfh = pandas.read_csv(self.history_path)
                m = dict(dfh.iloc[dfh.val_loss.idxmin()])
                experiment.log_multiple_metrics(m, prefix="best-epoch/")
            except FileNotFoundError as e:
                logger.warning(e)

        best = (0, None)
        # get optimizer suggested parameters
        for experiment in self.optimizer.get_experiments():
            score = expscore(experiment.get_metric("loss"), experiment.get_metric("val_loss"))
            if (score > best[0]):
                best = (score, experiment)

        (_, best_experiment) = best
        logger.info("Best experiment was {}" % best_experiment.id)
        self.optimal_experiment = best_experiment


    def evaluate(self, metric, batch_size=256, num_workers=8, eval_train=False, eval_skip=(), save=True, **kwargs):
        if self.optimal_experiment == None:
            logger.error("There is no optimal experiment to assess. Run at least one training first!")
        self.model =  construct_model_from_experiment(self.modelfactory, self.optimal_experiment, self.optconfig["parameters"])
        KerasTrainer.evaluate(self, metric, batch_size, num_workers, eval_train, eval_skip, save, **kwargs)


def expscore(loss, val_loss):
    return val_loss - (val_loss - loss)*0,1

def construct_model_from_experiment(modelfactory, experiment, parameters_list):
    configdict = {experiment.get_parameter(k): k for k in parameters_list}
    try:
        return modelfactory(**configdict)  # delegate parameters to function
    except ValueError as e:
        raise ValueError("Probably, the optimizer configuration dictionary contained a mistake.")