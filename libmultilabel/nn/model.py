from abc import abstractmethod

import lightning as L
import torch
import torch.nn.functional as F
import torch.optim as optim

from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from ..common_utils import dump_log
from ..nn.metrics import get_metrics, tabulate_metrics


class MultiLabelModel(L.LightningModule):
    """Abstract class handling Pytorch Lightning training flow.

    Args:
        num_classes (int): Total number of classes.
        learning_rate (float, optional): Learning rate for optimizer. Defaults to 0.0001.
        learning_rate_encoder (float, optional): Learning rate for encoder params. Defaults to None.
        learning_rate_classifier (float, optional): Learning rate for classifier params. Defaults to None.
        optimizer (str, optional): Optimizer name (i.e., sgd, adam, adamw, adamax). Defaults to 'adam'.
        optimizer_encoder (str, optional): Optimizer name for encoder params. Defaults to None (falls back to optimizer).
        optimizer_classifier (str, optional): Optimizer name for classifier params. Defaults to None (falls back to optimizer).
        momentum (float, optional): Momentum factor for SGD only. Defaults to 0.9.
        weight_decay (float, optional): Weight decay factor. Defaults to 0.
        weight_decay_encoder (float, optional): Weight decay for encoder params. Defaults to None (falls back to weight_decay).
        weight_decay_classifier (float, optional): Weight decay for classifier params. Defaults to None (falls back to weight_decay).
        metric_threshold (float, optional): The decision value threshold over which a label is predicted as positive. Defaults to 0.5.
        monitor_metrics (list, optional): Metrics to monitor while validating. Defaults to None.
        log_path (str): Path to a directory holding the log files and models.
        multiclass (bool, optional): Enable multiclass mode. Defaults to False.
        silent (bool, optional): Enable silent mode. Defaults to False.
        save_k_predictions (int, optional): Save top k predictions on test set. Defaults to 0.
    """

    def __init__(
        self,
        num_classes,
        learning_rate=0.0001,
        learning_rate_encoder=None,
        learning_rate_classifier=None,
        optimizer="adam",
        optimizer_encoder=None,
        optimizer_classifier=None,
        momentum=0.9,
        weight_decay=0,
        weight_decay_encoder=None,
        weight_decay_classifier=None,
        lr_scheduler=None,
        scheduler_config=None,
        val_metric=None,
        metric_threshold=0.5,
        monitor_metrics=None,
        log_path=None,
        multiclass=False,
        silent=False,
        save_k_predictions=0,
        **kwargs
    ):
        super().__init__()

        # optimizer
        self.learning_rate = learning_rate
        self.learning_rate_encoder = learning_rate_encoder
        self.learning_rate_classifier = learning_rate_classifier
        self.optimizer = optimizer
        self.optimizer_encoder = optimizer_encoder
        self.optimizer_classifier = optimizer_classifier
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.weight_decay_encoder = weight_decay_encoder
        self.weight_decay_classifier = weight_decay_classifier

        # lr_scheduler
        self.lr_scheduler = lr_scheduler
        self.scheduler_config = scheduler_config
        self.val_metric = val_metric

        # dump log
        self.log_path = log_path
        self.silent = silent
        self.save_k_predictions = save_k_predictions

        # metrics for evaluation
        self.multiclass = multiclass
        top_k = 1 if self.multiclass else None
        self.eval_metric = get_metrics(metric_threshold, monitor_metrics, num_classes, top_k=top_k)

        self.automatic_optimization = True

    @abstractmethod
    def shared_step(self, batch):
        """Return loss and predicted logits."""
        return NotImplemented

    def configure_optimizers(self):
        def build_optimizer(opt_name, params, lr, wd):
            if opt_name == "sgd":
                return optim.SGD(params, lr=lr, momentum=self.momentum, weight_decay=wd)
            elif opt_name == "adam":
                return optim.Adam(params, lr=lr, weight_decay=wd)
            elif opt_name == "adamw":
                return optim.AdamW(params, lr=lr, weight_decay=wd)
            elif opt_name == "adamax":
                return optim.Adamax(params, lr=lr, weight_decay=wd)
            else:
                raise RuntimeError(f"Unsupported optimizer: {opt_name}")

        split_needed = any(
            x is not None
            for x in [
                self.learning_rate_encoder,
                self.learning_rate_classifier,
                self.optimizer_encoder,
                self.optimizer_classifier,
                self.weight_decay_encoder,
                self.weight_decay_classifier,
            ]
        )

        if split_needed:
            # Split parameters
            encoder_params = []
            classifier_params = []
            for name, param in self.named_parameters():
                if not param.requires_grad:
                    continue
                if "encoder" in name or "distilbert" in name or "transformer" in name or "lm" in name:
                    encoder_params.append(param)
                else:
                    classifier_params.append(param)

            enc_lr = self.learning_rate_encoder if self.learning_rate_encoder is not None else self.learning_rate
            cls_lr = self.learning_rate_classifier if self.learning_rate_classifier is not None else self.learning_rate

            enc_wd = self.weight_decay_encoder if self.weight_decay_encoder is not None else self.weight_decay
            cls_wd = self.weight_decay_classifier if self.weight_decay_classifier is not None else self.weight_decay

            enc_opt_name = self.optimizer_encoder if self.optimizer_encoder is not None else self.optimizer
            cls_opt_name = self.optimizer_classifier if self.optimizer_classifier is not None else self.optimizer

            if enc_opt_name == cls_opt_name:
                parameters = [
                    {"params": encoder_params, "lr": enc_lr, "weight_decay": enc_wd},
                    {"params": classifier_params, "lr": cls_lr, "weight_decay": cls_wd},
                ]
                optimizer = build_optimizer(enc_opt_name, parameters, lr=self.learning_rate, wd=self.weight_decay)
                optimizers_list = None
                base_optimizer_for_scheduler = optimizer
            else:
                self.automatic_optimization = False
                opt_enc = build_optimizer(enc_opt_name, encoder_params, enc_lr, enc_wd)
                opt_cls = build_optimizer(cls_opt_name, classifier_params, cls_lr, cls_wd)
                optimizers_list = [opt_enc, opt_cls]
                base_optimizer_for_scheduler = None
        else:
            parameters = [p for p in self.parameters() if p.requires_grad]
            optimizer = build_optimizer(self.optimizer, parameters, self.learning_rate, self.weight_decay)
            optimizers_list = None
            base_optimizer_for_scheduler = optimizer

        total_steps = self.trainer.estimated_stepping_batches

        warmup_steps = 0
        if self.scheduler_config:
            warmup_ratio = self.scheduler_config.get("warmup_ratio", None)
            warmup_steps_cfg = self.scheduler_config.get("warmup_steps", None)
            if warmup_ratio is not None:
                warmup_steps = int(total_steps * warmup_ratio)
            elif warmup_steps_cfg is not None:
                warmup_steps = int(warmup_steps_cfg)

        def build_scheduler(opt):
            cfg = dict(self.scheduler_config or {})
            if self.lr_scheduler == "ReduceLROnPlateau":
                return {
                    "scheduler": optim.lr_scheduler.ReduceLROnPlateau(
                        opt, mode="min" if self.val_metric == "Loss" else "max", **cfg
                    ),
                    "monitor": self.val_metric,
                }
            elif self.lr_scheduler == "linear_schedule_with_warmup":
                scheduler = get_linear_schedule_with_warmup(
                    opt, num_warmup_steps=warmup_steps, num_training_steps=total_steps
                )
                return {"scheduler": scheduler, "interval": "step", "frequency": 1}
            elif self.lr_scheduler == "cosine_schedule_with_warmup":
                scheduler = get_cosine_schedule_with_warmup(
                    opt, num_warmup_steps=warmup_steps, num_training_steps=total_steps
                )
                return {"scheduler": scheduler, "interval": "step", "frequency": 1}
            else:
                raise RuntimeError(f"Unsupported learning rate scheduler: {self.lr_scheduler}")

        if optimizers_list is None:
            if self.lr_scheduler:
                lr_scheduler_config = build_scheduler(base_optimizer_for_scheduler)
                return {"optimizer": base_optimizer_for_scheduler, "lr_scheduler": lr_scheduler_config}
            return base_optimizer_for_scheduler
        else:
            if self.lr_scheduler:
                lr_scheduler_config = [build_scheduler(o) for o in optimizers_list]
                return {"optimizer": optimizers_list, "lr_scheduler": lr_scheduler_config}
            return optimizers_list

    def training_step(self, batch, batch_idx):
        loss, _ = self.shared_step(batch)

        if not self.automatic_optimization:
            opt_enc, opt_cls = self.optimizers()
            opt_enc.zero_grad()
            opt_cls.zero_grad()
            self.manual_backward(loss)
            opt_enc.step()
            opt_cls.step()

            # step "per-step" schedulers manually
            if self.lr_scheduler in ("linear_schedule_with_warmup", "cosine_schedule_with_warmup"):
                scheds = self.lr_schedulers()
                if not isinstance(scheds, (list, tuple)):
                    scheds = [scheds]
                for s in scheds:
                    s.step()

        # record train loss vs epoch
        self.log(
            "train/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        self._shared_eval_step(batch, batch_idx)

    def on_validation_epoch_end(self):
        return self._shared_eval_epoch_end(split="val")

    def test_step(self, batch, batch_idx):
        self._shared_eval_step(batch, batch_idx)

    def on_test_epoch_end(self):
        return self._shared_eval_epoch_end(split="test")

    def _shared_eval_step(self, batch, batch_idx):
        loss, pred_logits = self.shared_step(batch)
        pred_scores = torch.sigmoid(pred_logits)
        self.eval_metric.update(preds=pred_scores, target=batch["label"], loss=loss)

    def _shared_eval_epoch_end(self, split):
        """Get scores such as `Micro-F1`, `Macro-F1`, and monitor metrics defined
        in the configuration file in the end of an epoch.

        Args:
            step_outputs (list): List of the return values from the val or test step end.
            split (str): One of the `val` or `test`.

        Returns:
            metric_dict (dict): Scores for all metrics in the dictionary format.
        """
        metric_dict = self.eval_metric.compute()
        self.log_dict(metric_dict)
        for k, v in metric_dict.items():
            metric_dict[k] = v.item()
        if self.log_path:
            dump_log(metrics=metric_dict, split=split, log_path=self.log_path)
        self.print(tabulate_metrics(metric_dict, split))
        self.eval_metric.reset()
        return metric_dict

    def predict_step(self, batch, batch_idx):
        """`predict_step` is triggered when calling `trainer.predict()`.
        This function is used to get the top-k labels and their prediction scores.

        Args:
            batch (dict): A batch of text and label.
            batch_idx (int): Index of current batch.

        Returns:
            dict: Top k label indexes and the prediction scores.
        """
        pred_logits = self(batch)
        pred_scores = pred_logits.detach().cpu().numpy()
        # k = self.save_k_predictions
        # top_k_idx = argsort_top_k(pred_scores, k, axis=1)
        # top_k_scores = np.take_along_axis(pred_scores, top_k_idx, axis=1)

        # return {"top_k_pred": top_k_idx, "top_k_pred_scores": top_k_scores}
        return {"pred_scores": pred_scores}

    def forward(self, batch):
        """Compute predicted logits."""
        return self.network(batch)["logits"]

    def print(self, *args, **kwargs):
        """Print only from process 0 and not in silent mode. Use this in any
        distributed mode to log only once."""

        if not self.silent:
            # print() in LightningModule to print only from process 0
            super().print(*args, **kwargs)


class Model(MultiLabelModel):
    """A class that implements `MultiLabelModel` for initializing and training a neural network.

    Args:
        classes (list): List of class names.
        network (nn.Module): Network (i.e., CAML, KimCNN, or XMLCNN).
        loss_function (str, optional): Loss function name (i.e., binary_cross_entropy_with_logits,
            cross_entropy). Defaults to 'binary_cross_entropy_with_logits'.
        log_path (str): Path to a directory holding the log files and models.
    """

    def __init__(self, classes, network, loss_function="binary_cross_entropy_with_logits", log_path=None, **kwargs):
        super().__init__(num_classes=len(classes), log_path=log_path, **kwargs)
        self.save_hyperparameters(
            ignore=["log_path"]
        )  # If log_path is saved, loading the checkpoint will cause an error since each experiment has unique log_path (result_dir).
        self.classes = classes
        self.network = network
        self.configure_loss_function(loss_function)

    def configure_loss_function(self, loss_function):
        assert hasattr(
            F, loss_function
        ), """
            Invalid `loss_function`. Make sure the loss function is defined here:
            https://pytorch.org/docs/stable/nn.functional.html#loss-functions"""
        self.loss_function = getattr(F, loss_function)

    def shared_step(self, batch):
        """Return loss and predicted logits of the network.

        Args:
            batch (dict): A batch of text and label.

        Returns:
            loss (torch.Tensor): Loss between target and predict logits.
            pred_logits (torch.Tensor): The predict logits (batch_size, num_classes).
        """
        target_labels = batch["label"]
        pred_logits = self(batch)
        loss = self.loss_function(pred_logits, target_labels.float())

        return loss, pred_logits
