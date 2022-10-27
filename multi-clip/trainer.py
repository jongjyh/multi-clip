from transformers import CLIPProcessor
from transformers.trainer import *
from CLIP_benchmark_internal.evaluate import evaluate

class OurTrainer(Trainer):
    # teacher = None
    # di_loss = 0.
    # iv_loss = 0.
    # mg_loss = 0.
    processor:CLIPProcessor = None
    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)
        
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, PreTrainedModel):
            if isinstance(unwrap_model(self.model), PreTrainedModel):
                if state_dict is None:
                    state_dict = self.model.state_dict()
                unwrap_model(self.model).save_pretrained(output_dir, state_dict=state_dict)
            else:
                logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                if state_dict is None:
                    state_dict = self.model.state_dict()
                torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            self.model.save_pretrained(output_dir, state_dict=state_dict)
        # saving processor
        if self.processor is not None:
            self.processor.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        # teacher_outputs = self.teacher(input_ids=inputs.pop('teacher_input_ids'),
        #              attention_mask=inputs.pop('teacher_attention_mask')).pooler_output
        # teacher_outputs = inputs['labels']
        outputs = model(**inputs)
        # loss_fn = torch.nn.MSELoss()
        # direct_loss = loss_fn(outputs['direct_outputs'],teacher_outputs) 
        # invert_loss = loss_fn(outputs['pooled_outputs'],teacher_outputs)
        # merge_loss = loss_fn(outputs['merge_outputs'],teacher_outputs)
        
        # self.di_loss += direct_loss.detach()
        # self.iv_loss += invert_loss.detach()
        # self.mg_loss += merge_loss.detach()

        # outputs['teacher_outputs']=teacher_outputs
        # outputs['loss'] = direct_loss + invert_loss + merge_loss 
        # outputs['loss'] = merge_loss 
        # outputs['loss'] = direct_loss 
        
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss
    
    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log:
            if is_torch_tpu_available():
                xm.mark_step()

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()
            # di_loss_scalar = self._nested_gather(self.di_loss).mean().item()
            # iv_loss_scalar = self._nested_gather(self.iv_loss).mean().item()
            # mg_loss_scalar = self._nested_gather(self.mg_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss
            # self.di_loss -= self.di_loss
            # self.iv_loss -= self.iv_loss       
            # self.mg_loss -= self.mg_loss

            # logs["direct_loss"] = round(di_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            # logs["invert_loss"] = round(iv_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            # logs["merge_loss"] =  round(mg_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, self.state.global_step, metrics)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)
    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (`Dataset`, *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns
                not accepted by the `model.forward()` method are automatically removed. It must implement the `__len__`
                method.
            ignore_keys (`Lst[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        do_retrieve = True
        if do_retrieve:
            dataset_attr = (
                # "flickr30k", "flickr30k-cn", "imagenet1k", "imagenet1k_zh"
                "flickr30k", "flickr30k-cn"

            )
            sys.path.append(
                '/home/chenzhongzhi/multi-clip/multi-clip/CLIP_benchmark_internal')
            from CLIP_benchmark_internal.evaluate import evaluate as clip_evaluate
            all_metrics = {}
            for name in dataset_attr:
                metrics = clip_evaluate(
                    dataset_name=name,
                    model_name=self.args.output_dir,
                    pretrained='pretrained',
                    output=f'/home/chenzhongzhi/results/{self.args.run_name}.json',
                    dataset_root="/sharefs/baai-mmdataset/clip_benchmark_datasets",
                    recall_k=[1, 5, 10],
                    model=self.model,
                    processor=self.processor
                )

                for key in metrics.keys():
                    all_metrics[f"{metric_key_prefix}_{name}_{key}"] = metrics[key]
            output = EvalLoopOutput(predictions=None, label_ids=None, metrics=all_metrics, num_samples=1)
            self.log(output.metrics)

        else:
            output = eval_loop(
                eval_dataloader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if self.compute_metrics is None else None,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )

            total_batch_size = self.args.eval_batch_size * self.args.world_size
            output.metrics.update(
                speed_metrics(
                    metric_key_prefix,
                    start_time,
                    num_samples=output.num_samples,
                    num_steps=math.ceil(output.num_samples / total_batch_size),
                )
            )

            self.log(output.metrics)

        if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics
