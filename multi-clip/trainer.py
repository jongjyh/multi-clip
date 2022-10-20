from transformers.trainer import *
class OurTrainer(Trainer):
    teacher = None
    di_loss = 0.
    iv_loss = 0.
    mg_loss = 0.
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

    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        teacher_outputs = self.teacher(input_ids=inputs.pop('teacher_input_ids'),
                     attention_mask=inputs.pop('teacher_attention_mask'))[1]
        
        outputs = model(**inputs)
        loss_fn = torch.nn.MSELoss()
        direct_loss = loss_fn(outputs['direct_outputs'],teacher_outputs) 
        invert_loss = loss_fn(outputs['clip_outputs'],teacher_outputs)
        merge_loss = loss_fn(outputs['merge_outputs'],teacher_outputs)
        
        self.di_loss += direct_loss.detach()
        self.iv_loss += invert_loss.detach()
        self.mg_loss += merge_loss.detach()

        outputs['teacher_outputs']=teacher_outputs
        # outputs['loss'] = direct_loss + invert_loss + merge_loss 
        outputs['loss'] = direct_loss 
        
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
            di_loss_scalar = self._nested_gather(self.di_loss).mean().item()
            iv_loss_scalar = self._nested_gather(self.iv_loss).mean().item()
            mg_loss_scalar = self._nested_gather(self.mg_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss
            self.di_loss -= self.di_loss
            self.iv_loss -= self.iv_loss       
            self.mg_loss -= self.mg_loss

            logs["direct_loss"] = round(di_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["invert_loss"] = round(iv_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["merge_loss"] =  round(mg_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
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

    