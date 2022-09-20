import transformers
from transformers import PreTrainedModel,CLIPTextModel
from transformers import BertModel,BertPreTrainedModel
from transformers import AutoConfig
import torch 
from typing import Optional,List
import torch.nn as nn
from .modeling_berts import BertSeriesConfig,BertSeriesModelWithTransformation


    
    
class KDmodel(PreTrainedModel):
    def __init__(self,config,):
        super().__init__(config,)
        # init student and teacher
        self.teacher = CLIPTextModel.from_pretrained(config.teacher_model)
        student_config = BertSeriesConfig.from_pretrained(config.student_model)
        student_config.project_dim = self.teacher.config.hidden_size
        student_config.pooler_fn = config.pooler_fn
        student_config.learn_encoder = True
        self.student = BertSeriesModelWithTransformation.from_pretrained(config.student_model,**{'config':student_config})
        self.student_config = self.student.config
        self.teacher_config = self.teacher.config
        self.loss_fn =config.loss_fn
        self.alpha = config.alpha
         
        # exp setting
        self.layer_KD = config.layer_kd

        if self.layer_KD:
            self.linear_layers = nn.ModuleList([ nn.Linear(student_config.hidden_size,student_config.project_dim)for _ in range(student_config.num_hidden_layers + 1)])
            for m in self.linear_layers:self._init_weights(m)
        
        # freeze teacher and init weights
        self.freeze()
        
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.student_config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        
    def freeze(self):
        for _,m in self.teacher.named_parameters():
            m.requires_grad_(False)
        # for n,m in self.student.named_parameters():
        #     if 'embeddings.word_embeddings.weight' in n:
        #         print(n)
        #         m.requires_grad_(False)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        teacher_input_ids = None,
        teacher_attention_mask = None,
    ) :
        # pooler_output,hidden_states, attentions
        student_outputs = self.student(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            encoder_hidden_states,
            encoder_attention_mask,
            output_attentions,
            return_dict=return_dict,
            output_hidden_states=True,
        )

        # last_hidden_state 
        # pooler_output ## EOS's embedding
        # hidden_states ## layer embedding
        # attentions
        teacher_outputs = self.teacher(
            teacher_input_ids,
            teacher_attention_mask,
            output_hidden_states = True
        )
        
        
        # learn pooler
        x,y = student_outputs['pooler_output'],teacher_outputs.pooler_output
        # learn encoder
        x1,y1 = student_outputs['mimic_states'].mean(-2),teacher_outputs.hidden_states[-1].mean(-2)

        # loss 
        if self.loss_fn=='mse':
            loss_fn = torch.nn.MSELoss()
        elif self.loss_fn=='cosine':     
            from functools import partial
            loss_fn = torch.nn.CosineEmbeddingLoss()
            # partial for reduce redundant parameter 
            loss_fn = partial(loss_fn,target=torch.tensor([1.],device='cuda' if torch.cuda.is_available() else 'cpu'))

        loss = loss_fn(x,y)+loss_fn(x1,y1)
        
        if self.layer_KD:
            teacher_hidden_states,student_hidden_states = teacher_outputs.hidden_states,student_outputs['hidden_states']
            
            kd_loss = []
            for i,hidden_state in enumerate(student_hidden_states):
                _pooler_output = self.student.pooler(hidden_state)
                _x = self.linear_layers[i](_pooler_output)
                # _x = _pooler_output
                _y = teacher_hidden_states[i][torch.arange(_x.shape[0]), teacher_input_ids.argmax(dim=-1)]
                kd_loss.append(loss_fn(_x,_y))
            
            loss = loss + torch.stack(kd_loss).sum() * self.alpha
        
        return {
            'loss':loss,
            'student_pooler_output':x,
            'teacher_pooler_putput':y,
        }