import transformers
from transformers import PreTrainedModel,CLIPPreTrainedModel,CLIPTextModel
from transformers import BertModel,CLIPTextConfig,XLMRobertaTokenizer,XLMRobertaModel
from transformers import RobertaPreTrainedModel,RobertaConfig
import torch 
from typing import Optional,List,Union,Tuple

class KDmodel(RobertaPreTrainedModel):
    def __init__(self,config):
        super().__init__(config)
        self.student = XLMRobertaModel.from_pretrained("xlm-roberta-large")
        self.teacher = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        self.pooling = 'average'
        self.loss_fn = 'mse'
        self.freeze()
        
    def freeze(self):
        for n,m in self.teacher.named_parameters():
            m.requires_grad_(False)

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
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        teacher_input_ids = None,
        teacher_attention_mask = None,
    ) :
        student_hidden_states = self.student(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_values,
            use_cache,
            output_attentions,
            return_dict=return_dict,
            output_hidden_states=True,
        ).hidden_states

        teacher_hidden_states = self.teacher(
            teacher_input_ids,
            teacher_attention_mask,
            output_hidden_states = True
        ).hidden_states
        
        # pooling
        if self.pooling=='average':
            teacher_hidden_states = teacher_hidden_states[-1].mean(-2)
            student_hidden_states = student_hidden_states[-1].mean(-2)
        elif self.pooling=='single':
            teacher_hidden_states = teacher_hidden_states
            student_hidden_states = student_hidden_states

        # loss 
        if self.loss_fn=='mse':
            loss_fn = torch.nn.MSELoss()
        elif self.loss_fn=='cosine':     
            loss_fn = torch.nn.CosineEmbeddingLoss()
        elif self.loss_fn=='logits':     
            loss_fn = torch.nn.CrossEntropyLoss()
            
        loss = loss_fn(teacher_hidden_states,student_hidden_states)
        
        return {
            'loss':loss,
        }