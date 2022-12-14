from transformers.models.clip.modeling_clip import *
from .modeling_clip import OurCLIPTextTransformer
import torch.nn as nn
import torch
import copy
from transformers import  CLIPProcessor
from transformers.models.roberta.modeling_roberta import RobertaLayer,BaseModelOutputWithPastAndCrossAttentions
from transformers.models.clip.modeling_clip import CLIPOutput,_expand_mask
from .modeling_clip import OurCLIPModel
from .config_dict import STUDENT_MODEL_DICT,STUDENT_CONFIG_DICT
import torch.distributed as dist

def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    student_loss = contrastive_loss(similarity)
    return student_loss

class AllGather(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor, rank, world_size):
        output = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(output, tensor)
        ctx.rank = rank
        ctx.batch_size = tensor.shape[0]
        return torch.cat(output, 0)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[ctx.batch_size * ctx.rank: ctx.batch_size * (ctx.rank + 1)],
            None,
            None
        )
allgather = AllGather.apply

@dataclass
class OursCLIPOutput(CLIPOutput):
    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )

@dataclass
class KDCLIPOutput(CLIPOutput):
    loss:torch.FloatTensor  = None
    pooled_outputs:torch.FloatTensor  = None
    inputs_clip:torch.FloatTensor  = None
    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )

class ChCLIPConfig(CLIPConfig):
    def __init__(self, text_model_name=None,vision_model_name=None,text_config_dict=None, vision_config_dict=None, projection_dim=512, logit_scale_init_value=2.6592, num_layers=3,variant='invert',**kwargs):
        super().__init__(text_config_dict, vision_config_dict, projection_dim, logit_scale_init_value, **kwargs)
        if text_config_dict is None:
            text_config_dict = {}
        # when reload the config from local, we need name to select which class should be instanced.
        self.text_config = STUDENT_CONFIG_DICT[kwargs['text_config']['model_type']](**kwargs.pop('text_config'))
        self.num_layers = num_layers
        self.text_model_name = text_model_name
        self.vision_model_name = vision_model_name
        self.variant = variant
    
class CHCLIPProcess(CLIPProcessor):
    tokenizer_class = ("XLMRobertaTokenizer","XLMRobertaTokenizerFast")
    # tokenizer_class = ("BertTokenizer", "BertTokenizerFast")
    def __init__(self, feature_extractor, tokenizer):
        super().__init__(feature_extractor, tokenizer)
        # in some cases, we need to switch to different tokenizer
        # if tokenizer.vocab_size!=250002:
        #     print("tokenizer is not XLM-R, switched to BertTokenizer.")
        #     self.tokenizer = BertTokenizer.from_pretrained(tokenizer.name_or_path)

class ChineseCLIP(CLIPPreTrainedModel):
    config_class = ChCLIPConfig

    def __init__(self, config: ChCLIPConfig,clip_model=None):
        super().__init__(config)

        if not isinstance(config.vision_config, CLIPVisionConfig):
            raise ValueError(
                "config.vision_config is expected to be of type CLIPVisionConfig but is of type"
                f" {type(config.vision_config)}."
            )

        text_config = config.text_config
        vision_config = config.vision_config

        self.projection_dim = config.projection_dim
        self.text_embed_dim = text_config.project_dim
        self.vision_embed_dim = vision_config.hidden_size

        self.text_model = STUDENT_MODEL_DICT[text_config.model_type](text_config)
        self.vision_model = CLIPVisionTransformer(vision_config)

        self.visual_projection = nn.Linear(self.vision_embed_dim, self.projection_dim, bias=False)
        self.text_projection = nn.Linear(self.text_embed_dim, self.projection_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.ones([]) * self.config.logit_scale_init_value)

        # Initialize weights and apply final processing
        self.post_init()
                
    def get_text_features(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        token_type_ids=None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        r"""
        Returns:
            text_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The text embeddings obtained by
            applying the projection layer to the pooled output of [`CLIPTextModel`].

        Examples:

        ```python
        >>> from transformers import CLIPTokenizer, CLIPModel

        >>> model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")
        >>> text_features = model.get_text_features(**inputs)
        ```"""
        # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = text_outputs['pooler_output']
        text_features = self.text_projection(pooled_output)

        return text_features

    @add_start_docstrings_to_model_forward(CLIP_VISION_INPUTS_DOCSTRING)
    def get_image_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        r"""
        Returns:
            image_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The image embeddings obtained by
            applying the projection layer to the pooled output of [`CLIPVisionModel`].

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import CLIPProcessor, CLIPModel

        >>> model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> image_features = model.get_image_features(**inputs)
        ```"""
        # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = vision_outputs[1]  # pooled_output
        image_features = self.visual_projection(pooled_output)

        return image_features

    @add_start_docstrings_to_model_forward(CLIP_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CLIPOutput, config_class=CLIPConfig)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        token_type_ids=None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CLIPOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import CLIPProcessor, CLIPModel

        >>> model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(
        ...     text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True
        ... )

        >>> outputs = model(**inputs)
        >>> logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        >>> probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
        ```"""
        # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # intermediate_outputs = self.text_model(
        #     input_embeds=text_outputs['last_hidden_state'],
        #     attention_mask=attention_mask,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict,
        # )
        
        # inputs_clip = self.clip_head(intermediate_outputs['last_hidden_state'])
        # clip_outputs = self.clip_model(
        #     input_embeds=inputs_clip,
        #     attention_mask=attention_mask,
        #     position_ids=position_ids,
        #     token_type_ids=token_type_ids,
        #     output_attentions=False,
        #     output_hidden_states=False,
        #     return_dict=return_dict,
        # )

        image_embeds = vision_outputs[1]
        image_embeds = self.visual_projection(image_embeds)

        # text_embeds = self.text_model.get_text_embeds(text_outputs['pooler_output'],clip_outputs[1])
        text_embeds = text_outputs['pooler_output']
        text_embeds = self.text_projection(text_embeds)

        # normalized features
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.T

        loss = None
        if return_loss:
            loss = clip_loss(logits_per_text)

        if not return_dict:
            output = (logits_per_image, logits_per_text, text_embeds, image_embeds, text_outputs, vision_outputs)
            return ((loss,) + output) if loss is not None else output

        return OursCLIPOutput(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )

class RobertaLayerStack(nn.Module):
    def __init__(self,config,projection_dim,num_layers) -> None:
        super().__init__()
        copy_config = copy.deepcopy(config)
        copy_config.hidden_size = projection_dim
        copy_config.intermediate_size = projection_dim * 4
        copy_config.num_attention_heads = 8
        self.layer = nn.ModuleList([RobertaLayer(copy_config) for _ in range(num_layers)])
        self.gradient_checkpointing = False
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
        
# copy from bert.
class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config,clip_hidden_size):
        super().__init__()
        self.dense = nn.Linear(clip_hidden_size * 2, clip_hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(clip_hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class DoubleCLIP(CLIPPreTrainedModel):
    config_class = ChCLIPConfig

    def __init__(self, config: ChCLIPConfig):
        super().__init__(config)

        if not isinstance(config.vision_config, CLIPVisionConfig):
            raise ValueError(
                "config.vision_config is expected to be of type CLIPVisionConfig but is of type"
                f" {type(config.vision_config)}."
            )

        text_config = config.text_config
        vision_config = config.vision_config
        self.variant = config.variant

        self.projection_dim = config.projection_dim
        self.text_embed_dim = text_config.project_dim
        self.vision_embed_dim = vision_config.hidden_size

        self.text_model = STUDENT_MODEL_DICT[config.text_model_name].from_pretrained(config.text_model_name,config=text_config)
        clip_model = OurCLIPModel.from_pretrained(config.vision_model_name)

        self.vision_model = clip_model.vision_model
        self.text_teacher = clip_model.text_model if self.variant == 'invert' else None
        
        # inverse module and clip.
        if self.variant == 'invert':
            self.inverse_tfm = RobertaLayerStack(text_config,text_config.project_dim,config.num_layers)
            # self.merge_head = BertPredictionHeadTransform(config.text_config,config.text_config.project_dim)
            # init weights.
            self.inverse_tfm.apply(self._init_weights)
            # self.merge_head.apply(self._init_weights)

        self.visual_projection = nn.Linear(self.vision_embed_dim, self.projection_dim, bias=False)
        self.text_projection = nn.Linear(self.text_embed_dim, self.projection_dim, bias=False)

        # match CLIP weights
        self.visual_projection.weight.data = clip_model.visual_projection.weight.data
        self.text_projection.weight.data = clip_model.text_projection.weight.data
        self.logit_scale = nn.Parameter(torch.ones([]) * self.config.logit_scale_init_value)
        
        if self.variant == 'invert': self.freeze_clip()
        
    def freeze_clip(self):
        assert isinstance(self.text_teacher,OurCLIPTextTransformer)
        for _,params in self.text_teacher.named_parameters():
            params.requires_grad_(False)
        
        
    def get_text_features(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        token_type_ids=None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        r"""
        Returns:
            text_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The text embeddings obtained by
            applying the projection layer to the pooled output of [`CLIPTextModel`].

        Examples:

        ```python
        >>> from transformers import CLIPTokenizer, CLIPModel

        >>> model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")
        >>> text_features = model.get_text_features(**inputs)
        ```"""
        # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            mode=None
            # mode='middle'
        )
        
        # extend attention mask to [ batch_size ,seq_len ,seq_len ]
        if self.variant == 'invert':
            extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, text_outputs['projection_state'].size()[:-1])
            text_projection = text_outputs['projection_state']
            inputs_clip = self.inverse_tfm(
                hidden_states=text_projection,
                attention_mask=extended_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )[0]
            clip_outputs = self.text_teacher(
                inputs_embeds=inputs_clip,
                attention_mask=attention_mask,
                input_ids_=input_ids,
            )[1]
            pooled_outputs = clip_outputs
        else:
            pooled_outputs = text_outputs['pooler_output']

        text_embeds = self.text_projection(pooled_outputs)
        return text_embeds,inputs_clip 

    @add_start_docstrings_to_model_forward(CLIP_VISION_INPUTS_DOCSTRING)
    def get_image_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        r"""
        Returns:
            image_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The image embeddings obtained by
            applying the projection layer to the pooled output of [`CLIPVisionModel`].

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import CLIPProcessor, CLIPModel

        >>> model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> image_features = model.get_image_features(**inputs)
        ```"""
        # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = vision_outputs[1]  # pooled_output
        image_features = self.visual_projection(pooled_output)

        return image_features

    @add_start_docstrings_to_model_forward(CLIP_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CLIPOutput, config_class=CLIPConfig)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        token_type_ids=None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CLIPOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import CLIPProcessor, CLIPModel

        >>> model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(
        ...     text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True
        ... )

        >>> outputs = model(**inputs)
        >>> logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        >>> probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
        ```"""
        # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            mode=None
            # mode='middle',
        )

        # extend attention mask to [ batch_size ,seq_len ,seq_len ]
        if self.variant == 'invert':
            extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, text_outputs['projection_state'].size()[:-1])
            text_projection = text_outputs['projection_state']
            inputs_clip = self.inverse_tfm(
                hidden_states=text_projection,
                attention_mask=extended_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )[0]
            clip_outputs = self.text_teacher(
                inputs_embeds=inputs_clip,
                attention_mask=attention_mask,
                input_ids_=input_ids,
            )[1]
            pooled_outputs = clip_outputs
        else:
            pooled_outputs = text_outputs['pooler_output']

       
        image_embeds = vision_outputs[1]
        image_embeds = self.visual_projection(image_embeds)

        text_embeds = pooled_outputs
        text_embeds = self.text_projection(text_embeds)

        # normalized features
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.T

        loss = None
        if return_loss:
            loss = clip_loss(logits_per_text)

        if not return_dict:
            output = (logits_per_image, logits_per_text, text_embeds, image_embeds, text_outputs, vision_outputs)
            return ((loss,) + output) if loss is not None else output

        return OursCLIPOutput(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )

class DoubleCLIPWithKD(DoubleCLIP):
    
    def forward(self, input_ids: Optional[torch.LongTensor] = None, attention_mask: Optional[torch.Tensor] = None, position_ids: Optional[torch.LongTensor] = None, token_type_ids=None, return_loss: Optional[bool] = None, output_attentions: Optional[bool] = None, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None,labels: Optional[torch.Tensor] = None) -> Union[Tuple, CLIPOutput]:
        # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            mode=None
            # mode='middle',
        )

        # extend attention mask to [ batch_size ,seq_len ,seq_len ]
        if self.variant == 'invert':
            extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, text_outputs['projection_state'].size()[:-1])
            text_projection = text_outputs['projection_state']
            inputs_clip = self.inverse_tfm(
                hidden_states=text_projection,
                attention_mask=extended_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )[0]
            clip_outputs = self.text_teacher(
                inputs_embeds=inputs_clip,
                attention_mask=attention_mask,
                input_ids_=input_ids,
            )[1]
            pooled_outputs = clip_outputs
        else:
            pooled_outputs = text_outputs['pooler_output']

        # loss_fn = 'cl and mse'
        loss_fn = 'mse'
        loss = 0.
        if labels is not None and 'cl' in loss_fn:
            # normalized features
            teacher_embeds = labels / labels.norm(p=2, dim=-1, keepdim=True)
            student_embeds = pooled_outputs / pooled_outputs.norm(p=2, dim=-1, keepdim=True)
            teacher_embeds_all = allgather(student_embeds, torch.distributed.get_rank(), torch.distributed.get_world_size())
            student_embeds_all = allgather(teacher_embeds, torch.distributed.get_rank(), torch.distributed.get_world_size())
            logit_scale = self.logit_scale.exp()
            logits_per_student = torch.matmul(student_embeds_all, teacher_embeds_all.t()) * logit_scale
            loss += clip_loss(logits_per_student)
        if labels is not None and 'mse' in loss_fn:
            loss_f = torch.nn.MSELoss()
            loss += loss_f(pooled_outputs,labels)

        return KDCLIPOutput(
            loss=loss,
            pooled_outputs=pooled_outputs,
            inputs_clip=inputs_clip
        )
