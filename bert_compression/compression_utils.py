from typing import Tuple
from transformers import BertConfig, BertForSequenceClassification, BertModel
from functools import partial
from typing import Optional, Tuple

#from transformers.models.bert import configuration_bert
from transformers import AutoModel, AutoConfig, AutoModelForSequenceClassification
from .utils_svd import map_module, compress_linear
comp_types = {
    "svd": compress_linear,
    }

class CompressedBertConfig(BertConfig):
    """Class CompressedModelForCausalLM defines a configuration for compressed
    Mistral. Here, we split shape to input and output shape in order to serialize
    them to different fields in JSON.
    """

    def __init__(self, *args, shape_in: Tuple[int] = (),
                 shape_out: Tuple[int] = (), rank: int = 128, compression_type: str = 'svd', model_already_compressed = False,
                 layer_mask: str = r'/decoder/layer/\d+/(intermediate|output)', **kwargs):
        super().__init__(*args, **kwargs)
        self.shape_in = shape_in
        self.shape_out = shape_out
        self.rank = rank
        self.compression_type = compression_type
        self.layer_mask = layer_mask
        self.model_already_compressed = model_already_compressed

#CompressedBertConfig.register_for_auto_class()

class CompressedBertForSequenceClassification(BertForSequenceClassification):
    """Class TTCompressedBertForSequenceClassification defines a BERT-based model
    with compressed linear layers with TT.
    """

    LAYERS = r'/(de|en)coder/layers/\d+/fc[12]'
    #LAYERS = r'/encoder/layer/\d+/(intermediate|output)'

    config_class = CompressedBertConfig

    def __init__(self, config: CompressedBertConfig,
                shape: Optional[Tuple[Tuple[int], Tuple[int]]] = None,
                rank: Optional[int] = None,
                layer_mask: Optional[str] = None,
                compression_type: str = None,
                model_already_compressed: bool = False, 
                ):
        super().__init__(config)

        self.rank = rank or config.rank
        self.layer_mask = layer_mask or config.layer_mask
        self.compression_type = compression_type or config.compression_type
        self.model_already_compressed = model_already_compressed or config.model_already_compressed

        self.shape = shape
        if self.shape is None:
            self.shape = (tuple(self.config.shape_in),
                          tuple(self.config.shape_out))
        if self.model_already_compressed:
            compress_fn = partial(comp_types[self.compression_type], 
                        rank=self.rank, shape=self.shape,
                        weight=None, random_init=True, cholesky=False
                        )
            self = map_module(self, compress_fn, self.layer_mask)   

    def to_compression(self, compress: bool = False, weight = None):
        compress_fn = partial(comp_types[self.compression_type], 
                              rank=self.rank, shape=self.shape,
                              weight=weight, random_init=False, cholesky=False
                              )
        #if not compress:
        #    compress_fn = self.convert
        self = map_module(self, compress_fn, self.layer_mask)

        self.config.model_already_compressed = True

#CompressedBertForSequenceClassification.register_for_auto_class('AutoModelForSequenceClassification')

class CompressedBertModel(BertModel):
    """Class TTCompressedBertForSequenceClassification defines a BERT-based model
    with compressed linear layers with TT.
    """

    #LAYERS = r'/(de|en)coder/layers/\d+/fc[12]'
    #LAYERS = r'/encoder/layer/\d+/(intermediate|output)'

    config_class = CompressedBertConfig

    def __init__(self, config: CompressedBertConfig,
                shape: Optional[Tuple[Tuple[int], Tuple[int]]] = None,
                rank: Optional[int] = None,
                layer_mask: Optional[str] = None,
                compression_type: str = None,
                model_already_compressed: bool = False, 
                ):
        super().__init__(config)

        self.rank = rank or config.rank
        self.layer_mask = layer_mask or config.layer_mask
        self.compression_type = compression_type or config.compression_type
        self.model_already_compressed = model_already_compressed or config.model_already_compressed

        self.shape = shape
        if self.shape is None:
            self.shape = (tuple(self.config.shape_in),
                          tuple(self.config.shape_out))
        if self.model_already_compressed:
            compress_fn = partial(comp_types[self.compression_type], 
                        rank=self.rank, shape=self.shape,
                        weight=None, random_init=True, cholesky=True if self.compression_type else False
                        )
            self = map_module(self, compress_fn, self.layer_mask)   

    def to_compression(self, compress: bool = False, weight = None):
        compress_fn = partial(comp_types[self.compression_type], 
                              rank=self.rank, shape=self.shape,
                              weight=weight, random_init=False, cholesky=True if self.compression_type else False
                              )
        #if not compress:
        #    compress_fn = self.convert
        self = map_module(self, compress_fn, self.layer_mask)

        self.config.model_already_compressed = True

#AutoConfig.register("CompressedBert", CompressedBertConfig)
CompressedBertConfig.register_for_auto_class()
AutoModel.register(CompressedBertConfig, CompressedBertModel)
AutoModelForSequenceClassification.register(CompressedBertConfig, CompressedBertForSequenceClassification)
#CompressedBertModel.register_for_auto_class("AutoModel")