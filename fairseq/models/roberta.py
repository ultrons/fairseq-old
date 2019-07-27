# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
"""
RoBERTa: A Robustly Optimized BERT Pretraining Approach.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
from fairseq.data import encoders
from fairseq.models import (
    FairseqDecoder,
    FairseqLanguageModel,
    register_model,
    register_model_architecture,
)
from fairseq.modules import (
    LayerNorm,
    TransformerSentenceEncoder,
)
from fairseq.modules.transformer_sentence_encoder import init_bert_params


class RobertaHubInterface(nn.Module):
    """A simple PyTorch Hub interface to RoBERTa.

    Load RoBERTa::

        >>> roberta = torch.hub.load('pytorch/fairseq', 'roberta.large')
        >>> roberta.eval()  # disable dropout (or leave in train mode to finetune)

    Apply Byte-Pair Encoding (BPE) to input text::

        >>> tokens = roberta.encode('Hello world!')
        >>> tokens
        tensor([    0, 31414,   232,   328,     2])

    Extract features from RoBERTa::

        >>> last_layer_features = roberta.extract_features(tokens)
        >>> last_layer_features.size()
        torch.Size([1, 5, 1024])

        >>> all_layers = roberta.extract_features(tokens, return_all_hiddens=True)
        >>> len(all_layers)
        25
        >>> torch.all(all_layers[-1] == last_layer_features)
        tensor(1, dtype=torch.uint8)

    Use RoBERTa for sentence-pair classification tasks::

        >>> roberta = torch.hub.load('pytorch/fairseq', 'roberta.large.mnli')  # already finetuned
        >>> roberta.eval()  # disable dropout for evaluation

        >>> tokens = roberta.encode(
        ...   'Roberta is a heavily optimized version of BERT.',
        ...   'Roberta is not very optimized.'
        ... )
        >>> roberta.predict('mnli', tokens).argmax()
        tensor(0)  # contradiction

        >>> tokens = roberta.encode(
        ...   'Roberta is a heavily optimized version of BERT.',
        ...   'Roberta is based on BERT.'
        ... )
        >>> roberta.predict('mnli', tokens).argmax()
        tensor(2)  # entailment

    Register a new (randomly initialized) classification head::

        >>> roberta.register_classification_head('new_task', num_classes=3)
        >>> roberta.predict('new_task', tokens)
        tensor([[-1.1050, -1.0672, -1.1245]], grad_fn=<LogSoftmaxBackward>)

    Using the GPU::

        >>> roberta.cuda()
        >>> roberta.predict('new_task', tokens)
        tensor([[-1.1050, -1.0672, -1.1245]], device='cuda:0', grad_fn=<LogSoftmaxBackward>)
    """

    def __init__(self, args, task, model):
        super().__init__()
        self.args = args
        self.task = task
        self.model = model

        self.bpe = encoders.build_bpe(args)

        # this is useful for determining the device
        self.register_buffer('_float_tensor', torch.tensor([0], dtype=torch.float))

    @property
    def device(self):
        return self._float_tensor.device

    def encode(self, sentence: str, *addl_sentences) -> torch.LongTensor:
        bpe_sentence = '<s> ' + self.bpe.encode(sentence) + ' </s>'
        for s in addl_sentences:
            bpe_sentence += ' </s> ' + self.bpe.encode(s)
        tokens = self.task.source_dictionary.encode_line(bpe_sentence, append_eos=True)
        return tokens.long()

    def extract_features(self, tokens: torch.LongTensor, return_all_hiddens=False) -> torch.Tensor:
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        features, extra = self.model(
            tokens.to(device=self.device),
            features_only=True,
            return_all_hiddens=return_all_hiddens,
        )
        if return_all_hiddens:
            # convert from T x B x C -> B x T x C
            inner_states = extra['inner_states']
            return [inner_state.transpose(0, 1) for inner_state in inner_states]
        else:
            return features  # just the last layer's features

    def register_classification_head(
        self, name: str, num_classes: int = None, embedding_size: int = None, **kwargs
    ):
        self.model.register_classification_head(
            name, num_classes=num_classes, embedding_size=embedding_size, **kwargs
        )

    def predict(self, head: str, tokens: torch.LongTensor):
        features = self.extract_features(tokens)
        logits = self.model.classification_heads[head](features)
        return F.log_softmax(logits, dim=-1)


@register_model('roberta')
class RobertaModel(FairseqLanguageModel):

    @classmethod
    def hub_models(cls):
        return {
            'roberta.base': 'http://dl.fbaipublicfiles.com/fairseq/models/roberta.base.tar.gz',
            'roberta.large': 'http://dl.fbaipublicfiles.com/fairseq/models/roberta.large.tar.gz',
            'roberta.large.mnli': 'http://dl.fbaipublicfiles.com/fairseq/models/roberta.large.mnli.tar.gz',
        }

    def __init__(self, args, encoder):
        super().__init__(encoder)
        self.args = args

        # We follow BERT's random weight initialization
        self.apply(init_bert_params)

        self.classification_heads = nn.ModuleDict()

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--encoder-layers', type=int, metavar='L',
                            help='num encoder layers')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='H',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='F',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='A',
                            help='num encoder attention heads')
        parser.add_argument('--activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')
        parser.add_argument('--pooler-activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use for pooler layer')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--activation-dropout', type=float, metavar='D',
                            help='dropout probability after activation in FFN')
        parser.add_argument('--pooler-dropout', type=float, metavar='D',
                            help='dropout probability in the masked_lm pooler layers')

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present
        base_architecture(args)

        if not hasattr(args, 'max_positions'):
            args.max_positions = args.tokens_per_sample

        encoder = RobertaEncoder(args, task.source_dictionary)
        return cls(args, encoder)

    def register_classification_head(self, name, num_classes=None, inner_dim=None, **kwargs):
        """Register a classification head."""
        self.classification_heads[name] = RobertaClassificationHead(
            self.args.encoder_embed_dim,
            inner_dim or self.args.encoder_embed_dim,
            num_classes,
            self.args.pooler_activation_fn,
            self.args.pooler_dropout,
        )

    @property
    def supported_targets(self):
        return {'self'}

    @classmethod
    def from_pretrained(cls, model_name_or_path, checkpoint_file='model.pt', data_name_or_path='.', **kwargs):
        from fairseq import hub_utils
        x = hub_utils.from_pretrained(
            model_name_or_path,
            checkpoint_file,
            data_name_or_path,
            archive_map=cls.hub_models(),
            bpe='gpt2',
            **kwargs,
        )
        return RobertaHubInterface(x['args'], x['task'], x['models'][0])

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + '.' if name != '' else ''

        # recreate any classification heads present in the state dict
        for k in state_dict.keys():
            if not k.startswith(prefix + 'classification_heads.'):
                continue
            head_name = k[len(prefix + 'classification_heads.'):].split('.')[0]
            num_classes = state_dict[
                prefix + 'classification_heads.' + head_name + '.out_proj.weight'
            ].size(0)
            inner_dim = state_dict[
                prefix + 'classification_heads.' + head_name + '.dense.weight'
            ].size(0)
            self.register_classification_head(head_name, num_classes, inner_dim)


class RobertaLMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, embed_dim, output_dim, activation_fn, weight=None):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.layer_norm = LayerNorm(embed_dim)

        if weight is None:
            weight = nn.Linear(embed_dim, output_dim, bias=False).weight
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = F.linear(x, self.weight) + self.bias

        return x


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, input_dim, inner_dim, num_classes, activation_fn, pooler_dropout):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class RobertaEncoder(FairseqDecoder):
    """RoBERTa encoder.

    Implements the :class:`~fairseq.models.FairseqDecoder` interface required
    by :class:`~fairseq.models.FairseqLanguageModel`.
    """

    def __init__(self, args, dictionary):
        super().__init__(dictionary)
        self.args = args
        self.sentence_encoder = TransformerSentenceEncoder(
            padding_idx=dictionary.pad(),
            vocab_size=len(dictionary),
            num_encoder_layers=args.encoder_layers,
            embedding_dim=args.encoder_embed_dim,
            ffn_embedding_dim=args.encoder_ffn_embed_dim,
            num_attention_heads=args.encoder_attention_heads,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.activation_dropout,
            max_seq_len=args.max_positions,
            num_segments=0,
            encoder_normalize_before=True,
            apply_bert_init=True,
            activation_fn=args.activation_fn,
        )
        self.lm_head = RobertaLMHead(
            embed_dim=args.encoder_embed_dim,
            output_dim=len(dictionary),
            activation_fn=args.activation_fn,
            weight=self.sentence_encoder.embed_tokens.weight,
        )

    def forward(self, src_tokens, features_only=False, return_all_hiddens=False, **unused):
        """
        Args:
            src_tokens (LongTensor): input tokens of shape `(batch, src_len)`
            features_only (bool, optional): skip LM head and just return
                features. If True, the output will be of shape
                `(batch, src_len, embed_dim)`.
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).

        Returns:
            tuple:
                - the LM output of shape `(batch, src_len, vocab)`
                - a dictionary of additional data, where 'inner_states'
                  is a list of hidden states.
        """
        x, extra = self.extract_features(src_tokens, return_all_hiddens)
        if not features_only:
            x = self.output_layer(x)
        return x, extra

    def extract_features(self, src_tokens, return_all_hiddens=False, **unused):
        inner_states, _ = self.sentence_encoder(
            src_tokens, last_state_only=not return_all_hiddens,
        )
        features = inner_states[-1]
        return features, {'inner_states': inner_states if return_all_hiddens else None}

    def output_layer(self, features, **unused):
        return self.lm_head(features)

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self.args.max_positions


@register_model_architecture('roberta', 'roberta')
def base_architecture(args):
    args.encoder_layers = getattr(args, 'encoder_layers', 12)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 768)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 3072)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 12)

    args.activation_fn = getattr(args, 'activation_fn', 'gelu')
    args.pooler_activation_fn = getattr(args, 'pooler_activation_fn', 'tanh')

    args.dropout = getattr(args, 'dropout', 0.1)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.0)
    args.pooler_dropout = getattr(args, 'pooler_dropout', 0.0)


@register_model_architecture('roberta', 'roberta_base')
def roberta_base_architecture(args):
    base_architecture(args)


@register_model_architecture('roberta', 'roberta_large')
def roberta_large_architecture(args):
    args.encoder_layers = getattr(args, 'encoder_layers', 24)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 1024)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 4096)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 16)
    base_architecture(args)
