import torch
import torch.nn as nn
import torchvision
from models.__init__ import *
from models.lstm import LSTMClassifier
from models.transformer_encoder import TransformerEncoderClassifier
from models.mamba_model import MambaModel
from models.protonet import ProtoNet
from models.relationnet import RelationNet


def get_model(model_name, num_class, pretrained_path, device, **kwargs):
    """获取指定模型架构并加载预训练权重（如有提供）
    Args:
        model_name (str): 模型名称，如 'resnet34'
        num_class (int): 分类类别数
        pretrained_path (str): 预训练权重路径
        device (str): 设备类型 ('cpu' 或 'cuda')
        **kwargs: 额外的模型参数（如 hidden_size, window_size 等）
    Returns:
        torch.nn.Module: 构建好的模型
    """

    if model_name == 'resnet34':
        model = torchvision.models.resnet34(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, num_class)
        model = model.to(device)

    # ======================== 时序基线模型 ======================== #
    elif model_name == 'lstm':
        model = LSTMClassifier(
            input_size=kwargs.get('input_size', 1),
            hidden_size=kwargs.get('hidden_size', 64),
            num_layers=kwargs.get('num_layers', 2),
            num_classes=num_class,
            dropout=kwargs.get('dropout', 0.3),
            bidirectional=kwargs.get('bidirectional', False),
        )
        model = model.to(device)

    elif model_name == 'transformer':
        model = TransformerEncoderClassifier(
            input_size=kwargs.get('input_size', 1),
            d_model=kwargs.get('d_model', 64),
            nhead=kwargs.get('nhead', 4),
            num_layers=kwargs.get('num_layers', 2),
            dim_feedforward=kwargs.get('dim_feedforward', 128),
            num_classes=num_class,
            dropout=kwargs.get('dropout', 0.3),
            max_seq_len=kwargs.get('max_seq_len', 500),
        )
        model = model.to(device)

    elif model_name == 'mamba':
        model = MambaModel(
            input_size=kwargs.get('input_size', 1),
            d_model=kwargs.get('d_model', 64),
            n_layer=kwargs.get('n_layer', 2),
            d_conv=kwargs.get('d_conv', 4),
            d_state=kwargs.get('d_state', 16),
            expand=kwargs.get('expand', 2),
            num_classes=num_class,
            dropout=kwargs.get('dropout', 0.3),
        )
        model = model.to(device)

    # ======================== 小样本学习模型 ======================== #
    elif model_name == 'protonet':
        model = ProtoNet(
            input_size=kwargs.get('input_size', 30),
            hidden_size=kwargs.get('hidden_size', 64),
            embedding_size=kwargs.get('embedding_size', 32),
            num_classes=num_class,
            dropout=kwargs.get('dropout', 0.2),
        )
        model = model.to(device)

    elif model_name == 'relationnet':
        model = RelationNet(
            input_size=kwargs.get('input_size', 30),
            hidden_size=kwargs.get('hidden_size', 64),
            embedding_size=kwargs.get('embedding_size', 32),
            relation_size=kwargs.get('relation_size', 16),
            num_classes=num_class,
            dropout=kwargs.get('dropout', 0.2),
        )
        model = model.to(device)

    # ======================== 域适应模型 ======================== #
    elif model_name == 'mlda':
        from models.mlda_model import MLDAModel
        model = MLDAModel(
            input_dim=kwargs.get('input_size', 768),
            num_classes=num_class,
            feat_dim=kwargs.get('feat_dim', 32),
            dropout=kwargs.get('dropout', 0.05),
        )
        model = model.to(device)

    elif model_name == 'daeevit':
        from models.daeevit_model import DAEEGViTModel
        model = DAEEGViTModel(
            seq_len=kwargs.get('seq_len', 256),
            patch_size=kwargs.get('patch_size', 32),
            in_channels=kwargs.get('in_channels', 3),
            num_classes=num_class,
            embed_dim=kwargs.get('embed_dim', 64),
            depth=kwargs.get('depth', 4),
            num_heads=kwargs.get('num_heads', 4),
            mlp_ratio=kwargs.get('mlp_ratio', 4.0),
            qkv_bias=kwargs.get('qkv_bias', True),
            drop_ratio=kwargs.get('dropout', 0.1),
            attn_drop_ratio=kwargs.get('attn_drop_ratio', 0.0),
            drop_path_ratio=kwargs.get('drop_path_ratio', 0.1),
            mbconv_expand_ratio=kwargs.get('mbconv_expand_ratio', 4),
            mbconv_se_ratio=kwargs.get('mbconv_se_ratio', 0.25),
            representation_size=kwargs.get('representation_size', None),
        )
        model = model.to(device)

    elif model_name == 'lamsda':
        from models.lamsda_model import LAMSDAModel
        model = LAMSDAModel(
            in_channels=kwargs.get('in_channels', 3),
            seq_len=kwargs.get('seq_len', 256),
            num_classes=num_class,
            num_sources=kwargs.get('num_sources', 5),
            feature_dim=kwargs.get('feature_dim', 64),
            ds_hidden_dim=kwargs.get('ds_hidden_dim', 256),
        )
        model = model.to(device)

    # ======================== 经典域适应方法 ======================== #
    elif model_name == 'dann':
        from models.dann_model import DANNModel
        model = DANNModel(
            input_dim=kwargs.get('input_size', 768),
            num_classes=num_class,
            feat_dim=kwargs.get('feat_dim', 32),
            dropout=kwargs.get('dropout', 0.05),
            domain_hidden=kwargs.get('domain_hidden', 1024),
        )
        model = model.to(device)

    elif model_name == 'deepcoral':
        from models.deepcoral_model import DeepCORALModel
        model = DeepCORALModel(
            input_dim=kwargs.get('input_size', 768),
            num_classes=num_class,
            feat_dim=kwargs.get('feat_dim', 32),
            dropout=kwargs.get('dropout', 0.05),
        )
        model = model.to(device)

    # ======================== 域泛化方法 ======================== #
    elif model_name == 'interpcnn':
        from models.interpcnn_model import InterpretableCNN
        model = InterpretableCNN(
            in_channels=kwargs.get('in_channels', 3),
            seq_len=kwargs.get('seq_len', 256),
            num_classes=num_class,
            n_filters=kwargs.get('n_filters', 16),
            depth_multiplier=kwargs.get('depth_multiplier', 2),
            kernel_size=kwargs.get('kernel_size', 64),
            dropout=kwargs.get('dropout', 0.0),
        )
        model = model.to(device)

    else:
        raise ValueError(f"Model name '{model_name}' is not recognized.")

    return model

