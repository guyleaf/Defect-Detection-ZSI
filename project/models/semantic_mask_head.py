import torch
import torch.nn as nn
import  numpy as np
from .utils import ConvModule

class SemanticMaskHead(nn.Module):
    def __init__(self, 
        num_convs=4,
        roi_feat_size=14,
        in_channels=256,
        conv_kernel_size=3,
        conv_out_channels=256,
        upsample_method='deconv',
        upsample_ratio=2,
        num_classes=4,
        voc_path= None,
        vec_path='data/coco/word_w2v_withbg_65_15.txt',
        with_decoder=True,
        sync_bg=True,
        conv_cfg=None,
        norm_cfg=None,
    ):
        super(SemanticMaskHead).__init__()
        self.num_convs = num_convs
        self.in_channels = in_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_out_channels = conv_out_channels
        self.upsample_method = upsample_method
        self.upsample_ratio = upsample_ratio
        self.with_decoder = with_decoder
        self.sync_bg = sync_bg
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        

        # conv for upsampling
        self.convs = nn.ModuleList()
        for i in range(self.num_convs):
            in_channels = (self.in_channels if i == 0 else self.conv_out_channels)
            padding = (self.conv_kernel_size - 1) // 2
            self.convs.append(
                ConvModule(
                    in_channels,
                    self.conv_out_channels,
                    self.conv_kernel_size,
                    padding=padding,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg
                )
            )
        # convT for encoder
        self.convT = ConvModule(
                in_channels,
                300,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg
            )
        # dconvT for decoder
        if self.with_decoder:
            self.dconvT = ConvModule(
                        300,
                        in_channels,
                        3,
                        padding=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg
                    )

        upsample_in_channels = (self.conv_out_channels if self.num_convs > 0 else in_channels)

        if self.upsample_method is None:
            self.upsample = None

        elif self.upsample_method == 'deconv':
            self.upsample = nn.ConvTranspose2d(
                    upsample_in_channels,
                    self.conv_out_channels,
                    self.upsample_ratio,
                    stride=self.upsample_ratio
                )
        else:
            self.upsample = nn.Upsample(scale_factor=self.upsample_ratio, mode=self.upsample_method)

        # relu
        self.relu = nn.ReLU(inplace=True)

        # word related
        if voc_path is not None:
            voc = np.loadtxt(voc_path, dtype='float32', delimiter=',')
        else:
            voc = None

        vec_load = np.loadtxt(vec_path, dtype='float32', delimiter=',')
    
    def forward(self, x, bg_vector=None):
        # replace bg by ba-rpn
        if bg_vector and self.sync_bg:
            with torch.no_grad():
                self.conv_vec.weight.data[0] = bg_vector[0]
                if not self.seen_class:
                    self.conv_vec_unseen.weight.data[0] = bg_vector[0]
        
        # upsampling 
        for conv in self.convs:
            conv4_x = conv(x)
        if self.upsample is not None:
            conv4_x = self.upsample(conv4_x)
            if self.upsample_method == 'deconv':
                conv4_x = self.relu(conv4_x)

        # encoder
        x = self.convT(conv4_x)
        if self.voc is not None:
            x = self.conv_voc(x)
        if self.with_learnable_kernel:
            x = self.kernel_semantic(x)

        # decoder
        if self.with_decoder:
            if self.with_learnable_kernel:
                d_x = self.d_kernel_semantic(x)
            d_x = self.dconvT(d_x)

        # classification module
        mask_pred_seen = self.conv_vec(x)
        if not self.seen_class and not self.gzsd:
            mask_pred = self.con_vec_t(mask_pred_seen)
            mask_pred_unseen = self.conv_vec_unseen(mask_pred)
            return mask_pred_unseen
        if self.gzsd:
            mask_pred = self.con_vec_t(mask_pred_seen)
            mask_pred_unseen = self.conv_vec_unseen(mask_pred)
            return mask_pred_seen, mask_pred_unseen
        if not self.with_decoder:
            return mask_pred_seen
        else:
            return mask_pred_seen, conv4_x, d_x
        
        # TODO compute reconstructed error
        # TODO compute BCE loss
    
    def init_weights(self):
        pass
