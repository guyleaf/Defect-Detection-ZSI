import torch
import torch.nn as nn
from .utils import ConvModule

class SemanticMaskHead(nn.Module):
    def __init__(self):
        super(SemanticMaskHead).__init__()
        pass

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
