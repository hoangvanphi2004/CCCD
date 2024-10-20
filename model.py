import torch;
from torch import nn;
import pytorch_lightning as pl;
from functools import partial
from torchvision.models.detection.retinanet import retinanet_resnet50_fpn_v2, RetinaNetRegressionHead, RetinaNetClassificationHead;
from torchvision.transforms import v2 as T

class Model(pl.LightningModule):
    def __init__(self):
        super().__init__();

        self.net = retinanet_resnet50_fpn_v2(weights = 'DEFAULT')

        head_in_feats = self.net.head.classification_head.cls_logits.in_channels;
        num_anchors_head = self.net.head.classification_head.num_anchors; 
        class_head = RetinaNetClassificationHead(in_channels=head_in_feats, num_anchors=num_anchors_head, num_classes=5, norm_layer=partial(nn.GroupNorm, 32))
        reg_head = RetinaNetRegressionHead(in_channels=head_in_feats, num_anchors=num_anchors_head, norm_layer=partial(nn.GroupNorm, 32))

        self.net.head.classification_head = class_head
        self.net.head.regression_head = reg_head
        self.transform = T.Compose([T.ToDtype(torch.float, scale=True), T.ToPureTensor()]);

        # resnet = resnet18(pretrained = True);

        # resnet.eval();
        # for param in resnet.parameters():
        #     param.requires_grad = False;
        
        # self.net = nn.Sequential(
        #     resnet,
        #     nn.ReLU(),
        #     nn.Linear(1000, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 8)
        # );

        # self.loss_fn = nn.L1Loss();
    def forward(self, x):
        x = self.transform(x)
        return self.net(x);
    def training_step(self, batch, batch_idx):
        x, y = batch;
        x = self.transform(torch.stack(x))
        loss_dicts = self.net(x, y);
        loss = sum(v for v in loss_dicts.values());
        self.log_dict({'train_loss': loss}, on_epoch = True, prog_bar = True);
        return loss;

    def validation_step(self, batch, batch_idx):
        x, y = batch;
        x = self.transform(torch.stack(x))
        loss_dicts = self.net(x, y);
        loss = sum(loss_dict["scores"].sum() for loss_dict in loss_dicts);
        self.log('val_loss', loss);
        return loss;
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr = 1e-4, weight_decay = 1e-5);

#print(*list(retinanet_resnet50_fpn_v2().head.regression_head.children()))