import os

import torch
from opt import get_opts
from einops import rearrange
from dataset import YoutubeHighlightDataset
from torch.utils.data import DataLoader

from TASED import TASED_v2 
from torch.optim import Adam
from torch.nn import MSELoss

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

class VideoHighlightTrainer(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        TASED_net = TASED_v2()
        file_weight = './TASED_updated.pt'

        if os.path.isfile(file_weight):
            print ('loading weight file')
            weight_dict = torch.load(file_weight)
            model_dict = TASED_net.state_dict()
            for name, param in weight_dict.items():
                if 'module' in name:
                    name = '.'.join(name.split('.')[1:])
                if name in model_dict:
                    if param.size() == model_dict[name].size():
                        model_dict[name].copy_(param)
                    else:
                        print (' size? ' + name, param.size(), model_dict[name].size())
                else:
                    print (' name? ' + name)

            print (' loaded')
        else:
            print ('weight file?')
        
        self.Spatial_Module = TASED_net
        self.net = TASED_net
        self.criterion = MSELoss()
    
    def forward(self, x):
        return self.net(x)
    
    def setup(self, stage=None):
        self.train_dataset = YoutubeHighlightDataset(dataset = 'YouTube_Highlights', 
                                                    split='train', 
                                                    category = 'surfing', 
                                                    data_path = self.hparams.frame_path, 
                                                    L = 32
                                                    )
        
        self.val_dataset = YoutubeHighlightDataset(dataset = 'YouTube_Highlights', 
                                                    split='test', 
                                                    category = 'surfing', 
                                                    data_path = self.hparams.frame_path, 
                                                    L = 32
                                                    )
        
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=4,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=4,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True)
    
    def configure_optimizers(self):
        self.opt = Adam(self.net.parameters(), lr=self.hparams.lr)
        return self.opt
    
    def training_step(self, batch, batch_idx):
        x, label = batch
        y_pred = self(x)
        smap = self.Spatial_Module(x)
        mask = smap > 0.0005
        y_gt = label*torch.mul(smap,mask)
        loss = self.criterion(y_pred, y_gt)
        self.log('train/loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['level0'].float()
        item = self(x)
        loss = 0
        for i in range(self.net.level):
            HR_pred = item[i]
            loss += self.criterion(HR_pred, batch['level'+str(i+1)])
        
        psnr_ = psnr(HR_pred, batch['level'+str(self.net.level)])

        log = {'val_loss': loss,
               'val_psnr': psnr_,
               'HR_gt' : batch['level'+str(self.net.level)][0],
               'HR_pred' : item[self.net.level-1][0]}

        return log

    def validation_epoch_end(self, outputs):
        mean_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        mean_psnr = torch.stack([x['val_psnr'] for x in outputs]).mean()

        HR_gt = outputs[0]['HR_gt']
        HR_pred = outputs[0]['HR_pred']

        HR_gt = rearrange(HR_gt, 'c h w -> 1 c h w')
        HR_pred = rearrange(HR_pred, 'c h w -> 1 c h w')

        self.logger.experiment.add_images('val/pred', HR_pred)
        self.logger.experiment.add_images('val/gt', HR_gt)

        self.log('val/loss', mean_loss, prog_bar=True)
        self.log('val/psnr', mean_psnr, prog_bar=True)

if __name__ == '__main__':
    hparams = get_opts()
    system = VideoHighlightTrainer(hparams)

    pbar = TQDMProgressBar(refresh_rate=1)
    callbacks = [pbar]

    logger = TensorBoardLogger(save_dir="logs",
                               name=hparams.exp_name,
                               default_hp_metric=False)

    trainer = Trainer(max_epochs=hparams.num_epochs,
                      callbacks=callbacks,
                      logger=logger,
                      enable_model_summary=True,
                      accelerator='auto',
                      devices=1,
                      num_sanity_val_steps=0,
                      log_every_n_steps=1,
                      check_val_every_n_epoch=20,
                      benchmark=True)

    trainer.fit(system)