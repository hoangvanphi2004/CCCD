import torch;
import pytorch_lightning as pl;
from data_customize import DataModule;
from model import Model;
import matplotlib.pyplot as plt;
import matplotlib.patches as patches;
from pytorch_lightning.loggers import TensorBoardLogger;
from sys import argv

def train_model():
    data = DataModule(batch_size = 4, num_workers = 0);    
    model = Model()

    logger = TensorBoardLogger("tb_logs", name = "FilterLogger");
    trainer = pl.Trainer(logger = logger, profiler = "simple", accelerator = "gpu", devices = 1, min_epochs = 2, max_epochs = 3);
    trainer.fit(model, data);
    trainer.validate(model, data);
    
    torch.save(model.state_dict(), "model.pth");
    #torch.save(model.state_dict(), argv[1]);
train_model()