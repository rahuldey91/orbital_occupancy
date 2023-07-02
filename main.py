# main.py

import os
import sys

import torch
import config
import traceback
from hal.utils import misc

import pytorch_lightning as pl
import pytorch_lightning.callbacks as cbs
from pytorch_lightning.loggers import TensorBoardLogger

from model import Model
import hal.datasets as datasets
from matplotlib import pyplot as plt
import numpy as np

def main():
    # parse the arguments
    args = config.parse_args()

    if args.ngpu == 0:
        args.device = 'cpu'

    pl.seed_everything(args.manual_seed)

    callbacks = [cbs.RichProgressBar()]
    if args.save_results:
        logger = TensorBoardLogger(
            save_dir=args.logs_dir,
            log_graph=True,
            name=args.project_name
        )
        checkpoint = cbs.ModelCheckpoint(
            dirpath=os.path.join(args.save_dir, args.project_name),
            filename=args.project_name + '-{epoch:03d}-{val_loss:.3f}',
            monitor='val_loss',
            save_top_k=args.checkpoint_max_history,
            save_weights_only=True
            )
        enable_checkpointing = True
        callbacks.append(checkpoint)
    else:
        logger=False
        checkpoint=None
        enable_checkpointing=False

    if args.swa:
        callbacks.append(cbs.StochasticWeightAveraging())

    dataloader = getattr(datasets, args.dataset)(args)
    model = Model(args, dataloader)

    if args.ngpu == 0:
        accelerator = "cpu"
        strategy = None
        sync_batchnorm = False
    else:
        accelerator = "gpu"
        strategy = 'ddp'
        sync_batchnorm = True

    trainer = pl.Trainer(
        accelerator=accelerator,
        strategy=strategy,
        sync_batchnorm=sync_batchnorm,
        benchmark=False,
        callbacks=callbacks,
        enable_checkpointing=enable_checkpointing,
        logger=logger,
        min_epochs=1,
        max_epochs=args.nepochs,
        precision=args.precision
        )

    trainer.fit(model)

    color_dict = {(0,0): 'cyan', (0,1): 'yellow', (1,0): 'red', (1,1): 'orange'}
    fig = plt.plot()
    for i in range(1000):
        x = torch.rand(64) * 10 - 5
        y = torch.rand(64) * 10 - 5
        points = torch.stack((x, y)).transpose(1,0).to(model.device)
        with torch.no_grad():
            class_c, class_e = model(points)
        outputs = torch.stack((class_c, class_e)).transpose(1,0)
        colors = []
        for j in range(outputs.shape[0]):
            colors.append(color_dict[tuple(outputs[j].numpy())])
        # colors = np.array(colors)/255
        plt.scatter(points[:,0], points[:,1], c=colors, marker='.', s=5)
    plt.savefig(f'output_plot_f{args.data_fraction}.png')
    print('Done')



if __name__ == "__main__":
    misc.setup_graceful_exit()
    try:
        main()
    except (KeyboardInterrupt, SystemExit):
        # do not print stack trace when ctrl-c is pressed
        pass
    except Exception:
        traceback.print_exc(file=sys.stdout)
    finally:
        traceback.print_exc(file=sys.stdout)
        misc.cleanup()
