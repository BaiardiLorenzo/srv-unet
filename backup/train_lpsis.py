import os, utils

args = utils.ARArgs()
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = args.CUDA_DEVICE

import numpy as np
import torch
import data_loader as dl
# courtesy of https://github.com/Po-Hsun-Su/pytorch-ssim
import pytorch_ssim  
# courtesy of https://github.com/richzhang/PerceptualSimilarity
import lpips  

from torch import nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
# courtesy of https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Super-Resolution 
from models import Discriminator, SRResNet 
from pytorch_unet import SRUnet, UNet, SimpleResNet

def configure_generator(arch_name, args):
    if arch_name == 'srunet':
        generator = SRUnet(
            3, residual=True, scale_factor=dataset_upscale_factor, n_filters=args.N_FILTERS,
            downsample=args.DOWNSAMPLE, layer_multiplier=args.LAYER_MULTIPLIER
        )
    elif arch_name == 'unet':
        generator = UNet(3, residual=True, scale_factor=dataset_upscale_factor, n_filters=args.N_FILTERS)
    elif arch_name == 'srgan':
        generator = SRResNet()
    elif arch_name == 'espcn':
        generator = SimpleResNet(n_filters=64, n_blocks=6)
    else:
        raise Exception("Unknown architecture. Select one between:", args.archs)
    
    if args.MODEL_NAME is not None:
        print("Loading model: ", args.MODEL_NAME)
        state_dict = torch.load(args.MODEL_NAME)
        generator.load_state_dict(state_dict)

    return generator


if __name__ == '__main__':
    args = utils.ARArgs()
    # torch.autograd.set_detect_anomaly(True)

    print_model = args.VERBOSE
    arch_name = args.ARCHITECTURE
    dataset_upscale_factor = args.UPSCALE_FACTOR
    epochs = args.N_EPOCHS
    crf = args.CRF
    batch_size = args.BATCH_SIZE
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # Model
    generator = configure_generator(arch_name, args)
    critic = Discriminator()
    
    # Optimizers
    critic_opt = torch.optim.Adam(lr=1e-4, params=critic.parameters())
    gan_opt = torch.optim.Adam(lr=1e-4, params=generator.parameters())

    # Metrics
    lpips_loss = lpips.LPIPS(net='vgg', version='0.1')
    lpips_alex = lpips.LPIPS(net='alex', version='0.1')
    ssim = pytorch_ssim.SSIM()

    bce = nn.BCEWithLogitsLoss()

    # Move to device
    generator.to(device)
    lpips_loss.to(device)
    lpips_alex.to(device)
    critic.to(device)

    # Data loaders
    dataset_train = dl.ARDataLoader2(path=str(args.DATASET_DIR), crf=crf, patch_size=96, eval=False, use_ar=True)
    dataset_test = dl.ARDataLoader2(path=str(args.DATASET_DIR), crf=crf, patch_size=96, eval=True, use_ar=True)

    train_loader = DataLoader(
        dataset=dataset_train, batch_size=batch_size, num_workers=4, shuffle=True, pin_memory=True
    )
    eval_loader = DataLoader(
        dataset=dataset_test, batch_size=batch_size, num_workers=4, shuffle=True, pin_memory=True
    )

    print(f"Total epochs: {epochs}; Steps per epoch: {len(train_loader)}")

    # Wandb logging
    if args.WB_NAME:
        import wandb
        wandb.init(
            project=args.WB_NAME, 
            name=f'train_lpsis_crf-{crf}', 
            config=args
        )


    # Settings weights and lambda parameters for the loss
    w0, w1, l0 = args.W0, args.W1, args.L0

    # Training loop
    for epoch in range(epochs):

        # if e == max(n_epochs - starting_epoch, 0):
        #     utils.adjust_learning_rate(critic_opt, 0.1)
        #     utils.adjust_learning_rate(gan_opt, 0.1)

        # Training
        generator.train()
        critic.train()
        for i, batch in enumerate(tqdm(train_loader, total=len(train_loader), desc=f'Training epoch {epoch}/{epochs}')):
            x, y_true = batch
            x, y_true = x.to(device), y_true.to(device)

            # Train critic
            critic_opt.zero_grad()

            y_fake = generator(x)  # Forward pass on generator
            pred_true = critic(y_true)  # Forward pass on critic for real images
            loss_true = bce(pred_true, torch.ones_like(pred_true))

            pred_fake = critic(y_fake.detach())  # Forward pass on critic for fake images
            loss_fake = bce(pred_fake, torch.zeros_like(pred_fake))

            loss_critic = (loss_true + loss_fake)*0.5

            loss_critic.backward()
            critic_opt.step()

            # Train generator
            gan_opt.zero_grad()

            loss_lpips = lpips_loss(y_fake, y_true).mean()
            ssim_val = ssim(y_fake, y_true)
            loss_ssim = 1.0 - ssim_val

            pred_fake = critic(y_fake)
            bce_gen = bce(pred_fake, torch.ones_like(pred_fake))
            content_loss = w0 * loss_lpips + w1 * loss_ssim
            loss_gen = content_loss + l0 * bce_gen

            loss_gen.backward()
            gan_opt.step()

            # Logging
            print(f"Epoch: {epoch}, "
                  f"Loss discriminator: {loss_critic:.4f}, "
                  f"Loss generator: {loss_gen:.4f}, "
                  f"Content loss: {content_loss:.4f}, "
                  f"Loss LPIPS: {loss_lpips:.4f}, "
                  f"Loss SSIM: {loss_ssim:.4f}, "
                  f"Loss BCE: {bce_gen:.4f}, "
                  f"SSIM: {ssim_val:.4f}"
                  )
            
            if args.WB_NAME:
                wandb.log({
                    "epoch": epoch,
                    "Loss discriminator": loss_critic,
                    "Loss generator": loss_gen,
                    "Loss LPIPS": loss_lpips,
                    "Loss SSIM": loss_ssim,
                    "Loss BCE": bce_gen,
                    "Content loss": content_loss,
                    "SSIM": ssim_val
                })

        # Validation
        if (epoch + 1) % args.VALIDATION_FREQ == 0:
            ssim_validation = []
            lpips_validation = []

            generator.eval()
            for i, batch in enumerate(tqdm(eval_loader, total=len(eval_loader), desc=f'Validation epoch {epoch}/{epochs}')):
                x, y_true = batch
                with torch.no_grad():
                    x, y_true = x.to(device), y_true.to(device)

                    y_fake = generator(x)

                    ssim_val = ssim(y_fake, y_true).mean()
                    lpips_val = lpips_alex(y_fake, y_true).mean()

                    ssim_validation.append(ssim_val.item())
                    lpips_validation.append(lpips_val.item())

            ssim_mean = np.mean(ssim_validation)
            lpips_mean = np.mean(lpips_validation)

            print(f"Validation SSIM: {ssim_mean:.4f}, Validation LPIPS: {lpips_mean:.4f}")

            if args.WB_NAME:
                wandb.log({
                    "epoch": epoch,
                    "Validation SSIM": ssim_mean,
                    "Validation LPIPS": lpips_mean
                })

            # Save models
            generator_path = os.path.join(
                args.EXPORT_DIR, 
                f"{arch_name}_epoch{epoch}_ssim{ssim_mean:.4f}_lpips{lpips_mean:.4f}_crf{args.CRF}.pkl"
            )
            torch.save(generator.state_dict(), generator_path)

            # having critic's weights saved was not useful, better sparing storage!
            """
            if args.SAVE_CRITIC:
                critic_path = os.path.join(
                    args.EXPORT_DIR, 
                    f"critic_epoch{epoch}_ssim{ssim_mean:.4f}_lpips{lpips_mean:.4f}_crf{args.CRF}.pkl"
                )
                torch.save(critic.state_dict(), critic_path)
            """
