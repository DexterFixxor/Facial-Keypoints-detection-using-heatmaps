import cv2
from tqdm import tqdm

import torch.nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable

from src.cpm_model import CPM
from src.dataset import FaceKeypointsDataset
import src.utils as utils

import numpy as np
from config import cfg

import time
import matplotlib.pyplot as plt


def run_epoch(ep, net, optim, data, train=True):
    mode = 'train' if train else 'validation'
    utils.create_folder(f"{cfg.OUTPUT_PATH}/heatmap_results/{mode}/epoch_{epoch + 1}/")

    loss_list = []
    print("-" * 50)
    print("Training...") if train else print("Validating...")
    print(f"Epoch [{ep + 1}/{cfg.EPOCH}]")
    start = time.time()
    if train:
        model.train()
    else:
        model.eval()

    hetmap_weight = cfg.IMG_SIZE * cfg.IMG_SIZE * 68 / 1.0
    for i, sample in tqdm(enumerate(data), total=len(data)):
        image, heatmaps = sample['image'], sample['heatmaps']
        x = Variable(image).cuda() if train else image.to(cfg.DEVICE)
        y = Variable(heatmaps).cuda() if train else image.to(cfg.DEVICE)

        h1, h2, h3, h4, h5, h6 = net(x)
        loss1 = criterion(h1, y) * hetmap_weight
        loss2 = criterion(h2, y) * hetmap_weight
        loss3 = criterion(h3, y) * hetmap_weight
        loss4 = criterion(h4, y) * hetmap_weight
        loss5 = criterion(h5, y) * hetmap_weight
        loss6 = criterion(h6, y) * hetmap_weight

        loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6
        optim.zero_grad()
        loss.backward()
        optim.step()

        loss_list.append(loss.item())

        if (i+1) % 10 == 0 or i == 0:
            fig, axis = plt.subplots(4, 2, figsize=(12, 9))
            fig.tight_layout()
            axis[0, 0].imshow(np.transpose(x.detach().cpu().numpy(), (2, 3, 1, 0))[:, :, :, 0])
            axis[0, 0].set_title('Input')

            axis[0, 1].imshow(y.detach().cpu().numpy()[0, 0, :, :])
            axis[0, 1].set_title('Ground truth')

            axis[1, 0].imshow(h1.detach().cpu().numpy()[0, 0, :, :])
            axis[1, 0].set_title('Stage 1')

            axis[1, 1].imshow(h2.detach().cpu().numpy()[0, 0, :, :])
            axis[1, 1].set_title('Stage 2')

            axis[2, 0].imshow(h3.detach().cpu().numpy()[0, 0, :, :])
            axis[2, 0].set_title('Stage 3')

            axis[2, 1].imshow(h4.detach().cpu().numpy()[0, 0, :, :])
            axis[2, 1].set_title('Stage 4')

            axis[3, 0].imshow(h5.detach().cpu().numpy()[0, 0, :, :])
            axis[3, 0].set_title('Stage 5')

            axis[3, 1].imshow(h6.detach().cpu().numpy()[0, 0, :, :])
            axis[3, 1].set_title('Stage 6')

            plt.savefig(f"{cfg.OUTPUT_PATH}/heatmap_results/{mode}/epoch_{epoch + 1}/iter_{i + 1}_loss_{loss.item():0.4f}.png")
            plt.close(fig)

    print("Epoch time: {:.4f} seconds\t Error: {:.4f}".format(time.time() - start, np.mean(loss_list)))

    mean_epoch_loss = np.mean(loss_list)
    return mean_epoch_loss


if __name__ == "__main__":
    train_samples, valid_samples = utils.split_data(csv_path=f"{cfg.DATA_ROOT_PATH}/training_frames_keypoints.csv",
                                                    portion=cfg.TEST_SPLIT)

    train_dataset = FaceKeypointsDataset(csv_data=train_samples,
                                         max_h_padding=cfg.MAX_IMG_H,
                                         max_w_padding=cfg.MAX_IMG_W,
                                         img_size=(cfg.IMG_SIZE, cfg.IMG_SIZE),
                                         img_folder=f"{cfg.DATA_ROOT_PATH}/training")

    valid_dataset = FaceKeypointsDataset(csv_data=valid_samples,
                                         max_h_padding=cfg.MAX_IMG_H,
                                         max_w_padding=cfg.MAX_IMG_W,
                                         img_size=(cfg.IMG_SIZE, cfg.IMG_SIZE),
                                         img_folder=f"{cfg.DATA_ROOT_PATH}/training")

    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE,
                              num_workers=cfg.NUM_WORKERS,
                              shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.BATCH_SIZE,
                              num_workers=cfg.NUM_WORKERS,
                              shuffle=False, drop_last=True)

    print('-' * 50)
    print("Lenght of train_data: ", len(train_dataset))
    print("Lenght of valid data: ", len(valid_dataset))
    print("Using device: ", cfg.DEVICE)
    print('-' * 50)

    model = CPM(n_keypoints=68).to(cfg.DEVICE) #68 facial keypoints
    optimizer = Adam(model.parameters(), lr=cfg.LR)
    criterion = torch.nn.MSELoss()

    utils.save_config_params()

    min_val_loss = 1e10

    train_epoch_loss = []
    valid_epoch_loss = []

    for epoch in range(cfg.EPOCH):

        train_loss = run_epoch(ep=epoch, net=model,
                               optim=optimizer,
                               data=train_loader,
                               train=True)

        train_epoch_loss.append(train_loss)

        if epoch % cfg.VALID_EACH == 0:
            valid_loss = run_epoch(ep=epoch, net=model, optim=optimizer, data=valid_loader, train=False)

            valid_epoch_loss.append(valid_loss)

    torch.save(model.state_dict(), f"{cfg.OUTPUT_PATH}/cpm_net.pt")
    plt.figure(figsize=(10, 7))
    plt.plot(train_epoch_loss, color='orange', label='train loss')
    plt.plot(valid_epoch_loss, color='red', label='validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"{cfg.OUTPUT_PATH}/loss.png")
    plt.show()
