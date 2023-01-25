import torch
import numpy as np
import torch.nn as nn
from collections import OrderedDict

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pytorch_model_summary import summary
from tqdm import tqdm
import sys
import datetime
from data_feed import DataFeed
from model import GruModel


def train_model(num_epoch=1000, if_writer=False, portion=1.):
    num_classes = 64
    batch_size = 32
    val_batch_size = 64
    train_dir = './train_seqs.csv'
    val_dir = './test_seqs.csv'
    seq_len = 8
    train_loader = DataLoader(DataFeed(train_dir, seq_len, portion=portion), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(DataFeed(val_dir, seq_len, init_shuffle=False), batch_size=val_batch_size, shuffle=False)

    # check gpu acceleration availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print(device)

    now = datetime.datetime.now().strftime("%H_%M_%S")
    date = datetime.date.today().strftime("%y_%m_%d")

    # Instantiate the model
    net = GruModel(num_classes)
    # path to save the model
    PATH = './checkpoint/' + now + '_' + date + '_' + net.name + '' + '.pth'
    # print model summary
    if if_writer:
        h = net.initHidden(1)
        print(summary(net, torch.zeros((10, 1, 216)), h))
    # send model to GPU
    net.to(device)
    
    # set up loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 15])
    
    if if_writer:
        writer = SummaryWriter(comment=now + '_' + date + '_' + net.name)
    
    # train model
    for epoch in range(num_epoch):  # loop over the dataset multiple times
        net.train()
        running_loss = 0.
        running_acc = 1.
        with tqdm(train_loader, unit="batch", file=sys.stdout) as tepoch:
            for i, (lidar_img, beam, label) in enumerate(tepoch, 0):
                tepoch.set_description(f"Epoch {epoch}")
                # get the input np arrays, lidar_img sequence (batch_size, 8, 216) label (batch_size, 1)
                lidar_img = torch.swapaxes(lidar_img, 0, 1)
                lidar_img = torch.cat([lidar_img, torch.zeros_like(lidar_img[:3, ...])], dim=0)
                label = torch.cat([beam[..., -1:], label], dim=-1)
                label = torch.swapaxes(label, 0, 1)
                lidar_img = lidar_img.to(device)
                label = label.to(device)
                optimizer.zero_grad()
                # forward + backward + optimize
                h = net.initHidden(lidar_img.shape[1]).to(device)
                outputs, _ = net(lidar_img, h)
                outputs = outputs[-4:, ...]
                loss = criterion(outputs.view(-1, num_classes), label.flatten())
                prediction = torch.argmax(outputs, dim=-1)
                acc = (prediction == label).sum().item() / int(torch.sum(label!=-100).cpu())
                loss.backward()
                optimizer.step()
                # print statistics
                running_loss = (loss.item() + i * running_loss) / (i + 1)
                running_acc = (acc + i * running_acc) / (i + 1)
                log = OrderedDict()
                log['loss'] = running_loss
                log['acc'] = running_acc
                tepoch.set_postfix(log)
            scheduler.step()
            # validation
            predictions = []
            net.eval()
            with torch.no_grad():
                total = np.zeros((4,))
                top1_correct = np.zeros((4,))
                top2_correct = np.zeros((4,))
                top3_correct = np.zeros((4,))
                top5_correct = np.zeros((4,))
                val_loss = 0
                for (lidar_img, beam, label) in val_loader:
                    lidar_img = torch.swapaxes(lidar_img, 0, 1)
                    lidar_img = torch.cat([lidar_img, torch.zeros_like(lidar_img[:3, ...])], dim=0)
                    label = torch.cat([beam[..., -1:], label], dim=-1)
                    label = torch.swapaxes(label, 0, 1)
                    lidar_img = lidar_img.to(device)
                    label = label.to(device)
                    # forward + backward + optimize
                    h = net.initHidden(lidar_img.shape[1]).to(device)
                    outputs, _ = net(lidar_img, h)
                    outputs = outputs[-4:, ...]
                    val_loss += nn.CrossEntropyLoss(reduction='sum')(outputs.view(-1, num_classes), label.flatten()).item()
                    total += torch.sum(label!=-100, dim=-1).cpu().numpy()
                    prediction = torch.argmax(outputs, dim=-1)
                    top1_correct += torch.sum(prediction == label, dim=-1).cpu().numpy()
                    _, idx = torch.topk(outputs, 5, dim=-1)
                    idx = idx.cpu().numpy()
                    label = label.cpu().numpy()
                    for i in range(label.shape[0]):
                        for j in range(label.shape[1]):
                            top2_correct[i] += np.isin(label[i, j], idx[i, j, :2]).sum()
                            top3_correct[i] += np.isin(label[i, j], idx[i, j, :3]).sum()
                            top5_correct[i] += np.isin(label[i, j], idx[i, j, :5]).sum()
                    predictions.append(prediction.cpu().numpy())
                val_loss /= total.sum()
                val_top1_acc = top1_correct / total
                val_top2_acc = top2_correct / total
                val_top3_acc = top3_correct / total
                val_top5_acc = top5_correct / total
                print('val_loss={:.4f}'.format(val_loss), flush=True)
                print('accuracy', flush=True)
                print(np.stack([val_top1_acc, val_top2_acc, val_top3_acc, val_top5_acc], 0), flush=True)
        if if_writer:
            writer.add_scalar('Loss/train', running_loss, epoch)
            writer.add_scalar('Loss/test', val_loss, epoch)
            writer.add_scalar('acc/train', running_acc, epoch)
            writer.add_scalar('acc/test', val_top1_acc[0], epoch)
    if if_writer:
        writer.close()
    torch.save(net.state_dict(), PATH)
    print('Finished Training')

    net.to(device)
    net.load_state_dict(torch.load(PATH))

    net.eval()
    # test
    predictions = []
    raw_predictions = []
    out_label = []
    net.eval()
    with torch.no_grad():
        total = np.zeros((4,))
        top1_correct = np.zeros((4,))
        top2_correct = np.zeros((4,))
        top3_correct = np.zeros((4,))
        top5_correct = np.zeros((4,))
        val_loss = 0
        for (lidar_img, beam, label) in val_loader:
            lidar_img = torch.swapaxes(lidar_img, 0, 1)
            lidar_img = torch.cat([lidar_img, torch.zeros_like(lidar_img[:3, ...])], dim=0)
            label = torch.cat([beam[..., -1:], label], dim=-1)
            label = torch.swapaxes(label, 0, 1)

            lidar_img = lidar_img.to(device)
            label = label.to(device)

            # forward + backward + optimize
            h = net.initHidden(lidar_img.shape[1]).to(device)
            outputs, _ = net(lidar_img, h)
            outputs = outputs[-4:, ...]
            val_loss += nn.CrossEntropyLoss(reduction='sum')(outputs.view(-1, num_classes), label.flatten()).item()
            total += torch.sum(label != -100, dim=-1).cpu().numpy()
            prediction = torch.argmax(outputs, dim=-1)
            top1_correct += torch.sum(prediction == label, dim=-1).cpu().numpy()

            _, idx = torch.topk(outputs, 5, dim=-1)
            idx = idx.cpu().numpy()
            label = label.cpu().numpy()
            for i in range(label.shape[0]):
                for j in range(label.shape[1]):
                    top2_correct[i] += np.isin(label[i, j], idx[i, j, :2]).sum()
                    top3_correct[i] += np.isin(label[i, j], idx[i, j, :3]).sum()
                    top5_correct[i] += np.isin(label[i, j], idx[i, j, :5]).sum()

            predictions.append(prediction.cpu().numpy())
            raw_predictions.append(outputs.cpu().numpy())
            out_label.append(label)

        val_loss /= total.sum()
        val_top1_acc = top1_correct / total
        val_top2_acc = top2_correct / total
        val_top3_acc = top3_correct / total
        val_top5_acc = top5_correct / total

        predictions = np.concatenate(predictions, 1)
        raw_predictions = np.concatenate(raw_predictions, 1)
        out_label = np.concatenate(out_label, 1)

        val_acc = {'top1': val_top1_acc, 'top2': val_top2_acc, 'top3': val_top3_acc, 'top5': val_top5_acc}
        return val_loss, val_acc, predictions, raw_predictions


if __name__ == "__main__":
    torch.manual_seed(0)
    val_loss, val_acc, predictions, raw_predictions = train_model(num_epoch=20, if_writer=True, portion=1.)
    print(val_acc['top1'])
    print(val_acc['top5'])
    print('complete')




