from dorsalnet.paths import DERIVED_DATA
from dorsalnet.modelzoo import xception, separable_net, gabor_pyramid, dorsalnet, decoder
from dorsalnet.loaders import airsim
from dorsalnet.models import extract_subnet_dict

import argparse
import datetime
import itertools
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import torch.autograd.profiler as profiler
from torchvision import transforms
import torchvision.models as models
import torch.nn.functional as F

from dorsalnet.transforms import ThreedGaussianBlur, ThreedExposure

import wandb

from dorsalnet.paths import *


def get_all_layers(net, prefix=[]):
    if hasattr(net, "_modules"):
        lst = []
        for name, layer in net._modules.items():
            full_name = "_".join((prefix + [name]))
            lst = lst + [(full_name, layer)] + get_all_layers(layer, prefix + [name])
        return lst
    else:
        return []


def save_state(net, title, output_dir):
    datestr = str(datetime.datetime.now()).replace(":", "-")
    filename = os.path.join(output_dir, f"{title}-{datestr}.pt")
    torch.save(net.state_dict(), filename)
    return filename


def get_dataset(args):
    if args.dataset.startswith("airsim"):
        split = args.dataset.split("_")
        if len(split) > 1:
            split = split[-1]
        else:
            split = "batch1"

        trainset = airsim.AirSim(
            os.path.join(args.data_root, "airsim", split),
            split="train",
            regression=not args.softmax,
        )

        tuneset = airsim.AirSim(
            os.path.join(args.data_root, "airsim", split),
            split="tune",
            regression=not args.softmax,
        )

        train_transform = transforms.Compose(
            [
                ThreedGaussianBlur(5),
                transforms.Normalize(123.0, 75.0),
                ThreedExposure(0.3, 0.3),
            ]
        )

        eval_transform = transforms.Compose([transforms.Normalize(123.0, 75.0)])

        sz = 112
    else:
        raise NotImplementedError(f"{args.dataset} not implemented")

    return trainset, tuneset, train_transform, eval_transform, sz


def log_net(net, subnet, layers, writer, n):
    for name, layer in layers:
        if hasattr(layer, "weight"):
            writer.add_scalar(f"Weights/{name}/mean", layer.weight.mean(), n)
            writer.add_scalar(f"Weights/{name}/std", layer.weight.std(), n)
            writer.add_histogram(f"Weights/{name}/hist", layer.weight.view(-1), n)

        if hasattr(layer, "bias") and layer.bias is not None:
            writer.add_scalar(f"Biases/{name}/mean", layer.bias.mean(), n)
            writer.add_histogram(f"Biases/{name}/hist", layer.bias.view(-1), n)

    for name, param in net._parameters.items():
        writer.add_scalar(f"Weights/{name}/mean", param.mean(), n)
        writer.add_scalar(f"Weights/{name}/std", param.std(), n)
        writer.add_histogram(f"Weights/{name}/hist", param.view(-1), n)

    if hasattr(subnet, "conv1"):
        # NCHW
        if subnet.conv1.weight.ndim == 4:
            writer.add_images("Weights/conv1d/img", 0.25 * subnet.conv1.weight + 0.5, n)
        else:
            # NTCHW
            scale = 0.5 / abs(subnet.conv1.weight).max()
            writer.add_video(
                "Weights/conv1d/img",
                scale * subnet.conv1.weight.permute(0, 2, 1, 3, 4) + 0.5,
                n,
            )


def get_subnet(args, start_size):
    threed = False
    if args.submodel == "xception2d":
        subnet = xception.Xception(
            start_kernel_size=7, nblocks=args.num_blocks, nstartfeats=args.nfeats
        )
        sz = start_size // 2
        nfeats = args.nfeats

    elif args.submodel.startswith("shallownet"):

        symmetric = "symmetric" in args.submodel
        subnet = dorsalnet.ShallowNet(nstartfeats=args.nfeats, symmetric=symmetric)
        threed = True
        sz = ((start_size + 1) // 2 + 1) // 2
        nfeats = args.nfeats
    elif args.submodel.startswith("v1net"):
        subnet = dorsalnet.V1Net()
        threed = True
        sz = ((start_size + 1) // 2 + 1) // 2
        nfeats = args.nfeats

    elif args.submodel.startswith("dorsalnet"):

        symmetric = "untied" not in args.submodel
        subnet = dorsalnet.DorsalNet(symmetric, args.nfeats)

        # Lock in the shallow net features.
        # path = Path(args.ckpt_root) / 'model.ckpt-8700000-2021-01-03 22-34-02.540594.pt'
        # subnet.s1.requires_grad_(False)
        # checkpoint = torch.load(str(path))
        # subnet_dict = extract_subnet_dict(checkpoint)
        # subnet.s1.load_state_dict(subnet_dict)

        threed = True
        sz = ((start_size + 1) // 2 + 1) // 2
        nfeats = args.nfeats

    elif args.submodel.startswith("shallowdorsalnet"):

        symmetric = "untied" not in args.submodel
        subnet = dorsalnet.ShallowDorsalNet(symmetric, args.nfeats)

        # Lock in the shallow net features.
        # path = Path(args.ckpt_root) / 'model.ckpt-8700000-2021-01-03 22-34-02.540594.pt'
        # subnet.s1.requires_grad_(False)
        # checkpoint = torch.load(str(path))
        # subnet_dict = extract_subnet_dict(checkpoint)
        # subnet.s1.load_state_dict(subnet_dict)

        threed = True
        sz = ((start_size + 1) // 2 + 1) // 2
        nfeats = args.nfeats

    elif args.submodel == "gaborpyramid2d":
        subnet = nn.Sequential(
            gabor_pyramid.GaborPyramid(4), transforms.Normalize(2.2, 2.2)
        )
        sz = start_size // 2
        nfeats = args.nfeats
    elif args.submodel == "gaborpyramid3d":
        subnet = nn.Sequential(
            gabor_pyramid.GaborPyramid3d(4), transforms.Normalize(2.2, 2.2)
        )
        threed = True
        sz = start_size
        nfeats = args.nfeats
    elif args.submodel == "gaborpyramid3d_tiny":
        subnet = nn.Sequential(
            gabor_pyramid.GaborPyramid3d(2), transforms.Normalize(2.2, 2.2)
        )
        threed = True
        sz = start_size
        nfeats = args.nfeats
    return subnet, threed, sz, nfeats


def main(args):
    print("Main")
    output_dir = os.path.join(args.output_dir, args.exp_name)
    # Train a network
    try:
        os.makedirs(args.data_root)
    except FileExistsError:
        pass

    try:
        os.makedirs(output_dir)
    except FileExistsError:
        pass

    writer = SummaryWriter(comment=args.exp_name)
    writer.add_hparams(vars(args), {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        print("No CUDA! Sad!")

    trainset, tuneset, train_transform, eval_transform, start_sz = get_dataset(args)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, pin_memory=True
    )

    tuneloader = torch.utils.data.DataLoader(
        tuneset, batch_size=args.batch_size, shuffle=True, pin_memory=True
    )

    tuneloader_iter = iter(tuneloader)

    print("Init models")

    subnet, threed, sz, nfeats = get_subnet(args, start_sz)

    if args.load_conv1_weights:
        W = np.load(args.load_conv1_weights)
        subnet.conv1.weight.data = torch.tensor(W)

    subnet.to(device=device)
    if args.decoder == "average":
        net = decoder.Average(
            trainset.noutputs, trainset.nclasses, nfeats, threed=threed
        ).to(device)
    elif args.decoder == "center":
        net = decoder.Center(
            trainset.noutputs, trainset.nclasses, nfeats, threed=threed
        ).to(device)
    elif args.decoder == "point":
        net = decoder.Point(
            trainset.noutputs, trainset.nclasses, nfeats, threed=threed
        ).to(device)
    else:
        raise NotImplementedError(f"{args.decoder} not implemented")

    net.to(device=device)

    # Load a baseline with pre-trained weights
    if args.load_ckpt != "":
        net.load_state_dict(torch.load(args.load_ckpt))

    layers = get_all_layers(net)

    optimizer = optim.Adam(
        list(net.parameters()) + list(subnet.parameters()), lr=args.learning_rate
    )
    scheduler = None

    activations = {}

    def hook(name):
        def hook_fn(m, i, o):
            activations[name] = o

        return hook_fn

    if hasattr(subnet, "layers"):
        # Hook the activations
        for name, layer in subnet.layers:
            layer.register_forward_hook(hook(name))

    net.requires_grad_(True)
    subnet.requires_grad_(True)

    if args.softmax:
        loss_fun = nn.CrossEntropyLoss()
    else:
        loss_fun = nn.MSELoss()

    ll, m, n = 0, 0, 0
    tune_loss = 0.0

    running_loss = 0.0
    try:
        for epoch in range(args.num_epochs):  # loop over the dataset multiple times
            for data in trainloader: 
                net.train()

                # get the inputs; data is a list of [inputs, labels]
                # X.shape = (batch_size, C, timesteps, H, W) = (64, 3, 10, 112, 112)
                # X.max = 225, X.min = 0, X.mean ~ 120, X.std ~ 77
                # labels.shape = (batch_size, 5), (vx, vy, vz, yaw, pitch) 
                X, labels = data
                X, labels = X.to(device), labels.to(device)

                optimizer.zero_grad()

                # zero the parameter gradients
                X = train_transform(X) # shape still the same, max = 2.60, min = -2.24, mean = 0, std = 1
                X = subnet(X) # shape now = (64, 64, 10, 28, 28)
                outputs = net(X) # outputs.shape = (64, 72, 5)

                loss = loss_fun(outputs, labels)

                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()

                if not args.softmax:
                    label_mean = labels.mean()
                    writer.add_scalar("Labels/mean", label_mean, n)

                output_mean = outputs.mean()
                writer.add_scalar("Outputs/mean", output_mean, n)

                output_std = outputs.std()
                writer.add_scalar("Outputs/std", output_std, n)

                writer.add_scalar("Loss/train", loss.item(), n)

                if ll % args.print_frequency == args.print_frequency - 1:
                    log_net(net, subnet, layers, writer, n)
                    print(
                        "[%02d, %07d] average train loss: %.3f"
                        % (epoch + 1, n, running_loss / args.print_frequency)
                    )
                    running_loss = 0
                    ll = 0

                    if hasattr(subnet, "layers"):
                        for name, layer in subnet.layers:
                            writer.add_histogram(
                                f"Activations/{name}/hist",
                                activations[name].view(-1),
                                n,
                            )

                            writer.add_scalar(
                                f"Activations/{name}/mean", activations[name].mean(), n
                            )

                            writer.add_scalar(
                                f"Activations/{name}/std",
                                activations[name]
                                .permute(1, 0, 2, 3, 4)
                                .reshape(activations[name].shape[1], -1)
                                .std(dim=1)
                                .mean(),
                                n,
                            )

                if ll % 10 == 0:
                    net.eval()
                    try:
                        tune_data = next(tuneloader_iter)
                    except StopIteration:
                        tuneloader_iter = iter(tuneloader)
                        tune_data = next(tuneloader_iter)

                    # get the inputs; data is a list of [inputs, labels]
                    with torch.no_grad():
                        X, labels = tune_data
                        X, labels = X.to(device), labels.to(device)

                        X = eval_transform(X)
                        X = subnet(X)
                        outputs = net(X)

                        loss = loss_fun(outputs, labels)

                        writer.add_scalar("Loss/tune", loss.item(), n)

                        tune_loss += loss.item()
                    m += 1

                    if m == args.print_frequency:
                        print(f"tune accuracy: {tune_loss /  args.print_frequency:.3f}")
                        tune_loss = 0
                        m = 0

                if scheduler is not None:
                    scheduler.step()

                n += args.batch_size
                ll += 1

                if n % args.ckpt_frequency == 0:
                    save_state(subnet, f"model.ckpt-{n:07}", output_dir)

    except KeyboardInterrupt:
        pass

    filename = save_state(subnet, f"model.ckpt-{n:07}", output_dir)

    if args.no_wandb:
        print("Skipping W&B per config")
    else:
        if n > 10000:
            print("Saving to weight and biases")
            wandb.init(project="crcns-train_sim.py", config=vars(args))
            config = wandb.config
            wandb.watch(subnet, log="all")
            torch.save(subnet.state_dict(), os.path.join(wandb.run.dir, "model.pt"))
            print("done")
        else:
            print("Aborted too early, did not save results")


if __name__ == "__main__":
    desc = "Train a neural net"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument("--exp_name", required=True, help="Friendly name of experiment")
    parser.add_argument("--decoder", default="average", type=str, help="Decoder model")
    parser.add_argument(
        "--submodel",
        default="xception2d",
        type=str,
        help="Sub-model type (currently, either xception2d, gaborpyramid2d, gaborpyramid3d",
    )
    parser.add_argument(
        "--learning_rate", default=5e-3, type=float, help="Learning rate"
    )
    parser.add_argument(
        "--num_epochs", default=20, type=int, help="Number of epochs to train"
    )
    parser.add_argument("--image_size", default=112, type=int, help="Image size")
    parser.add_argument("--batch_size", default=1, type=int, help="Batch size")
    parser.add_argument("--nfeats", default=64, type=int, help="Number of features")
    parser.add_argument("--num_blocks", default=0, type=int, help="Num Xception blocks")
    parser.add_argument(
        "--warmup",
        default=5000,
        type=int,
        help="Number of iterations before unlocking tuning RFs and filters",
    )
    parser.add_argument(
        "--subset",
        default="-1",
        type=str,
        help="Fit data to a specific subset of the data",
    )
    parser.add_argument(
        "--ckpt_frequency", default=2500, type=int, help="Checkpoint frequency"
    )
    parser.add_argument(
        "--print_frequency", default=100, type=int, help="Print frequency"
    )
    parser.add_argument(
        "--virtual",
        default="",
        type=str,
        help="Create virtual cells by transforming the inputs (" ", rot or all)",
    )

    parser.add_argument(
        "--no_sample",
        default=False,
        help="Whether to use a normal gaussian layer rather than a sampled one",
        action="store_true",
    )
    parser.add_argument(
        "--no_wandb", default=False, help="Skip using W&B", action="store_true"
    )
    parser.add_argument(
        "--skip_existing", default=False, help="Skip existing runs", action="store_true"
    )
    parser.add_argument(
        "--softmax",
        default=False,
        help="Use softmax objective rather than regression",
        action="store_true",
    )

    parser.add_argument(
        "--load_conv1_weights", default="", help="Load conv1 weights in .npy format"
    )
    parser.add_argument("--load_ckpt", default="", help="Load checkpoint")
    parser.add_argument(
        "--dataset", default="airsim", help="Dataset (currently airsim only)"
    )
    parser.add_argument("--data_root", default=DERIVED_DATA, help="Data path")
    parser.add_argument("--ckpt_root", default=CHECKPOINTS, help="Data path")
    parser.add_argument(
        "--output_dir", default="./models", help="Output path for models"
    )

    args = parser.parse_args()
    main(args)
