


import torch
import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from torch.utils.tensorboard import SummaryWriter

###############################################################################
# define model architectures
# ##############################################################################
def compute_outdim(i_dim, stride, kernel, padding, dilation):
    o_dim = (i_dim + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1
    return o_dim

    # print("=> creating model CNN")
    #         model = CNN(
    #             channels_in=config["model::channels_in"],
    #             nlin=config["model::nlin"],
    #             dropout=config["model::dropout"],
    #             init_type=config["model::init_type"],
    #         )

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # seeds = config['seed']
        # seed_everything(seeds)

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.LeakyReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.LeakyReLU(),
            nn.Linear(in_features=84, out_features=10),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        # probs = F.softmax(logits, dim=1)
        return logits

class CNN(nn.Module):
    def __init__(self, channels_in, nlin="leakyrelu", dropout=0.2, init_type="uniform",):
        super().__init__()
        # print(channels_in)
        # init module list
        self.module_list = nn.ModuleList()
        ### ASSUMES 28x28 image size
        ## compose layer 1
        self.module_list.append(nn.Conv2d(channels_in, 8, 5))
        self.module_list.append(nn.MaxPool2d(2, 2))
        self.module_list.append(self.get_nonlin(nlin))
        # apply dropout
        if dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        ## compose layer 2
        self.module_list.append(nn.Conv2d(8, 6, 5))
        self.module_list.append(nn.MaxPool2d(2, 2))
        self.module_list.append(self.get_nonlin(nlin))
        ## add dropout
        if dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        ## compose layer 3
        self.module_list.append(nn.Conv2d(6, 4, 2))
        self.module_list.append(self.get_nonlin(nlin))
        ## add flatten layer
        self.module_list.append(nn.Flatten())
        ## add linear layer 1
        self.module_list.append(nn.Linear(3 * 3 * 4, 20))
        self.module_list.append(self.get_nonlin(nlin))
        ## add dropout
        if dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        ## add linear layer 1
        self.module_list.append(nn.Linear(20, 10))

        ### initialize weights with se methods
        # self.initialize_weights(init_type)

    def initialize_weights(self, init_type):
        # print("initialze model")
        for m in self.module_list:
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                if init_type == "xavier_uniform":
                    torch.nn.init.xavier_uniform_(m.weight)
                if init_type == "xavier_normal":
                    torch.nn.init.xavier_normal_(m.weight)
                if init_type == "uniform":
                    torch.nn.init.uniform_(m.weight)
                if init_type == "normal":
                    torch.nn.init.normal_(m.weight)
                if init_type == "kaiming_normal":
                    torch.nn.init.kaiming_normal_(m.weight)
                if init_type == "kaiming_uniform":
                    torch.nn.init.kaiming_uniform_(m.weight)
                # set bias to some small non-zero value
                m.bias.data.fill_(0.01)

    def get_nonlin(self, nlin):
        # apply nonlinearity
        if nlin == "leakyrelu":
            return nn.LeakyReLU()
        if nlin == "relu":
            return nn.ReLU()
        if nlin == "tanh":
            return nn.Tanh()
        if nlin == "sigmoid":
            return nn.Sigmoid()
        if nlin == "silu":
            return nn.SiLU()
        if nlin == "gelu":
            return nn.GELU()

    def forward(self, x):
        # forward prop through module_list
        n = len(self.module_list)
        out = None
        for i, layer in enumerate(self.module_list):
            x = layer(x)
            if i==(n-3):
                out = x
        return x, out

    def forward_activations(self, x):
        # forward prop through module_list
        activations = []
        for layer in self.module_list:
            x = layer(x)
            if (
                isinstance(layer, nn.Tanh)
                or isinstance(layer, nn.Sigmoid)
                or isinstance(layer, nn.ReLU)
                or isinstance(layer, nn.LeakyReLU)
                or isinstance(layer, nn.LeakyReLU)
                or isinstance(layer, nn.SiLU)
                or isinstance(layer, nn.GELU)
            ):
                activations.append(x)
        return x, activations


class CNN2(nn.Module):
    def __init__(
        self,
        channels_in,
        nlin="leakyrelu",
        dropout=0.2,
        init_type="uniform",
    ):
        super().__init__()
        # init module list
        self.module_list = nn.ModuleList()
        ### ASSUMES 28x28 image size
        ## compose layer 1
        self.module_list.append(nn.Conv2d(channels_in, 6, 5))
        self.module_list.append(nn.MaxPool2d(2, 2))
        self.module_list.append(self.get_nonlin(nlin))
        # apply dropout
        if dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        ## compose layer 2
        self.module_list.append(nn.Conv2d(6, 9, 5))
        self.module_list.append(nn.MaxPool2d(2, 2))
        self.module_list.append(self.get_nonlin(nlin))
        ## add dropout
        if dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        ## compose layer 3
        self.module_list.append(nn.Conv2d(9, 6, 2))
        self.module_list.append(self.get_nonlin(nlin))
        ## add flatten layer
        self.module_list.append(nn.Flatten())
        ## add linear layer 1
        self.module_list.append(nn.Linear(3 * 3 * 6, 20))
        self.module_list.append(self.get_nonlin(nlin))
        ## add dropout
        if dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        ## add linear layer 1
        self.module_list.append(nn.Linear(20, 10))

        ### initialize weights with se methods
        self.initialize_weights(init_type)

    def initialize_weights(self, init_type):
        # print("initialze model")
        for m in self.module_list:
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                if init_type == "xavier_uniform":
                    torch.nn.init.xavier_uniform_(m.weight)
                if init_type == "xavier_normal":
                    torch.nn.init.xavier_normal_(m.weight)
                if init_type == "uniform":
                    torch.nn.init.uniform_(m.weight)
                if init_type == "normal":
                    torch.nn.init.normal_(m.weight)
                if init_type == "kaiming_normal":
                    torch.nn.init.kaiming_normal_(m.weight)
                if init_type == "kaiming_uniform":
                    torch.nn.init.kaiming_uniform_(m.weight)
                # set bias to some small non-zero value
                m.bias.data.fill_(0.01)

    def get_nonlin(self, nlin):
        # apply nonlinearity
        if nlin == "leakyrelu":
            return nn.LeakyReLU()
        if nlin == "relu":
            return nn.ReLU()
        if nlin == "tanh":
            return nn.Tanh()
        if nlin == "sigmoid":
            return nn.Sigmoid()
        if nlin == "silu":
            return nn.SiLU()
        if nlin == "gelu":
            return nn.GELU()

    def forward(self, x):
        # forward prop through module_list
        for layer in self.module_list:
            x = layer(x)
        return x

    def forward_activations(self, x):
        # forward prop through module_list
        activations = []
        for layer in self.module_list:
            x = layer(x)
            if isinstance(layer, nn.Tanh):
                activations.append(x)
        return x, activations


class CNN3(nn.Module):
    def __init__(
        self,
        channels_in,
        n_class=10,
        nlin="leakyrelu",
        dropout=0.0,
        init_type="uniform",
    ):
        super().__init__()
        # init module list
        self.n_class= n_class
        self.module_list = nn.ModuleList()
        ### ASSUMES 32x32 image size
        ## chn_in * 32 * 32
        ## compose layer 0
        self.module_list.append(nn.Conv2d(channels_in, 16, 3))
        self.module_list.append(nn.MaxPool2d(2, 2))
        self.module_list.append(self.get_nonlin(nlin))
        # apply dropout
        if True:  # dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        ## 16 * 15 * 15
        ## compose layer 1
        self.module_list.append(nn.Conv2d(16, 32, 3))
        self.module_list.append(nn.MaxPool2d(2, 2))
        self.module_list.append(self.get_nonlin(nlin))
        # apply dropout
        if True:  # dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        ## 32 * 7 * 7 // 32 * 6 * 6
        ## compose layer 2
        self.module_list.append(nn.Conv2d(32, 15, 3))
        self.module_list.append(nn.MaxPool2d(2, 2))
        self.module_list.append(self.get_nonlin(nlin))
        ## add dropout
        if True:  # dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        ## 15 * 2 * 2
        self.module_list.append(nn.Flatten())
        ## add linear layer 1
        self.module_list.append(nn.Linear(15 * 2 * 2, 20))
        self.module_list.append(self.get_nonlin(nlin))
        ## add dropout
        if True:  # dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        ## add linear layer 1
        self.module_list.append(nn.Linear(20, n_class))

        ### initialize weights with se methods
        # self.initialize_weights(init_type)

    # def initialize_weights(self, init_type):
    #     # print("initialze model")
    #     for m in self.module_list:
    #         if type(m) == nn.Linear or type(m) == nn.Conv2d:
    #             if init_type == "xavier_uniform":
    #                 torch.nn.init.xavier_uniform_(m.weight)
    #             if init_type == "xavier_normal":
    #                 torch.nn.init.xavier_normal_(m.weight)
    #             if init_type == "uniform":
    #                 torch.nn.init.uniform_(m.weight)
    #             if init_type == "normal":
    #                 torch.nn.init.normal_(m.weight)
    #             if init_type == "kaiming_normal":
    #                 torch.nn.init.kaiming_normal_(m.weight)
    #             if init_type == "kaiming_uniform":
    #                 torch.nn.init.kaiming_uniform_(m.weight)
    #             # set bias to some small non-zero value
    #             m.bias.data.fill_(0.01)

    def get_nonlin(self, nlin):
        # apply nonlinearity
        if nlin == "leakyrelu":
            return nn.LeakyReLU()
        if nlin == "relu":
            return nn.ReLU()
        if nlin == "tanh":
            return nn.Tanh()
        if nlin == "sigmoid":
            return nn.Sigmoid()
        if nlin == "silu":
            return nn.SiLU()
        if nlin == "gelu":
            return nn.GELU()

    def forward(self, x):
        # forward prop through module_list
        for layer in self.module_list:
            x = layer(x)
        return x

    def forward_activations(self, x):
        # forward prop through module_list
        activations = []
        for layer in self.module_list:
            x = layer(x)
            if (
                isinstance(layer, nn.Tanh)
                or isinstance(layer, nn.Sigmoid)
                or isinstance(layer, nn.ReLU)
                or isinstance(layer, nn.LeakyReLU)
                or isinstance(layer, nn.LeakyReLU)
                or isinstance(layer, nn.SiLU)
                or isinstance(layer, nn.GELU)
            ):
                activations.append(x)
        return x, activations


from torchvision.models.resnet import ResNet, BasicBlock


class ResNet18(ResNet):
    def __init__(
        self,
        channels_in=3,
        out_dim=10,
        nlin="relu",  # doesn't yet do anything
        dropout=0.2,  # doesn't yet do anything
        init_type="kaiming_uniform",
    ):
        # call init from parent class
        super().__init__(block=BasicBlock, layers=[2, 2, 2, 2], num_classes=out_dim)
        # adpat first layer to fit dimensions
        self.conv1 = nn.Conv2d(
            channels_in,
            64,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )
        self.maxpool = nn.Identity()

        if init_type is not None:
            self.initialize_weights(init_type)

    def initialize_weights(self, init_type):
        """
        applies initialization method on all layers in the network
        """
        for m in self.modules():
            m = self.init_single(init_type, m)

    def init_single(self, init_type, m):
        """
        applies initialization method on module object
        """
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            if init_type == "xavier_uniform":
                torch.nn.init.xavier_uniform_(m.weight)
            if init_type == "xavier_normal":
                torch.nn.init.xavier_normal_(m.weight)
            if init_type == "uniform":
                torch.nn.init.uniform_(m.weight)
            if init_type == "normal":
                torch.nn.init.normal_(m.weight)
            if init_type == "kaiming_normal":
                torch.nn.init.kaiming_normal_(m.weight)
            if init_type == "kaiming_uniform":
                torch.nn.init.kaiming_uniform_(m.weight)
            # set bias to some small non-zero value
            try:
                m.bias.data.fill_(0.01)
            except Exception as e:
                pass
        return m




###############################################################################
# define FNNmodule
# ##############################################################################
# class NNmodule(nn.Module):
#     def __init__(self, config, cuda=False, seed=42, verbosity=0):
#         super(NNmodule, self).__init__()
#
#         # set verbosity
#         self.verbosity = verbosity
#
#         if cuda and torch.cuda.is_available():
#             self.cuda = True
#             if self.verbosity > 0:
#                 print("cuda availabe:: send model to GPU")
#         else:
#             self.cuda = False
#             if self.verbosity > 0:
#                 print("cuda unavailable:: train model on cpu")
#
#         # setting seeds for reproducibility
#         # https://pytorch.org/docs/stable/notes/randomness.html
#         torch.manual_seed(seed)
#         np.random.seed(seed)
#         if self.cuda:
#             torch.backends.cudnn.deterministic = True
#             torch.backends.cudnn.benchmark = False
#
#         # construct model
#         if config["model::type"] == "CNN":
#             # calling MLP constructor
#             if self.verbosity > 0:
#                 print("=> creating model CNN")
#             model = CNN(
#                 channels_in=config["model::channels_in"],
#                 nlin=config["model::nlin"],
#                 dropout=config["model::dropout"],
#                 init_type=config["model::init_type"],
#             )
#         elif config["model::type"] == "CNN2":
#             # calling MLP constructor
#             if self.verbosity > 0:
#                 print("=> creating model CNN")
#             model = CNN2(
#                 channels_in=config["model::channels_in"],
#                 nlin=config["model::nlin"],
#                 dropout=config["model::dropout"],
#                 init_type=config["model::init_type"],
#             )
#         elif config["model::type"] == "CNN3":
#             # calling MLP constructor
#             if self.verbosity > 0:
#                 print("=> creating model CNN")
#             model = CNN3(
#                 channels_in=config["model::channels_in"],
#                 nlin=config["model::nlin"],
#                 dropout=config["model::dropout"],
#                 init_type=config["model::init_type"],
#             )
#         elif config["model::type"] == "Resnet18":
#             # calling MLP constructor
#             if self.verbosity > 0:
#                 print("=> creating Resnet18")
#             model = ResNet18(
#                 channels_in=config["model::channels_in"],
#                 out_dim=config["model::o_dim"],
#                 nlin=config["model::nlin"],
#                 dropout=config["model::dropout"],
#                 init_type=config["model::init_type"],
#             )
#         else:
#             raise NotImplementedError("error: model type unkown")
#
#         if self.cuda:
#             model = model.cuda()
#
#         self.model = model
