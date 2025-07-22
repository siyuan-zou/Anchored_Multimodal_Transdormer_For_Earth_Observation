import torch
import torch.nn as nn

class Fine(nn.Module):
    """
    Initialize Fine Tuning of OmniSat after pretraining
    Args:
        encoder (torch.nn.Module): initialized model
        path (str): path of checkpoint of model to load
        output_size (int): size of output returned by encoder
        inter_dim (list): list of hidden dims of mlp after encoder
        p_drop (float): dropout parameter of mlp after encoder
        name (str): name of the weights from checkpoint to use
        freeze (bool); if True, freeze encoder to perform linear probing
        n_class (int): output_size of mlp
        pooling_method (str): type of pooling of tokens after transformer
        modalities (list): list of modalities to use
        last_block (bool): if True freeze all encoder except last block of transformer
        proj_only (bool): if True, load only weights from projectors
    """
    def __init__(self, 
                 output_size: int = 256,
                 num_patches: int = 36,
                 inter_dim: list = [],
                 p_drop: float = 0.3,
                 n_class: int = 15,
                 pooling_method: str = 'token'
                ):
        super().__init__()

        self.size = output_size
        self.global_pool = pooling_method
        self.num_patches = num_patches

        # set n_class to 0 if we want headless model
        self.n_class = n_class
        if n_class:
            if len(inter_dim) > 0:
                layers = [nn.Linear(self.size, inter_dim[0])]
                layers.append(nn.BatchNorm1d(inter_dim[0]))
                layers.append(nn.Dropout(p = p_drop))
                layers.append(nn.ReLU())
                for i in range(len(inter_dim) - 1):
                    layers.append(nn.Linear(inter_dim[i], inter_dim[i + 1]))
                    layers.append(nn.BatchNorm1d(inter_dim[i + 1]))
                    layers.append(nn.Dropout(p = p_drop))
                    layers.append(nn.ReLU())
                layers.append(nn.Linear(inter_dim[-1], n_class))
            else:
                layers = [nn.Linear(self.size, n_class)]
            self.head = nn.Sequential(*layers)
        
    def forward(self,x):
        """
        Forward pass of the network. Perform pooling of tokens after transformer 
        according to global_pool argument.
        """
        # print("before: ", x.shape) # torch.Size([32, 921600])

        if x.dim() == 2:
            B, _ = x.shape
            x = x.view(B, self.num_patches, self.size)
        # print("after: ", x.shape)

        if self.global_pool:
            if self.global_pool == 'avg':
                x = x[:, :, :].mean(dim=1)
            elif self.global_pool == 'max':
                x ,_ = torch.max(x[:, :, :],1)
            elif self.global_pool == 'avg_f': # return 32, 36
                x = x[:, :, :].mean(dim=2)
            elif self.global_pool == 'max_f': # return 32, 36
                x ,_ = torch.max(x[:, :, :],2)
            else:
                x = x[:, 0, :]
        if self.n_class:
            x = self.head(x) 

        # print(x.shape) # 32,256
        return x


if __name__ == "__main__":
    _ = Fine()