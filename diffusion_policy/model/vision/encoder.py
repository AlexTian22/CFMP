import torch
import torch.nn as nn
from typing import Dict, Tuple, Type, List
import torchvision.models as models
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from torchvision.models import vit_b_16
import torch.nn.functional as F
import numpy as np
import abc
from torchvision.models.vision_transformer import vit_b_16, VisionTransformer
torch.hub.set_dir('/DATA/disk0/ts/.cache/torch/hub/')


def create_mlp(
        input_dim: int,
        output_dim: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        squash_output: bool = False,
) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :return:
    """

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules


class Module(torch.nn.Module):
    """
    Base class for networks. The only difference from torch.nn.Module is that it
    requires implementing @output_shape.
    """
    @abc.abstractmethod
    def output_shape(self, input_shape=None):
        """
        Function to compute output shape from inputs to this module. 

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend 
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        raise NotImplementedError
class ConvBase(Module):
    """
    Base class for ConvNets.
    """
    def __init__(self):
        super(ConvBase, self).__init__()

    # dirty hack - re-implement to pass the buck onto subclasses from ABC parent
    def output_shape(self, input_shape):
        """
        Function to compute output shape from inputs to this module. 

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend 
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        raise NotImplementedError

    def forward(self, inputs):
        x = self.nets(inputs)
        if list(self.output_shape(list(inputs.shape)[1:])) != list(x.shape)[1:]:
            raise ValueError('Size mismatch: expect size %s, but got size %s' % (
                str(self.output_shape(list(inputs.shape)[1:])), str(list(x.shape)[1:]))
            )
        return x
class SpatialSoftmax(ConvBase):
    """
    Spatial Softmax Layer.

    Based on Deep Spatial Autoencoders for Visuomotor Learning by Finn et al.
    https://rll.berkeley.edu/dsae/dsae.pdf
    """
    def __init__(
        self,
        input_shape,
        num_kp=None,
        temperature=1.,
        learnable_temperature=False,
        output_variance=False,
        noise_std=0.0,
    ):
        """
        Args:
            input_shape (list): shape of the input feature (C, H, W)
            num_kp (int): number of keypoints (None for not use spatialsoftmax)
            temperature (float): temperature term for the softmax.
            learnable_temperature (bool): whether to learn the temperature
            output_variance (bool): treat attention as a distribution, and compute second-order statistics to return
            noise_std (float): add random spatial noise to the predicted keypoints
        """
        super(SpatialSoftmax, self).__init__()
        assert len(input_shape) == 3
        self._in_c, self._in_h, self._in_w = input_shape # (C, H, W)

        if num_kp is not None:
            self.nets = torch.nn.Conv2d(self._in_c, num_kp, kernel_size=1)
            self._num_kp = num_kp
        else:
            self.nets = None
            self._num_kp = self._in_c
        self.learnable_temperature = learnable_temperature
        self.output_variance = output_variance
        self.noise_std = noise_std

        if self.learnable_temperature:
            # temperature will be learned
            temperature = torch.nn.Parameter(torch.ones(1) * temperature, requires_grad=True)
            self.register_parameter('temperature', temperature)
        else:
            # temperature held constant after initialization
            temperature = torch.nn.Parameter(torch.ones(1) * temperature, requires_grad=False)
            self.register_buffer('temperature', temperature)

        pos_x, pos_y = np.meshgrid(
                np.linspace(-1., 1., self._in_w),
                np.linspace(-1., 1., self._in_h)
                )
        pos_x = torch.from_numpy(pos_x.reshape(1, self._in_h * self._in_w)).float()
        pos_y = torch.from_numpy(pos_y.reshape(1, self._in_h * self._in_w)).float()
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)

        self.kps = None

    def __repr__(self):
        """Pretty print network."""
        header = format(str(self.__class__.__name__))
        return header + '(num_kp={}, temperature={}, noise={})'.format(
            self._num_kp, self.temperature.item(), self.noise_std)

    def output_shape(self, input_shape):
        """
        Function to compute output shape from inputs to this module. 

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend 
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        assert(len(input_shape) == 3)
        assert(input_shape[0] == self._in_c)
        return [self._num_kp, 2]

    def forward(self, feature):
        """
        Forward pass through spatial softmax layer. For each keypoint, a 2D spatial 
        probability distribution is created using a softmax, where the support is the 
        pixel locations. This distribution is used to compute the expected value of 
        the pixel location, which becomes a keypoint of dimension 2. K such keypoints
        are created.

        Returns:
            out (torch.Tensor or tuple): mean keypoints of shape [B, K, 2], and possibly
                keypoint variance of shape [B, K, 2, 2] corresponding to the covariance
                under the 2D spatial softmax distribution
        """
        assert(feature.shape[1] == self._in_c)
        assert(feature.shape[2] == self._in_h)
        assert(feature.shape[3] == self._in_w)
        if self.nets is not None:
            feature = self.nets(feature)

        # [B, K, H, W] -> [B * K, H * W] where K is number of keypoints
        feature = feature.reshape(-1, self._in_h * self._in_w)
        # 2d softmax normalization
        attention = F.softmax(feature / self.temperature, dim=-1)
        # [1, H * W] x [B * K, H * W] -> [B * K, 1] for spatial coordinate mean in x and y dimensions
        expected_x = torch.sum(self.pos_x * attention, dim=1, keepdim=True)
        expected_y = torch.sum(self.pos_y * attention, dim=1, keepdim=True)
        # stack to [B * K, 2]
        expected_xy = torch.cat([expected_x, expected_y], 1)
        # reshape to [B, K, 2]
        feature_keypoints = expected_xy.view(-1, self._num_kp, 2)

        if self.training:
            noise = torch.randn_like(feature_keypoints) * self.noise_std
            feature_keypoints += noise

        if self.output_variance:
            # treat attention as a distribution, and compute second-order statistics to return
            expected_xx = torch.sum(self.pos_x * self.pos_x * attention, dim=1, keepdim=True)
            expected_yy = torch.sum(self.pos_y * self.pos_y * attention, dim=1, keepdim=True)
            expected_xy = torch.sum(self.pos_x * self.pos_y * attention, dim=1, keepdim=True)
            var_x = expected_xx - expected_x * expected_x
            var_y = expected_yy - expected_y * expected_y
            var_xy = expected_xy - expected_x * expected_y
            # stack to [B * K, 4] and then reshape to [B, K, 2, 2] where last 2 dims are covariance matrix
            feature_covar = torch.cat([var_x, var_xy, var_xy, var_y], 1).reshape(-1, self._num_kp, 2, 2)
            feature_keypoints = (feature_keypoints, feature_covar)

        if isinstance(feature_keypoints, tuple):
            self.kps = (feature_keypoints[0].detach(), feature_keypoints[1].detach())
        else:
            self.kps = feature_keypoints.detach()
        return feature_keypoints
    
    
class DinoEncoder(nn.Module):
    def __init__(self, 
                 observation_space: Dict, 
                 image_out_channel: int = 64,
                 state_out_channel: int = 9,
                 image_mlp_size: Tuple[int, ...] = (256, 128), 
                 state_mlp_size: Tuple[int, ...] = (16, 12), 
                 state_mlp_activation_fn: Type[nn.Module] = nn.ReLU,
                 dino_model_name: str = 'vit_base_patch16_224',
                 **kwargs):
        """
        Args:
            observation_space (Dict): Dictionary containing the shapes of the input observations.
            out_channel (int): The output dimension of the encoder.
            state_mlp_size (Tuple[int, ...]): The architecture of the MLP for processing the agent state.
            state_mlp_activation_fn (Type[nn.Module]): The activation function to use in the MLP.
            dino_model_name (str): The name of the DINO v2 model to use.
        """
        super().__init__()
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(112),  # Crop and resize with randomness
            transforms.RandomHorizontalFlip(),   # Randomly flip images horizontally
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Color jittering
            transforms.ToTensor(),  # Convert PIL Image or numpy.ndarray to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize the tensors
        ])
        self.dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        
        # Set the model to evaluation mode
        # self.dino_model.eval()  
        # for param in self.dino_model.parameters():
        #     param.requires_grad = False
        
        dino_output_dim = 768
        
        self.dino_mlp = nn.Sequential(*create_mlp(dino_output_dim, image_out_channel, [128], activation_fn=state_mlp_activation_fn))
        self.state_mlp = nn.Sequential(*create_mlp(observation_space['agent_pos'][0], state_out_channel, [12], activation_fn=state_mlp_activation_fn))
        
        # Calculate the total output dimension
        self.n_output_channels = image_out_channel + state_out_channel  # DINO features (64D) + state features (6D)
        
    def forward(self, observations: Dict) -> torch.Tensor:
        image = observations['image']
        batch_size = image.shape[0]
        transformed_images = []
        for i in range(batch_size):
            img = image[i]  # Get the i-th image in the batch
            if isinstance(img, torch.Tensor):
                img = to_pil_image(img.cpu().detach())  # Convert to PIL Image
            transformed_img = self.transform(img)  # Apply transformations
            transformed_images.append(transformed_img)
        
        transformed_images = torch.stack(transformed_images).to(image.device)
        # with torch.no_grad():  # No need to compute gradients for DINO features
        dino_features = self.dino_model(transformed_images)
        
        dino_features = self.dino_mlp(dino_features)
        
        agent_pos = observations['agent_pos']
        state_features = self.state_mlp(agent_pos)
        
        final_feat = torch.cat([dino_features, state_features], dim=-1)
        return final_feat
    
    def output_shape(self):
        return self.n_output_channels

class DinoEncoderSpatialSoftmax(nn.Module):
    def __init__(self, 
                 observation_space: Dict, 
                 image_out_channel: int = 64,
                 state_out_channel: int = 9,
                 image_mlp_size: Tuple[int, ...] = (256, 128), 
                 state_mlp_size: Tuple[int, ...] = (16, 12), 
                 state_mlp_activation_fn: Type[nn.Module] = nn.ReLU,
                 dino_model_name: str = 'vit_base_patch16_224',
                 **kwargs):
        super().__init__()
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(112),  # Crop and resize with randomness
            transforms.RandomHorizontalFlip(),   # Randomly flip images horizontally
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Color jittering
            transforms.ToTensor(),  # Convert PIL Image or numpy.ndarray to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize the tensors
        ])
        # Initialize the DINO v2 model
        self.dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')

        blocks_list = [block for block in self.dino_model.blocks]

        self.dino_model_backbone = nn.Sequential(
            self.dino_model.patch_embed,  # 提取 patch embedding
            *blocks_list,                 # 提取所有 Transformer blocks
        )
        # self.dino_model = nn.Sequential(*list(self.dino_model.children())[:-2]) 报错
        
        # Freeze the DINO model parameters
        # self.dino_model_backbone.eval()
        # for param in self.dino_model_backbone.parameters():
        #     param.requires_grad = False
        
        with torch.no_grad():
            test_input = torch.zeros(1, 3, 112, 112)  
            test_output = self.dino_model_backbone(test_input)
            batch_size = test_output.shape[0]
            test_output_reshaped = test_output.permute(0, 2, 1).reshape(batch_size, 768, 8, 8)
            dino_output_shape = test_output_reshaped.shape[1:]
            # print(dino_output_shape) # [768,8,8]         
        # SpatialSoftmax层（num_kp=32）
        self.spatial_softmax = SpatialSoftmax(
            input_shape=dino_output_shape,
            num_kp=32,
            temperature=1.0,
            noise_std=0.0
        )
        
        # Flatten + Linear层
        self.dino_mlp = nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Sequential(*create_mlp(64, image_out_channel, [128], activation_fn=state_mlp_activation_fn))  
        )
        self.state_mlp = nn.Sequential(*create_mlp(observation_space['agent_pos'][0], state_out_channel, [12], activation_fn=state_mlp_activation_fn))
        # Calculate the total output dimension
        self.n_output_channels = image_out_channel + state_out_channel  # DINO features (64D) + state features (6D)
        
        
    def forward(self, observations: Dict) -> torch.Tensor:
        """
        Args:
            observations (Dict): Dictionary containing the input observations.
                                Expected keys: 'image' and 'agent_pos'.
        Returns:
            torch.Tensor: The concatenated feature vector.
        """
        image = observations['image']

        batch_size = image.shape[0]
        transformed_images = []
        
        for i in range(batch_size):
            img = image[i]  # Get the i-th image in the batch
            if isinstance(img, torch.Tensor):
                img = to_pil_image(img.cpu().detach())  # Convert to PIL Image
            transformed_img = self.transform(img)  # Apply transformations
            transformed_images.append(transformed_img)
        
        # Stack transformed images into a batch
        transformed_images = torch.stack(transformed_images).to(image.device)

        dino_features = self.dino_model_backbone(transformed_images)
        dino_features_reshape = dino_features.permute(0, 2, 1).reshape(batch_size, 768, 8, 8)
        features = self.spatial_softmax(dino_features_reshape) # [B, 32*2=64]（假设SpatialSoftmax输出64维）
        image_features = self.dino_mlp(features)
        
        # Extract agent state features using the MLP
        agent_pos = observations['agent_pos']
        state_features = self.state_mlp(agent_pos)
        
        # Concatenate the features
        final_feat = torch.cat([image_features, state_features], dim=-1)
        
        return final_feat
    
    def output_shape(self):
        return self.n_output_channels
    
    
class ViTEncoder(nn.Module):
    def __init__(self, 
                 observation_space: Dict, 
                 image_out_channel: int = 64,
                 state_out_channel: int = 9,
                 image_mlp_size: Tuple[int, ...] = (256, 128), 
                 state_mlp_size: Tuple[int, ...] = (16, 12), 
                 state_mlp_activation_fn: Type[nn.Module] = nn.ReLU,
                 vit_model_name: str = 'vit_b_16',
                 **kwargs):
        """
        Args:
            observation_space (Dict): Dictionary containing the shapes of the input observations.
            image_out_channel (int): The output dimension of the image encoder.
            state_out_channel (int): The output dimension of the state encoder.
            state_mlp_size (Tuple[int, ...]): The architecture of the MLP for processing the agent state.
            state_mlp_activation_fn (Type[nn.Module]): The activation function to use in the MLP.
            vit_model_name (str): The name of the ViT model to use.
        """
        super().__init__()
        
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224),  # Crop and resize with randomness
            transforms.RandomHorizontalFlip(),   # Randomly flip images horizontally
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Color jittering
            transforms.ToTensor(),  # Convert PIL Image or numpy.ndarray to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize the tensors
        ])
        # Initialize the ViT model
        self.vit_model = models.vit_b_16(pretrained=True)  # Load a pretrained ViT model
        # print(self.vit_model )
        # self.vit_model.eval()  # Set the model to evaluation mode
        
        # Freeze the ViT model parameters
        # for param in self.vit_model.parameters():
        #     param.requires_grad = False
        
        # Get the output dimension of the ViT model
        vit_output_dim = self.vit_model.heads.head.in_features  # ViT 的输出维度
        self.vit_model.heads = nn.Identity()
        # MLP to map ViT features to image_out_channel dimensions
        self.vit_mlp = nn.Sequential(*create_mlp(vit_output_dim, image_out_channel, [128], activation_fn=state_mlp_activation_fn))
        
        # MLP to map agent state (9D) to state_out_channel dimensions
        self.state_mlp = nn.Sequential(*create_mlp(observation_space['agent_pos'][0], state_out_channel, [12], activation_fn=state_mlp_activation_fn))
        
        # Calculate the total output dimension
        self.n_output_channels = image_out_channel + state_out_channel  # ViT features + state features
        
    def forward(self, observations: Dict) -> torch.Tensor:
        """
        Args:
            observations (Dict): Dictionary containing the input observations.
                                Expected keys: 'image' and 'agent_pos'.
        Returns:
            torch.Tensor: The concatenated feature vector.
        """
                # Extract image features using ViT
        image = observations['image']
        
        # Handle batched input
        batch_size = image.shape[0]
        transformed_images = []
        
        for i in range(batch_size):
            img = image[i]  # Get the i-th image in the batch
            if isinstance(img, torch.Tensor):
                img = to_pil_image(img.cpu().detach())  # Convert to PIL Image
            transformed_img = self.transform(img)  # Apply transformations
            transformed_images.append(transformed_img)
        
        # Stack transformed images into a batch
        transformed_images = torch.stack(transformed_images).to(image.device)
        
        # with torch.no_grad():  # No need to compute gradients for ViT features
        vit_features = self.vit_model(transformed_images)

        
        # Map ViT features to image_out_channel dimensions
        vit_features = self.vit_mlp(vit_features)
        
        # Extract agent state features using the MLP
        agent_pos = observations['agent_pos']
        state_features = self.state_mlp(agent_pos)
        
        # Concatenate the features
        final_feat = torch.cat([vit_features, state_features], dim=-1)
        
        return final_feat
    
    def output_shape(self):
        return self.n_output_channels


class ViTEncoderSpatialSoftmax(nn.Module):
    def __init__(self, 
                 observation_space: Dict, 
                 image_out_channel: int = 64,
                 state_out_channel: int = 9,
                 image_mlp_size: Tuple[int, ...] = (256, 128), 
                 state_mlp_size: Tuple[int, ...] = (16, 12), 
                 state_mlp_activation_fn: Type[nn.Module] = nn.ReLU,
                 vit_model_name: str = 'vit_b_16',
                 **kwargs):
        super().__init__()
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224),  # Crop and resize with randomness
            transforms.RandomHorizontalFlip(),   # Randomly flip images horizontally
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Color jittering
            transforms.ToTensor(),  # Convert PIL Image or numpy.ndarray to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize the tensors
        ])
        # 获取预训练 ViT 模型
        self.vit_model: VisionTransformer = models.vit_b_16(pretrained=True)
        # self.vit_model.eval()
        # for param in self.vit_model.parameters():
        #     param.requires_grad = False
        self.spatial_softmax = SpatialSoftmax(input_shape=(768, 14, 14), num_kp=32)

        # image MLP
        self.image_mlp = nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Sequential(*create_mlp(64, image_out_channel, [128], activation_fn=state_mlp_activation_fn))   
        )

        # state MLP
        self.state_mlp = nn.Sequential(*create_mlp(observation_space['agent_pos'][0], state_out_channel,[12], activation_fn=state_mlp_activation_fn))

        self.n_output_channels = image_out_channel + state_out_channel

    def forward(self, observations: Dict) -> torch.Tensor:
        image = observations['image']
        batch_size = image.shape[0]
        transformed_images = []
        
        for i in range(batch_size):
            img = image[i]  # Get the i-th image in the batch
            if isinstance(img, torch.Tensor):
                img = to_pil_image(img.cpu().detach())  # Convert to PIL Image
            transformed_img = self.transform(img)  # Apply transformations
            transformed_images.append(transformed_img)
        
        # Stack transformed images into a batch
        transformed_images = torch.stack(transformed_images).to(image.device)
        # 获取 ViT patch token 特征图
        # with torch.no_grad():
        x = self.vit_model._process_input(transformed_images)  # patch embedding
        cls_token = self.vit_model.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.vit_model.encoder.pos_embedding
        x = self.vit_model.encoder.dropout(x)
        x = self.vit_model.encoder.layers(x)

        x_patch = x[:, 1:, :]  # 去掉 CLS token
        h = w = int(x_patch.shape[1] ** 0.5)
        x_spatial = x_patch.permute(0, 2, 1).reshape(-1, 768, h, w)

        # 接 SpatialSoftmax
        spatial_feat = self.spatial_softmax(x_spatial)  # 输出 [B, 64]（32个 keypoint）
        image_features = self.image_mlp(spatial_feat)

        # 状态特征
        agent_pos = observations['agent_pos']
        state_features = self.state_mlp(agent_pos)

        final_feat = torch.cat([image_features, state_features], dim=-1)
        return final_feat

    def output_shape(self):
        return self.n_output_channels

class ResNet18Encoder(nn.Module):
    def __init__(self, 
                 observation_space: Dict, 
                 image_out_channel: int = 64,
                 state_out_channel: int = 9,
                 image_mlp_size: Tuple[int, ...] = (256, 128), 
                 state_mlp_size: Tuple[int, ...] = (16, 12), 
                 state_mlp_activation_fn: Type[nn.Module] = nn.ReLU,
                 **kwargs):
        """
        Args:
            observation_space (Dict): Dictionary containing the shapes of the input observations.
            image_out_channel (int): The output dimension of the image encoder.
            state_out_channel (int): The output dimension of the state encoder.
            state_mlp_size (Tuple[int, ...]): The architecture of the MLP for processing the agent state.
            state_mlp_activation_fn (Type[nn.Module]): The activation function to use in the MLP.
        """
        super().__init__()
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(112),  # Crop and resize with randomness
            transforms.RandomHorizontalFlip(),   # Randomly flip images horizontally
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Color jittering
            transforms.ToTensor(),  # Convert PIL Image or numpy.ndarray to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize the tensors
        ])        
        # Initialize the ResNet model
        self.resnet_model = models.resnet18(pretrained=True)
        self.resnet_model.fc = nn.Identity()  # Replace the fully connected layer with identity
        
        # self.resnet_model.eval()  # Set the model to evaluation mode
        
        # Freeze the ResNet model parameters
        # for param in self.resnet_model.parameters():
        #     param.requires_grad = False
        
        # Get the output dimension of the ResNet model before the fc layer
        resnet_output_dim = 512  # For ResNet-18
        
        # MLP to map ResNet features to image_out_channel dimensions
        self.resnet_mlp = nn.Sequential(*create_mlp(resnet_output_dim, image_out_channel, [128], activation_fn=state_mlp_activation_fn))
        self.state_mlp = nn.Sequential(*create_mlp(observation_space['agent_pos'][0], state_out_channel, [12], activation_fn=state_mlp_activation_fn))
        
        # Calculate the total output dimension
        self.n_output_channels = image_out_channel + state_out_channel  # ResNet features + state features
    def forward(self, observations: Dict) -> torch.Tensor:
        image = observations['image']
        batch_size = image.shape[0]
        transformed_images = []
        
        for i in range(batch_size):
            img = image[i]  # Get the i-th image in the batch
            if isinstance(img, torch.Tensor):
                img = to_pil_image(img.cpu().detach())  # Convert to PIL Image
            transformed_img = self.transform(img)  # Apply transformations
            transformed_images.append(transformed_img)
        
        # Stack transformed images into a batch
        transformed_images = torch.stack(transformed_images).to(image.device)      
        features = self.resnet_model(transformed_images)     
        # print("resnet_features",features.shape)
        image_features = self.resnet_mlp(features)
        
        # Extract agent state features using the MLP
        agent_pos = observations['agent_pos']   
        state_features = self.state_mlp(agent_pos)     
        # Concatenate the features
        final_feat = torch.cat([image_features, state_features], dim=-1)
        
        return final_feat
    
    def output_shape(self):
        return self.n_output_channels
    
class ResNet18SpatialSoftmax(nn.Module):
    def __init__(self, 
                 observation_space: Dict, 
                 image_out_channel: int = 64,
                 state_out_channel: int = 9,
                 image_mlp_size: Tuple[int, ...] = (256, 128), 
                 state_mlp_size: Tuple[int, ...] = (16, 12), 
                 state_mlp_activation_fn: Type[nn.Module] = nn.ReLU,
                 **kwargs):
        """
        Args:
            observation_space (Dict): Dictionary containing the shapes of the input observations.
            image_out_channel (int): The output dimension of the image encoder.
            state_out_channel (int): The output dimension of the state encoder.
            state_mlp_size (Tuple[int, ...]): The architecture of the MLP for processing the agent state.
            state_mlp_activation_fn (Type[nn.Module]): The activation function to use in the MLP.
        """
        super().__init__()
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(112),  # Crop and resize with randomness
            transforms.RandomHorizontalFlip(),   # Randomly flip images horizontally
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Color jittering
            transforms.ToTensor(),  # Convert PIL Image or numpy.ndarray to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize the tensors
        ])        
        # 加载 ResNet18 模型
        self.resnet = models.resnet18(pretrained=True)
        
        # 移除最后的 AdaptiveAvgPool2d 和 fc 层
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])
        input_image_shape = observation_space['image']
        with torch.no_grad():
            test_input = torch.zeros(1, *input_image_shape)  # 创建一个测试输入张量
            test_output = self.resnet(test_input)
            resnet_output_shape = test_output.shape[1:]  # 获取除去批次维度的输出形状
            # print(resnet_output_shape)

        # SpatialSoftmax层（num_kp=32）
        self.spatial_softmax = SpatialSoftmax(
            input_shape=resnet_output_shape,
            num_kp=32,
            temperature=1.0,
            noise_std=0.0
        )
        
        # Flatten + Linear层
        self.resnet_mlp = nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Sequential(*create_mlp(64, image_out_channel, [128], activation_fn=state_mlp_activation_fn)) 
        )
        self.state_mlp = nn.Sequential(*create_mlp(observation_space['agent_pos'][0], state_out_channel, [12], activation_fn=state_mlp_activation_fn))                
        # Calculate the total output dimension
        self.n_output_channels = image_out_channel + state_out_channel  # ResNet features + state features
    
    def forward(self, observations: Dict) -> torch.Tensor:
        image = observations['image']
        batch_size = image.shape[0]
        transformed_images = []
        
        for i in range(batch_size):
            img = image[i]  # Get the i-th image in the batch
            if isinstance(img, torch.Tensor):
                img = to_pil_image(img.cpu().detach())  # Convert to PIL Image
            transformed_img = self.transform(img)  # Apply transformations
            transformed_images.append(transformed_img)
        
        # Stack transformed images into a batch
        transformed_images = torch.stack(transformed_images).to(image.device)     
        features = self.resnet(transformed_images)
        # print("resnet_features",features.shape)
        features = self.spatial_softmax(features) # [B, 32*2=64]（假设SpatialSoftmax输出64维）
        # print("resnet_spatial_features",features.shape)
        image_features = self.resnet_mlp(features)
        
        # Extract agent state features using the MLP
        agent_pos = observations['agent_pos']
        state_features = self.state_mlp(agent_pos)        
        # Concatenate the features
        final_feat = torch.cat([image_features, state_features], dim=-1)
        
        return final_feat
    
    def output_shape(self):
        return self.n_output_channels
    
class ResNet34Encoder(nn.Module):
    def __init__(self, 
                 observation_space: Dict, 
                 image_out_channel: int = 64,
                 state_out_channel: int = 9,
                 image_mlp_size: Tuple[int, ...] = (256, 128), 
                 state_mlp_size: Tuple[int, ...] = (16, 12), 
                 state_mlp_activation_fn: Type[nn.Module] = nn.ReLU,
                 **kwargs):
        """
        Args:
            observation_space (Dict): Dictionary containing the shapes of the input observations.
            image_out_channel (int): The output dimension of the image encoder.
            state_out_channel (int): The output dimension of the state encoder.
            state_mlp_size (Tuple[int, ...]): The architecture of the MLP for processing the agent state.
            state_mlp_activation_fn (Type[nn.Module]): The activation function to use in the MLP.
        """
        super().__init__()
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(112),  # Crop and resize with randomness
            transforms.RandomHorizontalFlip(),   # Randomly flip images horizontally
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Color jittering
            transforms.ToTensor(),  # Convert PIL Image or numpy.ndarray to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize the tensors
        ])        
        # Initialize the ResNet model
        self.resnet_model = models.resnet34(pretrained=True)
        self.resnet_model.fc = nn.Identity()  # Replace the fully connected layer with identity
        
        # self.resnet_model.eval()  # Set the model to evaluation mode
        
        # Freeze the ResNet model parameters
        # for param in self.resnet_model.parameters():
        #     param.requires_grad = False
        
        # Get the output dimension of the ResNet model before the fc layer
        resnet_output_dim = 512  # For ResNet-18
        
        # MLP to map ResNet features to image_out_channel dimensions
        self.resnet_mlp = nn.Sequential(*create_mlp(resnet_output_dim, image_out_channel, [128], activation_fn=state_mlp_activation_fn))
        self.state_mlp = nn.Sequential(*create_mlp(observation_space['agent_pos'][0], state_out_channel, [12], activation_fn=state_mlp_activation_fn))
        
        # Calculate the total output dimension
        self.n_output_channels = image_out_channel + state_out_channel  # ResNet features + state features
    def forward(self, observations: Dict) -> torch.Tensor:
        image = observations['image']
        batch_size = image.shape[0]
        transformed_images = []
        
        for i in range(batch_size):
            img = image[i]  # Get the i-th image in the batch
            if isinstance(img, torch.Tensor):
                img = to_pil_image(img.cpu().detach())  # Convert to PIL Image
            transformed_img = self.transform(img)  # Apply transformations
            transformed_images.append(transformed_img)
        
        # Stack transformed images into a batch
        transformed_images = torch.stack(transformed_images).to(image.device)      
        features = self.resnet_model(transformed_images)     
        # print("resnet_features",features.shape)
        image_features = self.resnet_mlp(features)
        
        # Extract agent state features using the MLP
        agent_pos = observations['agent_pos']   
        state_features = self.state_mlp(agent_pos)     
        # Concatenate the features
        final_feat = torch.cat([image_features, state_features], dim=-1)
        
        return final_feat
    
    def output_shape(self):
        return self.n_output_channels
    
class ResNet34SpatialSoftmax(nn.Module):
    def __init__(self, 
                 observation_space: Dict, 
                 image_out_channel: int = 64,
                 state_out_channel: int = 9,
                 image_mlp_size: Tuple[int, ...] = (256, 128), 
                 state_mlp_size: Tuple[int, ...] = (16, 12), 
                 state_mlp_activation_fn: Type[nn.Module] = nn.ReLU,
                 **kwargs):
        """
        Args:
            observation_space (Dict): Dictionary containing the shapes of the input observations.
            image_out_channel (int): The output dimension of the image encoder.
            state_out_channel (int): The output dimension of the state encoder.
            state_mlp_size (Tuple[int, ...]): The architecture of the MLP for processing the agent state.
            state_mlp_activation_fn (Type[nn.Module]): The activation function to use in the MLP.
        """
        super().__init__()
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(112),  # Crop and resize with randomness
            transforms.RandomHorizontalFlip(),   # Randomly flip images horizontally
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Color jittering
            transforms.ToTensor(),  # Convert PIL Image or numpy.ndarray to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize the tensors
        ])        
        
        self.resnet = models.resnet34(pretrained=True)
        
        # 移除最后的 AdaptiveAvgPool2d 和 fc 层
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])
        input_image_shape = observation_space['image']
        with torch.no_grad():
            test_input = torch.zeros(1, *input_image_shape)  # 创建一个测试输入张量
            test_output = self.resnet(test_input)
            resnet_output_shape = test_output.shape[1:]  # 获取除去批次维度的输出形状
            # print(resnet_output_shape)

        # SpatialSoftmax层（num_kp=32）
        self.spatial_softmax = SpatialSoftmax(
            input_shape=resnet_output_shape,
            num_kp=32,
            temperature=1.0,
            noise_std=0.0
        )
        
        # Flatten + Linear层
        self.resnet_mlp = nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Sequential(*create_mlp(64, image_out_channel, [128], activation_fn=state_mlp_activation_fn)) 
        )
        self.state_mlp = nn.Sequential(*create_mlp(observation_space['agent_pos'][0], state_out_channel, [12], activation_fn=state_mlp_activation_fn))                
        # Calculate the total output dimension
        self.n_output_channels = image_out_channel + state_out_channel  # ResNet features + state features
    
    def forward(self, observations: Dict) -> torch.Tensor:
        image = observations['image']
        batch_size = image.shape[0]
        transformed_images = []
        
        for i in range(batch_size):
            img = image[i]  # Get the i-th image in the batch
            if isinstance(img, torch.Tensor):
                img = to_pil_image(img.cpu().detach())  # Convert to PIL Image
            transformed_img = self.transform(img)  # Apply transformations
            transformed_images.append(transformed_img)
        
        # Stack transformed images into a batch
        transformed_images = torch.stack(transformed_images).to(image.device)     
        features = self.resnet(transformed_images)
        # print("resnet_features",features.shape)
        features = self.spatial_softmax(features) # [B, 32*2=64]（假设SpatialSoftmax输出64维）
        # print("resnet_spatial_features",features.shape)
        image_features = self.resnet_mlp(features)
        
        # Extract agent state features using the MLP
        agent_pos = observations['agent_pos']
        state_features = self.state_mlp(agent_pos)        
        # Concatenate the features
        final_feat = torch.cat([image_features, state_features], dim=-1)
        
        return final_feat
    
    def output_shape(self):
        return self.n_output_channels
    
    
class ResNet50Encoder(nn.Module):
    def __init__(self, 
                 observation_space: Dict, 
                 image_out_channel: int = 64,
                 state_out_channel: int = 9,
                 image_mlp_size: Tuple[int, ...] = (256, 128), 
                 state_mlp_size: Tuple[int, ...] = (16, 12), 
                 state_mlp_activation_fn: Type[nn.Module] = nn.ReLU,
                 **kwargs):
        """
        Args:
            observation_space (Dict): Dictionary containing the shapes of the input observations.
            image_out_channel (int): The output dimension of the image encoder.
            state_out_channel (int): The output dimension of the state encoder.
            state_mlp_size (Tuple[int, ...]): The architecture of the MLP for processing the agent state.
            state_mlp_activation_fn (Type[nn.Module]): The activation function to use in the MLP.
        """
        super().__init__()
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(112),  # Crop and resize with randomness
            transforms.RandomHorizontalFlip(),   # Randomly flip images horizontally
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Color jittering
            transforms.ToTensor(),  # Convert PIL Image or numpy.ndarray to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize the tensors
        ])        
        # Initialize the ResNet model
        self.resnet_model = models.resnet50(pretrained=True)
        self.resnet_model.fc = nn.Identity()  # Replace the fully connected layer with identity
        
        # self.resnet_model.eval()  # Set the model to evaluation mode
        
        # Freeze the ResNet model parameters
        # for param in self.resnet_model.parameters():
        #     param.requires_grad = False
        
        # Get the output dimension of the ResNet model before the fc layer
        resnet_output_dim = 2048  # For ResNet-50
        
        # MLP to map ResNet features to image_out_channel dimensions
        self.resnet_mlp = nn.Sequential(*create_mlp(resnet_output_dim, image_out_channel, [128], activation_fn=state_mlp_activation_fn))
        self.state_mlp = nn.Sequential(*create_mlp(observation_space['agent_pos'][0], state_out_channel, [12], activation_fn=state_mlp_activation_fn))
        
        # Calculate the total output dimension
        self.n_output_channels = image_out_channel + state_out_channel  # ResNet features + state features
    def forward(self, observations: Dict) -> torch.Tensor:
        image = observations['image']
        batch_size = image.shape[0]
        transformed_images = []
        
        for i in range(batch_size):
            img = image[i]  # Get the i-th image in the batch
            if isinstance(img, torch.Tensor):
                img = to_pil_image(img.cpu().detach())  # Convert to PIL Image
            transformed_img = self.transform(img)  # Apply transformations
            transformed_images.append(transformed_img)
        
        # Stack transformed images into a batch
        transformed_images = torch.stack(transformed_images).to(image.device)      
        features = self.resnet_model(transformed_images)     
        # print("resnet_features",features.shape)
        image_features = self.resnet_mlp(features)
        
        # Extract agent state features using the MLP
        agent_pos = observations['agent_pos']   
        state_features = self.state_mlp(agent_pos)     
        # Concatenate the features
        final_feat = torch.cat([image_features, state_features], dim=-1)
        
        return final_feat
    
    def output_shape(self):
        return self.n_output_channels



class ResNet50SpatialSoftmax(nn.Module):
    def __init__(self, 
                 observation_space: Dict, 
                 image_out_channel: int = 64,
                 state_out_channel: int = 9,
                 image_mlp_size: Tuple[int, ...] = (256, 128), 
                 state_mlp_size: Tuple[int, ...] = (16, 12), 
                 state_mlp_activation_fn: Type[nn.Module] = nn.ReLU,
                 **kwargs):
        super().__init__()
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(112),  # Crop and resize with randomness
            transforms.RandomHorizontalFlip(),   # Randomly flip images horizontally
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Color jittering
            transforms.ToTensor(),  # Convert PIL Image or numpy.ndarray to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize the tensors
        ])        
        
        self.resnet = models.resnet50(pretrained=True)
        # self.resnet = models.resnet50(pretrained=False)
        
        # 移除最后的 AdaptiveAvgPool2d 和 fc 层
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])
        input_image_shape = observation_space['image']
        with torch.no_grad():
            test_input = torch.zeros(1, *input_image_shape)  # 创建一个测试输入张量
            test_output = self.resnet(test_input)
            resnet_output_shape = test_output.shape[1:]  # 获取除去批次维度的输出形状
            # print(resnet_output_shape)

        # SpatialSoftmax层（num_kp=32）
        self.spatial_softmax = SpatialSoftmax(
            input_shape=resnet_output_shape,
            num_kp=32,
            temperature=1.0,
            noise_std=0.0
        )
        
        # Flatten + Linear层
        self.resnet_mlp = nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Sequential(*create_mlp(64, image_out_channel, [128], activation_fn=state_mlp_activation_fn)) 
        )
        self.state_mlp = nn.Sequential(*create_mlp(observation_space['agent_pos'][0], state_out_channel, [12], activation_fn=state_mlp_activation_fn))                
        # Calculate the total output dimension
        self.n_output_channels = image_out_channel + state_out_channel  # ResNet features + state features
    
    def forward(self, observations: Dict) -> torch.Tensor:
        image = observations['image']
        batch_size = image.shape[0]
        transformed_images = []
        
        for i in range(batch_size):
            img = image[i]  # Get the i-th image in the batch
            if isinstance(img, torch.Tensor):
                img = to_pil_image(img.cpu().detach())  # Convert to PIL Image
            transformed_img = self.transform(img)  # Apply transformations
            transformed_images.append(transformed_img)
        
        # Stack transformed images into a batch
        transformed_images = torch.stack(transformed_images).to(image.device)     
        features = self.resnet(transformed_images)
        # print("resnet_features",features.shape)
        features = self.spatial_softmax(features) # [B, 32*2=64]（假设SpatialSoftmax输出64维）
        # print("resnet_spatial_features",features.shape)
        image_features = self.resnet_mlp(features)
        
        # Extract agent state features using the MLP
        agent_pos = observations['agent_pos']
        state_features = self.state_mlp(agent_pos)        
        # Concatenate the features
        final_feat = torch.cat([image_features, state_features], dim=-1)
        
        return final_feat
    
    def output_shape(self):
        return self.n_output_channels    
if __name__ == '__main__':
    import torch

    observation_space = {
        'image': (3, 127, 127),  
        'agent_pos': (9,)
    }
    
    image_input = torch.randn((1, *observation_space['image']))  
    agent_pos_input = torch.randn((1, *observation_space['agent_pos']))  
    

    observations = {'image': image_input, 'agent_pos': agent_pos_input}
    
    
    
    # dino_encoder = DinoEncoder(observation_space=observation_space)
    vit_encoder = ViTEncoder(observation_space=observation_space)
    # resnet18_encoder = ResNet18Encoder(observation_space=observation_space)
    # print(vit_encoder)
    
    # print(dino_encoder)

    # output_features =dino_encoder(observations)
    output_features =vit_encoder(observations)
    # output_features = resnet18_encoder(observations)
    
    print("Output feature shape:", output_features.shape)