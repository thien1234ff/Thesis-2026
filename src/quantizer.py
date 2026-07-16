import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn import Module, Parameter
from collections import OrderedDict

# Import block classes for type-checking in quantize_model
from src.models import IBasicBlock, Depth_Wise

# ============================================================
# Core Quantization Utilities
# ============================================================

def clamp(input_tensor, min_val, max_val, inplace=False):
    """Clamp tensor input_tensor to (min_val, max_val)."""
    if inplace:
        input_tensor.clamp_(min_val, max_val)
        return input_tensor
    return torch.clamp(input_tensor, min_val, max_val)


def linear_quantize(input_tensor, scale, zero_point, inplace=False):
    """Quantize single-precision input tensor with scaling factor and zero-point."""
    if len(input_tensor.shape) == 4:
        scale = scale.view(-1, 1, 1, 1)
        zero_point = zero_point.view(-1, 1, 1, 1)
    elif len(input_tensor.shape) == 2:
        scale = scale.view(-1, 1)
        zero_point = zero_point.view(-1, 1)

    if inplace:
        input_tensor.mul_(scale).sub_(zero_point).round_()
        return input_tensor
    return torch.round(scale * input_tensor - zero_point)


def linear_dequantize(input_tensor, scale, zero_point, inplace=False):
    """Map integer tensor back to float representation with scale and zero-point."""
    if len(input_tensor.shape) == 4:
        scale = scale.view(-1, 1, 1, 1)
        zero_point = zero_point.view(-1, 1, 1, 1)
    elif len(input_tensor.shape) == 2:
        scale = scale.view(-1, 1)
        zero_point = zero_point.view(-1, 1)

    if inplace:
        input_tensor.add_(zero_point).div_(scale)
        return input_tensor
    return (input_tensor + zero_point) / scale


def asymmetric_linear_quantization_params(num_bits, saturation_min, saturation_max,
                                          integral_zero_point=True, signed=True):
    """Compute scale and zero-point parameters with the given range and bits constraint."""
    n = 2**num_bits - 1
    scale = n / torch.clamp((saturation_max - saturation_min), min=1e-8)
    zero_point = scale * saturation_min

    if integral_zero_point:
        if isinstance(zero_point, torch.Tensor):
            zero_point = zero_point.round()
        else:
            zero_point = float(round(zero_point))
    if signed:
        zero_point += 2**(num_bits - 1)
    return scale, zero_point


class AsymmetricQuantFunction(Function):
    """Autograd Function for asymmetric quantization.
    Uses Straight-Through Estimator (STE) for backpropagation.
    """
    @staticmethod
    def forward(ctx, x, k, x_min=None, x_max=None):
        if x_min is None or x_max is None:
            x_min, x_max = x.min(), x.max()
        scale, zero_point = asymmetric_linear_quantization_params(k, x_min, x_max)
        new_quant_x = linear_quantize(x, scale, zero_point, inplace=False)
        n = 2**(k - 1)
        new_quant_x = torch.clamp(new_quant_x, -n, n - 1)
        quant_x = linear_dequantize(new_quant_x, scale, zero_point, inplace=False)
        return quant_x

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-Through Estimator (STE) returns incoming gradient directly
        return grad_output, None, None, None


# ============================================================
# Quantization Layer Wrappers
# ============================================================

class QuantAct(Module):
    """Class to quantize activation tensors dynamically or statically."""
    def __init__(self, activation_bit, full_precision_flag=False, running_stat=True, beta=0.9):
        super(QuantAct, self).__init__()
        self.activation_bit = activation_bit
        self.full_precision_flag = full_precision_flag
        self.running_stat = running_stat
        self.register_buffer('x_min', torch.zeros(1))
        self.register_buffer('x_max', torch.zeros(1))
        self.register_buffer('beta', torch.Tensor([beta]))
        self.register_buffer('beta_t', torch.ones(1))
        self.act_function = AsymmetricQuantFunction.apply

    def __repr__(self):
        return "{0}(activation_bit={1}, full_precision_flag={2}, running_stat={3}, Act_min: {4:.2f}, Act_max: {5:.2f})".format(
            self.__class__.__name__, self.activation_bit,
            self.full_precision_flag, self.running_stat, self.x_min.item(),
            self.x_max.item())

    def fix(self):
        """Fix/Freeze the activation range (stops dynamic updates)."""
        self.running_stat = False

    def unfix(self):
        """Unfreeze/Resume dynamic updates of the activation range."""
        self.running_stat = True

    def disable_observer(self):
        """Disable observer updates (called during 3-phase freezing)."""
        self.fix()

    def forward(self, x):
        if self.running_stat:
            x_min = x.data.min()
            x_max = x.data.max()
            self.x_min += -self.x_min + min(self.x_min, x_min)
            self.x_max += -self.x_max + max(self.x_max, x_max)

        if not self.full_precision_flag:
            quant_act = self.act_function(x, self.activation_bit, self.x_min, self.x_max)
            return quant_act
        else:
            return x


class QuantActPreLu(Module):
    """Quantized PReLU module wrapping weight quantization and post-activation quantization."""
    def __init__(self, act_bit, full_precision_flag=False, running_stat=True):
        super(QuantActPreLu, self).__init__()
        self.activation_bit = act_bit
        self.full_precision_flag = full_precision_flag
        self.running_stat = running_stat
        self.act_function = AsymmetricQuantFunction.apply
        self.quantAct = QuantAct(activation_bit=act_bit, running_stat=True)

    def __repr__(self):
        return "{0}(activation_bit={1}, full_precision_flag={2})".format(
            self.__class__.__name__, self.activation_bit, self.full_precision_flag)

    def set_param(self, prelu):
        self.weight = Parameter(prelu.weight.data.clone())

    def fix(self):
        self.running_stat = False
        self.quantAct.fix()

    def unfix(self):
        self.running_stat = True
        self.quantAct.unfix()

    def disable_observer(self):
        self.fix()

    def forward(self, x):
        w = self.weight
        if not self.full_precision_flag:
            x_transform = w.data.detach()
            a_min = x_transform.min(dim=0).values
            a_max = x_transform.max(dim=0).values
            w = self.act_function(self.weight, self.activation_bit, a_min, a_max)

        x = F.prelu(x, weight=w)
        x = self.quantAct(x)
        return x


class Quant_Linear(Module):
    """Quantized Linear layer wrapping weight asymmetric quantization."""
    def __init__(self, weight_bit, full_precision_flag=False):
        super(Quant_Linear, self).__init__()
        self.full_precision_flag = full_precision_flag
        self.weight_bit = weight_bit
        self.weight_function = AsymmetricQuantFunction.apply

    def __repr__(self):
        return "{0}(weight_bit={1}, full_precision_flag={2})".format(
            self.__class__.__name__, self.weight_bit, self.full_precision_flag)

    def set_param(self, linear):
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = Parameter(linear.weight.data.clone())
        if linear.bias is not None:
            self.bias = Parameter(linear.bias.data.clone())
        else:
            self.bias = None

    def forward(self, x):
        w = self.weight
        if not self.full_precision_flag:
            x_transform = w.data.detach()
            w_min = x_transform.min(dim=1).values
            w_max = x_transform.max(dim=1).values
            w = self.weight_function(self.weight, self.weight_bit, w_min, w_max)
        return F.linear(x, weight=w, bias=self.bias)


class Quant_Conv2d(Module):
    """Quantized Conv2d layer wrapping weight asymmetric quantization."""
    def __init__(self, weight_bit, full_precision_flag=False):
        super(Quant_Conv2d, self).__init__()
        self.full_precision_flag = full_precision_flag
        self.weight_bit = weight_bit
        self.weight_function = AsymmetricQuantFunction.apply

    def __repr__(self):
        return "{0}(weight_bit={1}, full_precision_flag={2})".format(
            self.__class__.__name__, self.weight_bit, self.full_precision_flag)

    def set_param(self, conv):
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        self.weight = Parameter(conv.weight.data.clone())
        if conv.bias is not None:
            self.bias = Parameter(conv.bias.data.clone())
        else:
            self.bias = None

    def forward(self, x):
        w = self.weight
        if not self.full_precision_flag:
            x_transform = w.data.contiguous().view(self.out_channels, -1)
            w_min = x_transform.min(dim=1).values
            w_max = x_transform.max(dim=1).values
            w = self.weight_function(self.weight, self.weight_bit, w_min, w_max)

        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


# ============================================================
# Model Quantization Wrappers
# ============================================================

def quantize_model(model, weight_bit=6, act_bit=6):
    """Recursively converts standard FP32 PyTorch modules into fake-quantized equivalents."""
    if type(model) == nn.Conv2d:
        quant_mod = Quant_Conv2d(weight_bit=weight_bit)
        quant_mod.set_param(model)
        return quant_mod
    elif type(model) == nn.Linear:
        quant_mod = Quant_Linear(weight_bit=weight_bit)
        quant_mod.set_param(model)
        return quant_mod
    elif type(model) == nn.PReLU:
        quant_mod = QuantActPreLu(act_bit=act_bit)
        quant_mod.set_param(model)
        return quant_mod
    elif type(model) in [nn.ReLU, nn.ReLU6]:
        return nn.Sequential(*[model, QuantAct(activation_bit=act_bit)])
    elif type(model) == nn.Sequential or isinstance(model, nn.Sequential):
        mods = OrderedDict()
        for n, m in model.named_children():
            # Check for block levels that require post-residual quantization
            if isinstance(m, IBasicBlock):
                mods[n] = nn.Sequential(*[
                    quantize_model(m, weight_bit=weight_bit, act_bit=act_bit),
                    QuantAct(activation_bit=act_bit)
                ])
            elif isinstance(m, Depth_Wise) and m.residual:
                mods[n] = nn.Sequential(*[
                    quantize_model(m, weight_bit=weight_bit, act_bit=act_bit),
                    QuantAct(activation_bit=act_bit)
                ])
            else:
                mods[n] = quantize_model(m, weight_bit=weight_bit, act_bit=act_bit)
        return nn.Sequential(mods)
    else:
        q_model = copy.deepcopy(model)
        for attr in dir(model):
            mod = getattr(model, attr)
            if isinstance(mod, nn.Module) and 'norm' not in attr:
                setattr(q_model, attr, quantize_model(mod, weight_bit=weight_bit, act_bit=act_bit))
        return q_model


def freeze_model(model):
    """Freeze all activation observers in the quantized model."""
    if isinstance(model, (QuantAct, QuantActPreLu)):
        model.fix()
    elif type(model) == nn.Sequential:
        for _, m in model.named_children():
            freeze_model(m)
    else:
        for attr in dir(model):
            mod = getattr(model, attr)
            if isinstance(mod, nn.Module) and 'norm' not in attr:
                freeze_model(mod)
        return model


def unfreeze_model(model):
    """Unfreeze all activation observers in the quantized model."""
    if isinstance(model, (QuantAct, QuantActPreLu)):
        model.unfix()
    elif type(model) == nn.Sequential:
        for _, m in model.named_children():
            unfreeze_model(m)
    else:
        for attr in dir(model):
            mod = getattr(model, attr)
            if isinstance(mod, nn.Module) and 'norm' not in attr:
                unfreeze_model(mod)
        return model
