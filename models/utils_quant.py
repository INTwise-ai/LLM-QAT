# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# 2023.07.05 - Modified weight quantization
#              Meta Platforms, Inc. <zechunliu@meta.com>
#
# Copyright 2021 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import torch
import torch.nn as nn

T = 5


def soft_clamp(x, min, max):
    # x * LogisticSigmoid[x - min] *LogisticSigmoid[-x + max] + min*LogisticSigmoid[-x + min] + max*LogisticSigmoid[x - max]
    return x * torch.sigmoid(x - min) * torch.sigmoid(-x + max) + min * torch.sigmoid(-x + min) + max * torch.sigmoid(x - max)

CLAMP = soft_clamp
    

class SymQuantizer(torch.autograd.Function):
    """
    uniform quantization
    """

    @staticmethod
    def forward(ctx, input, clip_val, num_bits, layerwise, s=None):
        """
        :param ctx:
        :param input: tensor to be quantized
        :param clip_val: clip the tensor before quantization
        :param quant_bits: number of bits
        :return: quantized tensor
        """
        # input = torch.clamp(input, clip_val[0], clip_val[1])
        # input = torch.where(input < clip_val[1], input, clip_val[1])
        # input = torch.where(input > clip_val[0], input, clip_val[0])
        
        # Dynamic scaling for initialization

        if s is None:
            if layerwise:
                max_input = torch.max(torch.abs(input)).expand_as(input)
            else:
                if input.ndimension() <= 3:
                    # weight & hidden layer
                    max_input = (
                        torch.max(torch.abs(input), dim=-1, keepdim=True)[0]
                        .expand_as(input)
                        .detach()
                    )
                elif input.ndimension() == 4:
                    # TODO: attention score matrix, calculate alpha / beta per head
                    tmp = input.view(input.shape[0], input.shape[1], -1)
                    max_input = (
                        torch.max(torch.abs(tmp), dim=-1, keepdim=True)[0]
                        .unsqueeze(-1)
                        .expand_as(input)
                        .detach()
                    )
                else:
                    raise ValueError
            # Dynamic scaling: scale = max representable value / max input value
            s = (2 ** (num_bits - 1) - 1) / (max_input + 1e-6)

        max_repr = 2 ** (num_bits - 1) - 1
        min_repr = -2 ** (num_bits - 1)

        input_scaled = input * s
        output = CLAMP(torch.round(input_scaled), min_repr, max_repr).div(s + 1e-6)

        # -----BEGIN INTWise Estimator-----
        delta = input_scaled - torch.floor(input_scaled) + 0.5

        ctx.save_for_backward(input, delta, clip_val) # save for later use
        # -----END INTWise Estimator-----

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx: saved non-clipped full-precision tensor and clip_val
        :param grad_output: gradient ert the quantized tensor
        :return: estimated gradient wrt the full-precision tensor
        """
        
        input, delta, clip_val = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(clip_val[1])] = 0
        grad_input[input.le(clip_val[0])] = 0

        # -----BEGIN INTWise Estimator-----

        # single hump approximation
        grad_mat = math.exp(-T) + torch.exp(-T * delta) * T / (1 + torch.exp(-T * delta)) ** 2

        # element-wise product
        grad_input = grad_input * grad_mat
        
        # -----END INTWise Estimator-----


        return grad_input, None, None, None





class QuantizeLinear(nn.Linear):
    def __init__(
        self,
        *kargs,
        bias=False,
        w_bits=32,
        a_bits=32,
        act_layerwise=False,
        weight_layerwise=False,
    ):
        super(QuantizeLinear, self).__init__(*kargs, bias=False)
        self.w_bits = w_bits
        self.a_bits = a_bits
        self.act_layerwise = act_layerwise
        self.weight_layerwise = weight_layerwise
        # params for weight quant
        # if self.w_bits < 32:
        #     self.weight_clip_val = Parameter(torch.tensor([-2.0, 2.0]), requires_grad=False)
        if self.a_bits < 32 and self.a_bits > 2:
            self.act_quantizer = SymQuantizer
            # Learnable scaling factor
            self.s_weight = nn.Parameter(torch.tensor([1.0]), requires_grad=False)
            self.s_input = nn.Parameter(torch.tensor([1.0]), requires_grad=False)

    def forward(self, input_):
        # quantize weight
        assert len(self.weight.size()) == 2
        real_weights = self.weight

        if self.w_bits >= 32:
            weight = self.weight
        elif self.w_bits >= 3:
            weight_clip_val = torch.tensor([-2.0, 2.0])
            weight = SymQuantizer.apply(
                real_weights, weight_clip_val, self.w_bits, self.weight_layerwise, s=self.s_weight
            )
        else:
            if self.w_bits == 1:
                if self.weight_layerwise:
                    scaling_factor = torch.mean(abs(real_weights)).detach()
                else:
                    scaling_factor = torch.mean(
                        abs(real_weights), dim=1, keepdim=True
                    ).detach()
                quan_weights_no_grad = scaling_factor * (
                    torch.sign(real_weights / scaling_factor)
                )
            # elif self.w_bits == 2:
            #     scaling_factor = 4/3 * torch.mean(abs(real_weights), dim=1, keepdim=True).detach()
            #     quan_weights_no_grad = scaling_factor * (torch.round(torch.clamp(real_weights/scaling_factor, -1, 1)))
            else:
                num_bits = 2 ** (self.w_bits - 1)
                clip_val = 1 - 1e-2
                if self.weight_layerwise:
                    scaling_factor = 2 * torch.mean(abs(real_weights)).detach()
                else:
                    scaling_factor = (
                        2 * torch.mean(abs(real_weights), dim=1, keepdim=True).detach()
                    )
                quan_weights_no_grad = (
                    scaling_factor
                    * (
                        torch.round(
                            torch.clamp(
                                real_weights / scaling_factor, -clip_val, clip_val
                            )
                            * num_bits
                            - 0.5
                        )
                        + 0.5
                    )
                    / num_bits
                )

            weight = (
                quan_weights_no_grad.detach() - real_weights.detach() + real_weights
            )
        # Quantize inputs
        if self.a_bits < 32 and self.a_bits > 2:
            act_clip_val = torch.tensor([-2.0, 2.0])
            input_ = self.act_quantizer.apply(
                input_, act_clip_val, self.a_bits, self.act_layerwise, s=self.s_input
            )

        out = nn.functional.linear(input_, weight)
        if self.bias is not None:
            out += self.bias.view(1, -1).expand_as(out)

        return out
