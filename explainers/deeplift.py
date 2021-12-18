import copy
import numpy as np
import torch
from torch.autograd import Variable
from explainers.base import Explainer


class DeepLIFTExplainer(Explainer):

    def __init__(self, device, gnn_model_path):
        super(DeepLIFTExplainer, self).__init__(device, gnn_model_path)

        # Rescale rules for activation
        def hook_fn_backward(module, grad_input, grad_output):

            delta_out = self.delta_y.pop()
            delta_in = self.delta_x.pop()

            is_near_zero = torch.zeros_like(delta_in)
            is_near_zero[abs(delta_in) < self.NEAR_ZERO_THRESHOLD] = 1.0
            far_from_zero = 1.0 - is_near_zero

            # preventing computation overflows
            grads = [None for _ in grad_input]
            grads[0] = (is_near_zero * grad_input[0] + far_from_zero * grad_output[0] * (
                        delta_out / delta_in)).float()
            return tuple(grads)

        self.NEAR_ZERO_THRESHOLD = 1e-3

        for module in self.model.modules():
            if 'ReLU' in module.__class__.__name__ or\
                    module.__class__.__name__ in ['Sigmoid', 'Tanh']:
                module.register_backward_hook(hook_fn_backward)

    def __reference__(self, graph):

        # obtain input and output data
        def hook_fn_forward(module, input, output):
            activation_input.append(input[0])
            activation_output.append(output)

        activation_input = []
        activation_output = []

        model = torch.load(self.path).to(self.device)
        model.to(self.device)
        for module in model.modules():
            if 'ReLU' in module.__class__.__name__ or \
                    module.__class__.__name__ in ['Sigmoid', 'Tanh']:
                module.register_forward_hook(hook_fn_forward)


        ref_model = torch.load(self.path).to(self.device)
        ref_model.to(self.device)
        for module in ref_model.modules():
            if 'ReLU' in module.__class__.__name__ or \
                    module.__class__.__name__ in ['Sigmoid', 'Tanh']:
                module.register_forward_hook(hook_fn_forward)

        model(
            graph.x,
            graph.edge_index,
            graph.edge_attr,
            graph.batch
        )
        ref_model(
            torch.zeros_like(graph.x).float(),
            graph.edge_index,
            torch.zeros_like(graph.edge_attr).float(),
            graph.batch
        )

        act_in = np.array(activation_input)
        act_out = np.array(activation_output)

        half = int(len(act_in) / 2)
        self.delta_x = list(act_in[:half] - act_in[half:])
        self.delta_y = list(act_out[:half] - act_out[half:])

    def explain_graph(self, graph,
                      model=None,
                      draw_graph=0,
                      vis_ratio=0.2):

        if model == None:
            model = self.model

        self.__reference__(graph)
        edge_attr = Variable(graph.edge_attr, requires_grad=True)

        # For DeepLIFTExplainer, self.model needs to
        # forward once to trigger the backward hooks
        pred = model(graph.x,
                        graph.edge_index,
                        edge_attr,
                        graph.batch)

        pred[0, graph.y].backward()

        edge_imp = pow(edge_attr.grad, 2).sum(dim=1).cpu().numpy()
        edge_imp = self.norm_imp(edge_imp)

        if draw_graph:
            self.visualize(graph, edge_imp, self.name,vis_ratio=vis_ratio)
        self.last_result = (graph, edge_imp)

        return edge_imp
