import torch
import torch.nn as nn


class SDT(nn.Module):
    """
    Soft Decision Tree implementation with regularization.

    Parameters:
        input_dim : int
            Dimension of input features
        output_dim : int
            Dimension of output predictions
        depth : int, default=5
            Tree depth where depth=0 means single root node
        lambda_reg : float, default=1e-3
            Coefficient for the tree regularization term
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        depth=5,
        lambda_reg=1e-3
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.depth = depth
        self.lambda_reg = lambda_reg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._validate_parameters()

        # Tree structure parameters
        self.internal_node_num = 2 ** self.depth - 1
        self.leaf_node_num = 2 ** self.depth

        # Layer-wise regularization coefficients
        self.regularization_factors = [
            self.lambda_reg * (2 ** (-d)) for d in range(self.depth)
        ]

        # Internal node structure: Linear + Sigmoid
        self.internal_nodes = nn.Sequential(
            nn.Linear(self.input_dim + 1, self.internal_node_num, bias=False),
            nn.Sigmoid()
        )

        # Leaf node prediction layer
        self.leaf_predictions = nn.Linear(
            self.leaf_node_num, 
            self.output_dim, 
            bias=False
        )

    def forward(self, X, is_train=False):
        """
        Forward pass with optional regularization term.
        
        Parameters:
            X : torch.Tensor
                Input tensor (batch_size, input_dim)
            is_train : bool, default=False
                Flag to return regularization term during training
            
        Returns:
            torch.Tensor or tuple
        """
        path_probs, reg_term = self._compute_tree_paths(X)
        predictions = self.leaf_predictions(path_probs)
        
        return (predictions, reg_term) if is_train else predictions

    def _compute_tree_paths(self, X):
        """
        Compute path probabilities with dimension alignment fix.
        """
        batch_size = X.size(0)
        X = self._add_bias_term(X)
        
        # Get routing probabilities for all nodes
        routing_logits = self.internal_nodes(X)
        routing_probs = torch.stack([routing_logits, 1 - routing_logits], dim=2)
        
        path_probs = X.new_ones((batch_size, 1, 1))
        reg_term = torch.tensor(0.0).to(self.device)
        
        start_idx, end_idx = 0, 1
        for layer in range(self.depth):
            layer_probs = routing_probs[:, start_idx:end_idx, :]
            # Reshape layer probabilities for dimension alignment
            layer_probs_flat = layer_probs.view(batch_size, -1, 1)

            path_probs = path_probs.repeat_interleave(2, dim=1) * layer_probs_flat

            reg_term += self._compute_layer_regularization(layer, path_probs, layer_probs)
            start_idx, end_idx = end_idx, end_idx + 2 ** (layer + 1)
        
        return path_probs.view(batch_size, -1), reg_term                

    def _compute_layer_regularization(self, layer, path_probs, layer_probs):
        """
        Compute entropy-based regularization for a layer.
        """
        batch_size = path_probs.size(0)
        current_paths = path_probs.view(batch_size, -1)
        next_paths = layer_probs.view(batch_size, -1)
        
        reg = torch.tensor(0.0).to(self.device)
        layer_factor = self.regularization_factors[layer]
        
        for node_idx in range(next_paths.size(1)):
            parent_idx = node_idx // 2
            alpha = torch.sum(
                next_paths[:, node_idx] * current_paths[:, parent_idx]
            ) / (torch.sum(current_paths[:, parent_idx]) + 1e-8)
            
            node_reg = -0.5 * (alpha.log() + (1 - alpha).log())
            reg += layer_factor * node_reg
            
        return reg

    def _add_bias_term(self, X):
        """
        Add constant bias term to input features.
        """
        return torch.cat([
            torch.ones(X.size(0), 1).to(self.device),
            X.view(X.size(0), -1)
        ], dim=1)

    def _validate_parameters(self):
        """
        Validate initialization parameters.
        """
        if self.depth < 0:
            raise ValueError(f"Tree depth must be ≥ 0, got {self.depth}")
            
        if self.lambda_reg < 0:
            raise ValueError(
                f"Regularization coefficient must be ≥ 0, got {self.lambda_reg}"
            )