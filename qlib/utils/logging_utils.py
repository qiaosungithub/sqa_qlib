import torch

def print_params(model: torch.nn.Module, logger):
    """Print the parameters of a PyTorch model, aligned by name."""
    params = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    if not params:
        logger.info("No trainable parameters.")
        return
    max_name_len = max(len(name) for name, _ in params)
    for name, param in params:
        logger.info(f"{name.ljust(max_name_len)} : {str(tuple(param.data.shape))}")
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters())}")