import torch



def override_lr_and_reset(optimizer_d, lr: float) -> None:
    """Overrides discriminator LR and clears optimizer state."""
    if optimizer_d is None:
        return
    for pg in optimizer_d.param_groups:
        pg['lr'] = float(lr)
        if 'initial_lr' in pg:
            pg['initial_lr'] = float(lr)
    optimizer_d.state.clear()


def load_discriminator_weights(ckpt_path, netD, optimizer_d, lr_after: float) -> None:
    """Loads discriminator weights from checkpoint and resets optimizer state."""
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt['netD'] if isinstance(ckpt, dict) and 'netD' in ckpt else ckpt
    netD.load_state_dict(state_dict)
    override_lr_and_reset(optimizer_d, lr_after)
