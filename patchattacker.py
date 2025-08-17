import torch

class PatchAttacker:
    def __init__(self, model, mean, std, patch_size, image_size=224, epsilon=10.0, steps=1000, step_size=0.05,
                  device=None):
        device = device or next(model.parameters()).device
        mean = torch.tensor(mean, device=device, dtype=torch.float32)
        std  = torch.tensor(std,  device=device, dtype=torch.float32)

        self.eps_l2 = float((epsilon / std).mean().item())

        self.model = model.to(device).eval()
        self.device = device
        self.mean, self.std = mean, std
        self.image_size = image_size
        self.steps = steps

        # valid pixel bounds in normalized space
        self.lb = ((0.0 - mean) / std)[None, :, None, None]          # [1,C,1,1]
        self.ub = ((1.0 - mean) / std)[None, :, None, None]          # [1,C,1,1]

        self.patch_w = int((patch_size*(image_size**2))**0.5)
        self.patch_l = int((patch_size*(image_size**2))**0.5)

        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')

    @torch.no_grad()
    def _rand_patch_indices(self, N, H, W):
        # inclusive upper bound for placement
        wy = torch.randint(0, H - self.patch_w + 1, (N,1), device=self.device)
        wx = torch.randint(0, W - self.patch_l + 1, (N,1), device=self.device)
        b  = torch.arange(N, device=self.device)[:, None]
        c1 = torch.zeros((N,1), dtype=torch.long, device=self.device)  # channel index (broadcast later)
        idx = torch.cat([b, c1, wy, wx], dim=1)  # [N,4]: (n, 0, y, x)

        idx_list = [idx]
        for dy in range(self.patch_w):
            for dx in range(self.patch_l):
                idx_list.append(idx + torch.tensor([0,0,dy,dx], device=self.device))
        return torch.cat(idx_list, dim=0)  # [(N*patch_w*patch_l),4]

    def perturb(self, inputs, labels, norm='l2', random_count=1):
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        N, C, H, W = inputs.shape
        best_x = None
        best_loss = None

        for _ in range(random_count):
            # mask: [N,1,H,W] -> expand to channels when used
            idx_list = self._rand_patch_indices(N, H, W)
            mask = torch.zeros((N, 1, H, W), dtype=torch.bool, device=self.device)
            mask[idx_list[:,0], idx_list[:,1], idx_list[:,2], idx_list[:,3]] = True
            mask_c = mask.expand(-1, C, -1, -1)   # [N,C,H,W]
            
            x = inputs.clone()

            x_init = inputs.detach().clone()
            x.requires_grad_(True)

            for _ in range(self.steps):
                # Only the patch region is taken from x; elsewhere from x_init
                adv = torch.where(mask_c, x, x_init)
                logits = self.model(adv)
                loss_ind = self.criterion(logits, labels)         # [N]
                loss = loss_ind.sum()

                grad, = torch.autograd.grad(loss, x, retain_graph=False, create_graph=False)

                g = grad.view(N, -1)
                g_norm = g.norm(2, dim=1).clamp_min(1e-12).view(N,1,1,1)
                step = (grad / g_norm) * (self.eps_l2 * 0.05)   # step size ~5% of eps_l2

                # update only inside the patch
                with torch.no_grad():
                    x.add_(step * mask_c.to(step.dtype))

                    delta = (x - x_init) * mask_c
                    d = delta.view(N, -1).norm(2, dim=1).view(N,1,1,1).clamp_min(1e-12)
                    scale = (self.eps_l2 / d).clamp_max(1.0)
                    x.copy_(x_init + delta * scale)

                    # pixel bounds
                    x.clamp_(self.lb, self.ub)
                x.requires_grad_(True)

            # evaluate this random start’s result
            with torch.no_grad():
                adv = torch.where(mask_c, x, x_init)
                loss_now = self.criterion(self.model(adv), labels)  # [N]

            if best_loss is None:
                best_loss = loss_now
                best_x = adv
            else:
                take_new = best_loss.lt(loss_now)[:, None, None, None]  # keep the "worst" (largest loss)
                best_x = torch.where(take_new, adv, best_x)
                best_loss = torch.where(best_loss.lt(loss_now), loss_now, best_loss)

        return best_x
