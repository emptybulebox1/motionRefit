import math
import torch
import torch.nn.functional as F
import math

def linear_beta_schedule(timesteps):
    scale = 1.0 # for 100 steps
    beta_start = scale * 0.0001
    beta_end = scale   * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)

class GaussianDiffusion:
    def __init__(
        self,
        device,
        fix_mode=False,
        text_emb=False,
        fixed_frames=2,
        seq_len=16,
        timesteps=100,
        beta_schedule='linear',
    ):
        self.device = device
        self.fix_mode = fix_mode            # autoregressive
        self.fixed_frames = fixed_frames    # number of frames to fix
        self.timesteps = timesteps
        self.text_emb = text_emb
        self.seq_len = seq_len
        
        
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            raise NotImplementedError('cosine schedule is not implemented yet!')
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')
        
        self.betas = betas.to(self.device)
        self.alphas = (1. - self.betas).to(self.device)
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0).to(self.device)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.).to(self.device)
        
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to(self.device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod).to(self.device)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod).to(self.device)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod).to(self.device)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1).to(self.device)
        
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        ).to(self.device)
        # below: log calculation clipped because the posterior variance is 0 at the beginning
        # of the diffusion chain
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min =1e-20)).to(self.device)
        
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        ).to(self.device)
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * torch.sqrt(self.alphas)
            / (1.0 - self.alphas_cumprod)
        ).to(self.device)
    
    # get the param of given timestep t
    def _extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t).float()
        out = out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(self.device)
        return out
    
    # forward diffusion (using the nice property): q(x_t | x_0)
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    # Get the mean and variance of q(x_t | x_0).
    def q_mean_variance(self, x_start, t):
        mean = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = self._extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance
    
    # Compute the mean and variance of the diffusion posterior: q(x_{t-1} | x_t, x_0)
    def q_posterior_mean_variance(self, x_start, x_t, t):
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    # compute x_0 from x_t and pred noise: the reverse of `q_sample`
    def predict_start_from_noise(self, x_t, t, noise):
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )
    
    # compute predicted mean and variance of p(x_{t-1} | x_t)
    def p_mean_variance(self, model, x_t, t, clip_denoised=True, **kwargs):
        # predict noise using model
        assert 'text' in kwargs, 'text is required'
        assert 'prog_ind' in kwargs, 'prog_ind is required'
        assert 'joints_orig' in kwargs, 'joints_orig is required'
        pred_noise = model(x_t, t, 
                           text_emb=kwargs['text'], 
                           prog_ind=kwargs['prog_ind'], 
                           joints_orig=kwargs['joints_orig'])
        
        # use cfg for text embedding:
        if kwargs['use_cfg']:
            pred_noise_empty = model(x_t, t, 
                            text_emb=torch.zeros_like(kwargs['text']), 
                            prog_ind=kwargs['prog_ind'], 
                            joints_orig=kwargs['joints_orig'])
            pred_noise = pred_noise_empty + kwargs['cfg_alpha'] * (pred_noise - pred_noise_empty)
            
        # get the predicted x_0: different from the algorithm2 in the paper
        x_recon = self.predict_start_from_noise(x_t, t, pred_noise)
        
        if clip_denoised:
            x_recon = torch.clamp(x_recon, min=-1., max=1.)
        model_mean, posterior_variance, posterior_log_variance = \
                    self.q_posterior_mean_variance(x_recon, x_t, t)
        return model_mean, posterior_variance, posterior_log_variance
    
    # denoise_step: sample x_{t-1} from x_t and pred_noise
    # @torch.no_grad()
    def p_sample(self, model, x_t, t, clip_denoised=True, **kwargs):
        if 'disc_model' in kwargs:
            disc_model = kwargs['disc_model']
            try:
                cg_alpha = kwargs['cg_alpha'] # default 1.0
                cg_diffusion_steps = kwargs['cg_diffusion_steps']
            except:
                print("cg_alpha and cg_diffusion_steps are not provided!")
                print("Using default values: cg_alpha=1.0, cg_diffusion_steps=5")
                cg_alpha = 1.0
                cg_diffusion_steps = 5
        # predict mean and variance
        model_mean, _, model_log_variance = self.p_mean_variance(model, x_t, t,
                                                    clip_denoised=clip_denoised, **kwargs)
        model_mean = torch.tensor(model_mean, requires_grad=True)
        noise = torch.randn_like(x_t)
        # no noise when t == 0
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1))))
        if t.item() < cg_diffusion_steps:
            pred_syn = disc_model(model_mean, t) # y = f(theta, x)  theta fixed 
            pred_syn.backward()
           
            grad = model_mean.grad * cg_alpha
            model_mean = model_mean - nonzero_mask * (0.5 * model_log_variance).exp() * grad

        # compute x_{t-1}
        pred_img = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return pred_img
    
    # denoise: reverse diffusion
    # @torch.no_grad()
    def p_sample_loop(self, model, shape, fixed_points=None, **kwargs):
        batch_size = shape[0]
        device = next(model.parameters()).device
        
        # start from pure noise (for each example in the batch)
        img = torch.randn(shape, device=device)
        # notice that if we are in fixed mode, we need to fix the first 2 frames
        if self.fix_mode:
            assert not (fixed_points is None), 'fixed_points is required for fixed mode'
            img[:, :self.fixed_frames, :] = fixed_points
        imgs = []

        for i in reversed(range(0, self.timesteps)):
            img = self.p_sample(model, img, torch.full((batch_size,), i, device=device, dtype=torch.long), **kwargs)
            if self.fix_mode:
                img[:, :self.fixed_frames, :] = fixed_points
            imgs.append(img)
        return imgs


    # sample new images
    # @torch.no_grad()
    def sample(self, model, batch_size=1, seq_len=16, channels=135, 
               fixed_points=None, clip_denoised=True, **kwargs):
        return self.p_sample_loop(model, shape=(batch_size, seq_len, channels), 
                     fixed_points=fixed_points, clip_denoised=clip_denoised, **kwargs)
       
    # compute train losses
    def train_losses(self, model, x_start, t, mask=None, **kwargs):
        assert not (mask is None and self.fixed_mode), 'mask is required for fixed mode'
        if mask is None:
            mask = torch.zeros_like(x_start).to(dtype=torch.bool, device=self.device)

        mask_inv = torch.logical_not(mask)
        # generate random noise
        noise = torch.randn_like(x_start).to(device=self.device)
        noise[mask] = 0.
        
        # get x_t
        x_noisy = self.q_sample(x_start, t, noise=noise)
        predicted_noise = model(x_noisy, t, text_emb=kwargs['text'], prog_ind=kwargs['prog_ind'], joints_orig=kwargs['joints_orig'])

        loss = F.smooth_l1_loss(noise[mask_inv], predicted_noise[mask_inv])
        return loss