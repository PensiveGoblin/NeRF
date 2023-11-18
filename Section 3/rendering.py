import torch



def rendering(model, rays_o, rays_d, tn, tf, nb_bins=100, device="cpu", white_background=True):
    #tn and tf are borns of integral (each are scalar?)
    #nb_bins is the number of points sampled?
    #rays_o origin of rays, Dim=(H*W,3)

    t=torch.linspace(tn,tf,nb_bins).to(device) # [nb_bins]
    delta=torch.cat((t[1:]-t[:-1] , torch.tensor([1e10]).to(device))) # we add infinity value at the end

    # Now we want to compute every point along the rays
    # naive approach:
    # rays_o+t*rays_d but the dimensions don t match

    x=rays_o.unsqueeze(1) + t.unsqueeze(0).unsqueeze(-1)*rays_d.unsqueeze(1) #[nb_rays,nb_bins,3]
    colors, density = model.intersect(x.reshape(-1,3))

    colors=colors.reshape(x.shape[0],nb_bins,3) # shape [nb_rays, nb_bins, 3]
    density=density.reshape(x.shape[0],nb_bins) # shape [nb_rays, nb_bins]

    alpha=1-torch.exp(-density * delta.unsqueeze(0)) # shape [nb_rays, nb_bins]
    weights=accumulated_transmittance(1-alpha)* alpha # shape [nb_rays, nb_bins]
    
    if white_background:
        c=(weights.unsqueeze(-1)  * colors).sum(1) #[nb_ray,3]
        weight_sum=weights.sum(-1)#[nb_ray]
        return c+1-weight_sum.unsqueeze(-1)
    else:
        c=(weights.unsqueeze(-1)  * colors).sum(1) #[nb_ray,3]
    return c


def accumulated_transmittance(beta):
    #beta = 1-alpha which is a trick to compute faster
    #input shape [nb_rays, nb_bins]
    #output shape [nb_rays, nb_bins]
    """
    T = torch.ones(beta.shape)
    for i in range(1,beta.shape[1]):
        T[:,i]=T[:,i-1]*beta[:,i-1]
    """

    #can be made faster using cumprod
    
    T=torch.cumprod(beta,1)
    #T[:,1:]=T[:,:-1] # might caus diff problems
    #T[:,0]=1.  might caus diff problems
    

    return torch.cat((torch.ones(T.shape[0],1, device=T.device),T[:,:-1]),dim=1)