from msmtools.flux import tpt as TPT

def tpt(T, source, target, **params):
    flux = TPT(T, source, target, **params)

    net_flux = flux.net_flux
    total_flux = flux.total_flux
    f_comm = flux.forward_committor
    b_comm = flux.backward_committor

    return net_flux, total_flux, f_comm, b_comm

