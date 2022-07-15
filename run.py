import numpy as np
import matplotlib.pyplot as plt
import h5py

import tools

make_plots = True

# estimates of gain fluctuation power spectrum for each feed from season 1
filename = 'Cf_prior_data.hdf5'
with h5py.File(filename, mode="r") as my_file:
    alpha_prior = np.array(my_file['alpha_prior'][()])
    fknee_prior = np.array(my_file['fknee_prior'][()])
    sigma0_prior = np.array(my_file['sigma0_prior'][()])

basis_size = 50  # destriper basis length in number of samples

# we want to make 1D flattened arrays to put into destriper
d_full = np.array([], dtype=float)
d_clean_full = np.array([], dtype=float)
ra_full = np.array([], dtype=float)
dec_full = np.array([], dtype=float)
dg_full = np.array([], dtype=float)
l1_dir = '/mn/stornext/d22/cmbco/comap/protodir/level1/'
file_list = ['2021-07/comap-0022001-2021-07-15-142002.hd5', '2021-12/comap-0025911-2021-12-10-082530.hd5']  # ['../comap-0025911-2021-12-10-082530.hd5']
for filename in file_list:
    d_mean, d_clean_mean, ra, dec, dg = tools.filter_obsid_data(l1_dir + filename, sigma0_prior, fknee_prior, alpha_prior, make_plots=make_plots)

    n_feeds, Ntod = d_mean.shape
    
    n_baselines = Ntod // basis_size
    Ntod = basis_size * n_baselines  # make sure we get a whole number of baselines in the data, so one baseline does not overlap feeds or scans

    d_mean = d_mean[:, :Ntod].flatten()
    d_clean_mean = d_clean_mean[:, :Ntod].flatten()
    dg = dg[:, :Ntod].flatten()
    ra = ra[:, :Ntod].flatten()
    dec = dec[:, :Ntod].flatten()

    d_full = np.append(d_full, d_mean)
    d_clean_full = np.append(d_clean_full, d_clean_mean)
    ra_full = np.append(ra_full, ra)
    dec_full = np.append(dec_full, dec)
    dg_full = np.append(dg_full, dg)

# format needed for destriper function
pointing = np.zeros((len(ra_full), 2))
pointing[:, 0] = ra_full
pointing[:, 1] = dec_full

nside = 150

lims = np.array([[82, 85], [20, 24]])  # Tau A limits

ra_bins = np.linspace(lims[0, 0], lims[0, 1], nside+1)
dec_bins = np.linspace(lims[1, 0], lims[1, 1], nside+1)


# we do no noise weighting in this destriping, so this is far from optimal
map_destripe_clean, map_nw, hitmap = tools.make_map_from_datasets(d_clean_full, pointing, lims, nside, basis_size=basis_size)
map_destripe, map_nw, hitmap = tools.make_map_from_datasets(d_full, pointing, lims, nside, basis_size=basis_size)
map_dg, map_nw, hitmap = tools.make_map_from_datasets(dg_full, pointing, lims, nside, basis_size=basis_size)

np.savetxt('map_clean.txt', map_destripe_clean)
np.savetxt('map_unclean.txt', map_destripe)
np.savetxt('map_dg.txt', map_dg)
np.savetxt('nhit.txt', hitmap)

vmax = 0.1

plt.figure()
plt.title('nhit')
plt.imshow(hitmap, interpolation='none')
plt.savefig('figures/nhit.png')

plt.figure()
plt.title('dg')
plt.imshow(map_dg, interpolation='none')
plt.savefig('figures/dg.png')
plt.colorbar()

plt.figure()
plt.title('destriped difference')
plt.imshow(map_destripe_clean - map_destripe, interpolation='none', vmin=-vmax, vmax=vmax)
plt.savefig('figures/destriped_diff.png')
plt.colorbar()

plt.figure()
plt.title('destriped')
plt.imshow(map_destripe, interpolation='none', vmin=-vmax, vmax=vmax)
plt.savefig('figures/destriped.png')
plt.colorbar()

plt.figure()
plt.title('destriped_clean')
plt.imshow(map_destripe_clean, interpolation='none', vmin=-vmax, vmax=vmax)
plt.savefig('figures/destriped_clean.png')
plt.colorbar()

plt.show()
