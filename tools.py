import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import LinearOperator
import scipy

def find_index(arr, n, K):
    # Traverse the array
    for i in range(n):
         
        # If K is found
        if arr[i] == K:
            return i
             
        # If arr[i] exceeds K
        elif arr[i] > K:
            return i
             
    # If all array elements are smaller
    return n

def make_map_from_datasets(y, pointing, lims, nside, basis_size):
    Ntod = len(y)
    ra = pointing[:, 0]
    dec = pointing[:, 1]
    Nbasis = Ntod//basis_size

    data = np.ones(Ntod)
    col_idx = np.arange(Ntod, dtype=int)
    row_idx = np.arange(Ntod, dtype=int)//basis_size

    F = csc_matrix((data, (col_idx, row_idx)), shape=(Ntod, Nbasis))

    Nsidemap = nside
    Nmap = Nsidemap**2

    ra_bins = np.linspace(lims[0, 0], lims[0, 1], Nsidemap+1)
    dec_bins = np.linspace(lims[1, 0], lims[1, 1], Nsidemap+1)

    ra_inds = np.array([find_index(ra_bins, Nsidemap, ra_val) for ra_val in ra])
    dec_inds = np.array([find_index(dec_bins, Nsidemap, dec_val) for dec_val in dec])

    P = csc_matrix((np.ones(Ntod), (np.arange(Ntod, dtype=int), ra_inds + Nsidemap*dec_inds)), shape=(Ntod, Nmap))

    Corr_wn_inv = scipy.sparse.diags(np.ones(Ntod))

    PT = csc_matrix(P.T)
    FT = csc_matrix(F.T)
    inv_PT_C_P = scipy.sparse.diags(1.0/(PT.dot(Corr_wn_inv).dot(P)).diagonal())
    P_inv_PT_C_P = P.dot(inv_PT_C_P)
    FT_C_F = FT.dot(Corr_wn_inv).dot(F)
    FT_C_P_inv_PT_C_P = FT.dot(Corr_wn_inv.dot(P_inv_PT_C_P))
    PT_C_F = PT.dot(Corr_wn_inv).dot(F)

    def LHS(a):
        a1 = FT_C_F.dot(a)
        a2 = PT_C_F.dot(a)
        a3 = FT_C_P_inv_PT_C_P.dot(a2)
        return a1 - a3

    A = LinearOperator((Nbasis,Nbasis), matvec=LHS)
    b = F.T.dot(Corr_wn_inv).dot(y) - FT_C_P_inv_PT_C_P.dot(PT.dot(Corr_wn_inv).dot(y))

    def solve_cg(A, b):
        num_iter = 0
        x0 = np.zeros(Nbasis)
        a, info = scipy.sparse.linalg.cg(A, b, x0=x0)
        return a, info

    a, info = solve_cg(A, b)
    map_nw = inv_PT_C_P.dot(PT.dot(Corr_wn_inv).dot(y))
    map_destripe = inv_PT_C_P.dot(PT.dot(Corr_wn_inv).dot(y-F.dot(a)))
    template_map = inv_PT_C_P.dot(PT.dot(Corr_wn_inv).dot(F.dot(a)))
    hitmap = PT.dot(P).diagonal()

    map_destripe = map_destripe.reshape((Nsidemap, Nsidemap))
    map_nw = map_nw.reshape((Nsidemap, Nsidemap))
    template_map = template_map.reshape((Nsidemap, Nsidemap))
    hitmap = hitmap.reshape((Nsidemap, Nsidemap))
    return map_destripe, map_nw, hitmap

def bin_data_maxbin(arr, slope=1.0, cut=20, maxbin=2000):
    # code for binning power spectra in a nice way

    n = len(arr)
    n_coadd = []
    cumsum = 0
    while np.sum(n_coadd) < n:
        n_add = min(1 + int((cumsum//cut) ** slope), maxbin)
        n_coadd.append(n_add)
        cumsum += n_add
    sum = np.sum(n_coadd[:-1])
    n_coadd[-2] += n - sum
    n_coadd = np.array(n_coadd[:-1]).astype(int)
    newarr = np.zeros(len(n_coadd))
    cumsum = 0
    for i in range(len(n_coadd)):
        newarr[i] = np.sum(arr[cumsum:cumsum+n_coadd[i]]) / n_coadd[i]
        cumsum += n_coadd[i]
    return newarr

def PS_1f(freqs, sigma0, fknee, alpha):
    return sigma0**2*(1 + (freqs/fknee)**alpha)

def PS_1f_nown(freqs, sigma0, fknee, alpha):
    return sigma0**2*(freqs/fknee)**alpha

def gain_temp_sep(y, P, F, sigma0_g, fknee_g, alpha_g, samprate=50):
    freqs = np.fft.rfftfreq(len(y[0]), d=1.0/samprate)
    n_freqs, n_tod = y.shape
    Cf = PS_1f(freqs, sigma0_g, fknee_g, alpha_g)
    Cf[0] = 1
    
    sigma0_est = np.std(y[:,1:] - y[:,:-1], axis=1)/np.sqrt(2)
    sigma0_est = np.mean(sigma0_est)
    Z = np.eye(n_freqs, n_freqs) - P.dot(np.linalg.inv(P.T.dot(P))).dot(P.T)
    
    RHS = np.fft.rfft(F.T.dot(Z).dot(y))

    z = F.T.dot(Z).dot(F)
    a_bestfit_f = RHS/(z + sigma0_est**2/Cf)
    a_bestfit = np.fft.irfft(a_bestfit_f, n=n_tod)

    m_bestfit = np.linalg.inv(P.T.dot(P)).dot(P.T).dot(y - F*a_bestfit)
    
    return a_bestfit, m_bestfit


def fit_gain_fluctuations(y_feed, tsys, sigma0_prior, fknee_prior, alpha_prior, feed, make_plots=False):
    """
    Model: y(t, nu) = dg(t) + dT(t) / Tsys(nu) + alpha(t) / Tsys(nu) (nu - nu_0) / nu_0, nu_0 = 30 GHz
    """

    nsb, Nfreqs, Ntod = y_feed.shape

    scaled_freqs = np.linspace(-4.0 / 30, 4.0 / 30, 4 * 1024)  # (nu - nu_0) / nu_0
    scaled_freqs = scaled_freqs.reshape((4, 1024))
    scaled_freqs[(0, 2), :] = scaled_freqs[(0, 2), ::-1]  # take into account flipped sidebands

    P = np.zeros((4, Nfreqs, 2))
    F = np.zeros((4, Nfreqs, 1))
    P[:, :,0] = 1 / tsys
    P[:, :,1] = scaled_freqs/tsys
    F[:, :,0] = 1

    end_cut = 100
    # Remove edge frequencies and the bad middle frequency
    y_feed[:, :4] = 0
    y_feed[:, -end_cut:] = 0
    P[:, :4] = 0
    P[:, -end_cut:] = 0
    F[:, :4] = 0
    F[:, -end_cut:] = 0
    F[:, 512] = 0
    P[:, 512] = 0
    y_feed[:, 512] = 0

    calibrated = y_feed * tsys[:, :, None]  # only used for plotting
    calibrated[(0, 2), :] = calibrated[(0, 2), ::-1]

    # Reshape to flattened grid
    P = P.reshape((4 * Nfreqs, 2))
    F = F.reshape((4 * Nfreqs, 1))
    y_feed = y_feed.reshape((4 * Nfreqs, Ntod))

    # Fit dg, dT and alpha
    a_feed, m_feed = gain_temp_sep(y_feed, P, F, sigma0_prior, fknee_prior, alpha_prior)
    dg = a_feed[0]
    dT = m_feed[0]
    alpha = m_feed[1]

    ## Plotting section 
    if make_plots:
        maxind = np.argmax(y_feed.mean(0), axis=0)
        plt.figure()
        plt.title('Feed %i' % (feed+1))
        plt.plot(y_feed[:, maxind] / F[:, 0], label='normalized data, y')
        plt.plot(P.dot(m_feed)[:, maxind] / F[:, 0], label='Pm, i.e. best fit linear temp model')
        plt.legend()
        plt.savefig('figures/Pm_model_%02i.png' % (feed+1), bbox_inches='tight')
        plt.figure()
        plt.title('Feed %i' % (feed+1))
        f = np.linspace(26, 34, 4 * 1024)
        plt.plot(f, (calibrated[:, :, maxind].flatten()) / (calibrated[:, :, maxind].flatten() != 0), label='calibrated data')
        x = np.linspace(-4.0 / 30, 4.0 / 30, 4 * 1024)
        plt.plot(f, dT[maxind] + x * alpha[maxind], label=r'$\delta T + \alpha (\nu - \nu_0) / \nu_0$')
        plt.xlabel('Frequency [GHz]')
        plt.ylabel('Brightness temperature [K]')
        plt.savefig('figures/calibrated_%02i.png' % (feed+1), bbox_inches='tight')
        plt.legend()
    return dg, dT, alpha

def filter_obsid_data(filename, sigma0_prior, fknee_prior, alpha_prior, n_cut=4000, make_plots=False):
    n_feeds = 1 # choose fewer feeds for quicker runtime (during debugging etc)

    # removing n_cut samples at start and end of observation to avoid vane measurements 
    # (this should clearly be done in a more proper way)
    with h5py.File(filename, mode="r") as my_file:
        ra = np.array(my_file['/spectrometer/pixel_pointing/pixel_ra'][:n_feeds, n_cut:-n_cut])
        dec = np.array(my_file['/spectrometer/pixel_pointing/pixel_dec'][:n_feeds, n_cut:-n_cut])
    obsid = filename.split('/')[-1][6:13]
    print(obsid)
    n_tod = len(ra[0])

    # calibration database
    filename_tsys = 'level1_database_selected.h5'
    with h5py.File(filename_tsys, mode="r") as my_file:
        tsys_all = my_file['/obsid/' + obsid + '/Tsys_obsidmean'][()][:n_feeds]

    d_mean = np.zeros((n_feeds, n_tod))
    d_clean_mean = np.zeros((n_feeds, n_tod))
    dg = np.zeros((n_feeds, n_tod))
    for feed in range(n_feeds):
        tsys = tsys_all[feed]
        with h5py.File(filename, mode="r") as my_file:
            tod = np.array(my_file['/spectrometer/tod'][feed, :, :, n_cut:-n_cut])

        # normalization
        tod_mean = tod.mean(2)
        d = tod / tod_mean[:, :, None] - 1.0 


        # Fit gain fluctuations
        dg[feed], dT, alpha = fit_gain_fluctuations(d, tsys, sigma0_prior[feed], fknee_prior[feed], alpha_prior[feed], feed, make_plots=make_plots)

        # Plotting section (just for diagnostic purposes)
        if make_plots:
            print(feed + 1)
            plt.figure()
            plt.title('Feed %i' % (feed+1))
            plt.plot(d.mean((0, 1)), label='freq. averaged normalized data')
            plt.plot(dg[feed], label=r'$\delta g$, best fit gain fluctuations')
            plt.plot(d.mean((0, 1)) - dg[feed], label='gain subtracted')
            plt.legend()
            plt.xlabel('Time samples')
            plt.savefig('figures/gain_tod_%02i.png' % (feed+1), bbox_inches='tight')
            plt.show()

        # remove gain fluctuations
        d_clean = d[:, :, :] - dg[feed][None, None, :]

        # calibrate
        d = d * tsys[:, :, None]
        d_clean = d_clean * tsys[:, :, None]

        # sum over frequencies (should probably mask some of the frequencies)
        d_mean[feed] = np.sum(d / tsys[:, :, None] ** 2, (0, 1)) / np.sum(1.0 / tsys[:, :] ** 2, (0, 1))
        d_clean_mean[feed] = np.sum(d_clean / tsys[:, :, None] ** 2, (0, 1)) / np.sum(1.0 / tsys[:, :] ** 2, (0, 1))

        ## Plotting 
        if make_plots:
            n_ps = 10000  # number of samples to include in power spectrum (we want to avoid actual tauA data)
            samprate= 50
            f = np.abs(np.fft.rfftfreq(n_ps) * samprate)
            observes_tauA_early = [5, 12, 13, 14, 15, 16]  # specific to obsid 22001
            if (feed+1) in observes_tauA_early:
                ps = np.abs(np.fft.rfft(d_mean[feed, -n_ps:])) ** 2
                ps_clean = np.abs(np.fft.rfft(d_clean_mean[feed, -n_ps:])) ** 2
            else:
                ps = np.abs(np.fft.rfft(d_mean[feed, :n_ps])) ** 2
                ps_clean = np.abs(np.fft.rfft(d_clean_mean[feed, :n_ps])) ** 2
            cut = 3
            slope = 0.5
            f = bin_data_maxbin(f, slope=slope, cut=cut, maxbin=4000)
            ps = bin_data_maxbin(ps, slope=slope, cut=cut, maxbin=4000)
            ps_clean = bin_data_maxbin(ps_clean, slope=slope, cut=cut, maxbin=4000)
            
            plt.figure()
            plt.title('Feed %i' % (feed+1))
            plt.loglog(f, ps, label='continuum signal')
            plt.loglog(f, ps_clean, label='gain cleaned continuum signal')
            plt.legend()
            plt.ylabel('Power Spectral Density')
            plt.xlabel('Frequency [Hz]')
            plt.savefig('figures/ps_data_cleaned_%02i.png' % (feed+1), bbox_inches='tight')

            n_tod = len(dg[feed])
            ps_dg = np.abs(np.fft.rfft(dg[feed])) ** 2 / n_tod
            f = np.abs(np.fft.rfftfreq(n_tod) * samprate)
            f = bin_data_maxbin(f, slope=slope, cut=cut, maxbin=4000)
            ps_dg = bin_data_maxbin(ps_dg, slope=slope, cut=cut, maxbin=4000)
            plt.figure()
            plt.title('Feed %i' % (feed+1))
            plt.loglog(f, ps_dg, label=r'$\delta g$')
            plt.loglog(f, PS_1f_nown(f, sigma0_prior[feed], fknee_prior[feed], alpha_prior[feed]), label='prior')
            plt.legend()
            plt.ylabel('Power Spectral Density')
            plt.xlabel('Frequency [Hz]')
            plt.savefig('figures/ps_gain_prior_%02i.png' % (feed+1), bbox_inches='tight')
            plt.show()
    return d_mean, d_clean_mean, ra, dec, dg
