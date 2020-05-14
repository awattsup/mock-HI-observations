import numpy as np 
import matplotlib.pyplot as plt 






def main():

	vlos = np.array([100,98,96,100,101,102,104,106,94,95,96,97])

	# spectra = calc_particle_spectrum(vlos,1000,5,14)
	# for ii in range(12):
	# 	plt.plot(np.arange(-0.5*1000,0.5*1000) + 0.5,spectra[:,ii])
	# plt.plot(np.arange(-0.5*1000,0.5*1000) + 0.5,np.sum(spectra,axis=1))
	# plt.show()

	dx_edges = (np.arange(512 + 1) - 0.5 * 512) 
	spacebins = np.arange(512) - 0.5 * 512 + 0.5
	print(dx_edges)
	print(spacebins)






def create_HI_datacube(coordinates, velocities, HI_masses, Vtherm = 7, params = None, filename = None):
	'''
		coordinates - line of sight coordinates
		Vlos - line of sight velocities 
		HI_masses - HI masses
		Vtherm - FWHM of Gaussian thermal velocity dispersion
	'''


	if params ==  None:							#		default datacube parameters
		params = {'dist':50,					#[Mpc] 	distance to source for flux conversion
				'Rmax':30,						#[kpc]	+/- spatial limits
				'dX':0.5,						#[kpc]	base spatial resolution
				'FWHM':2,						#[kpc]	beam FWHM for final spatial resolution
				'Vmax':400,						#[km/s]	+/- velocity limits
				'dV':5,							#[km/s] velocity resoltuion
				'rms':0}						#[mJy] 	input RMS noise

	Npix = (2.e0 * params['Rmax']) / params['dX']
	print('Npix in datacube,' params['Npix'])
	Nchan = (2.e0 * params['Vmax']) / params['dV']
	print('Nchan in datacube', params['Nchan'])

	coordinates = coordinates / params['dX'] 			#convert to pixel coordinates
	velocities = velocities / params['dV']			#convert to channel locations


	dX_edges = np.arange(Npix + 1) - 0.5 * Npix			#spatial bin edges in pixels
	dV_edges = np.arange(Nchan + 1) - 0.5 * Nchan		#spectrum bin edges in channels

	datacube = np.zeros([Npix, Npix, Nchan])	


	spectra = calc_particle_spectrum(velocities, Nchan, Vtherm)

	for yy in range(Npix):
		ylow = dX_edges[yy]
		yhigh = dX_edges[yy + 1]
		for xx in range(Npix):
			xlow = dX_edges[xx]
			xhigh = dX_edges[xx + 1]

			datacube[yy,xx,:] =  np.sum(spectra[:,
								(coordinates[:,0] >= xlow) & (coordinates[:,0] < xhigh) & 
								(coordinates[:,1] >= ylow) & (coordinates[:,1] < yhigh)],
								axis = 1)

	mjy_conv = mjy_conversion(params['dist'], params['dV'])
	datacube *= mjy_conversion

	datacube[:,:,:] += nprand.normal(np.zeros([Npix, Npix , Nchan]), params['rms'])				#add noise

	FWHM = params['FWHM'] / (2.355 * params['dX'])

	for vv in range(Nvel):
		datacube[:,:,vv]  = convolve(datacube[:,:,vv], Gaussian2DKernel(FHWM))

	pix_coords = (np.arange(Npix) - 0.5 * Npix + 0.5) * dX
	vel_mid = (np.arange(Nchan) - 0.5 * Nchan + 0.5) * dV


	# if filename == None:
	# 	return spacebins, velbins, datacube
	# else:
	# 	datacube = datacube.reshape((Npix , Npix * Nchan))
	# 	header = 'dist = {dist}\n cubephys = {cubephys}\n dx = {dx}\n Vlim = {Vlim}\n Vres = {Vres}\n'.format(
	# 		dist = params['dist'], cubephys = params['cubephys'], dx = dx, Vres = params['Vres'], Vlim = params['Vlim'])
	# 	np.savetxt(filename, datacube, header = header,fmt = "%.6e")










def gaussian_CDF(x,mu,sigma):
	cdf = 0.5e0 * (1.e0 + erf( (x - mu) / (np.sqrt(2.e0) * sigma) ))
	return cdf


def norm_gaussian(xx,mu,sigma):
	prob = 1.e0 / (sigma * np.sqrt(2.e0*np.pi)) * np.exp(-0.5e0*( ((xx - mu)*sigma) *((xx - mu) * sigma) ))
	return prob


def calc_particle_spectrum(Vlos,Nchan,Vtherm):
	Vtherm = np.min([Vtherm,14])							#maximum FWHM of 14 km/s
	Vtherm /= 2.355											#convert FHWM to sigma
	print(np.arange(-0.5*Nchan,0.5*Nchan) + 0.5)						
	if type(Vlos) == np.ndarray:
		if type(Vtherm) != np.ndarray:
			Vtherm = np.ones(len(Vlos)) * Vtherm
		spectrum = []
		for ii in range(len(Vlos)):
			spectrum.append(norm_gaussian(np.arange(Nchan) - (0.5 * Nchan) + 0.5, Vlos[ii], Vtherm[ii]))
		spectrum = np.array(spectrum).T
	else:
		spectrum = norm_gaussian(np.arange(Nchan) - (0.5 * Nchan) + 0.5, Vlos, Vtherm)

	return spectrum


def mjy_conversion(dist, Vres):
	conv = 1.e3 / (2.356e5  * (dist ** 2.e0) * Vres)
	return conv

if __name__ == '__main__':
	main()

