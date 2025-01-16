import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
import psycopg2
import argparse
import pickle
from astropy.io import fits
import pfs.datamodel as datamodel
from pfs.utils.fibers import spectrographFromFiberId
from pfs.utils.coordinates.CoordTransp import rotation
import pfs.utils.coordinates.DistortionCoefficients as DCoeff


class fluxCalQA:
    "A summary class to make flux calibration QA plots"
    
    #-------------------------------------------------------------
    def __init__(self, butler, verbose = False, isGen3 = True):
        self.butler = butler
        self.isGen3 = isGen3

        # there must be a better way to do this
        if isGen3:
            from lsst.daf.butler import Butler
            self.butlerRoot = '%s/%s' % (butler.get_datastore_roots()['FileDatastore@<butlerRoot>'], butler.collections[0])
            self.butlerCalibRoot = 'unknown'
        else:
            import lsst.daf.persistence as dafPersist
            self.butlerRoot = butler._repos._inputs[0].repoArgs.root
            self.butlerCalibRoot = butler._repos._inputs[0].repoArgs.mapperArgs['calibRoot']

        self.verbose = verbose
        self.marker = ['', 'x', 'o', 'd', '+']  # for spectrograph 1-4
        self.fiberOffsetThres = 0.05 # 50um to define NOTCONVERGED for data taken before Oct 2024
        self.fluxCalVectorColorWave = [470., 866.]  # wavelengths to compute fluxCalVector ratios 
        
        self.__version__ = '2.2.2'
        
        
    #-------------------------------------------------------------
    # get the OBJECT card from calexp as it is gone from all the other files
    def getObjectGen3(self, visit):
        for spectrograph in range(1,5):
            for arm in ['b', 'r', 'n', 'm']:
                try:
                    calexp = self.butler.get('calexp', dict(visit=visit, spectrograph=spectrograph, arm=arm))
                    #print(f'found calexp for visit={visit}, spectrograph={spectrograph}, arm={arm}')
                    return calexp.getInfo().getVisitInfo().object
                except:
                    pass
        return 'unknown'
    
    #-------------------------------------------------------------
    # get header info from fits - replace this method to use visitInfo - but isn't it doing the same thing?
    def getHeaderInfo(self, path, visit, keys):
        
        info = {}
        
        with fits.open(path) as hdul:
            header = hdul[0].header
            for key in keys:
                if self.isGen3:
                    if key == 'OBJECT':  # this card is missing from pfsMerged in gen3
                        info[key] = self.getObjectGen3(visit)
                        continue
                info[key] = header[key]
                
        return info
        
        
    #-------------------------------------------------------------
    # show observation information from opdb
    def drawObsInfo(self, ax, pfsConfig, visit):  
        
        # opdb query to get basic data - this set of data is actually in visitInfo
        with psycopg2.connect('postgresql://pfs@pfsa-db:5432/opdb') as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT pfs_visit_id, time_exp_start, exptime, altitude, azimuth, insrot, tel_ra, tel_dec, outside_humidity '
                           'FROM sps_exposure JOIN tel_status USING (pfs_visit_id) JOIN env_condition USING (pfs_visit_id) '
                           f'WHERE pfs_visit_id = {visit} LIMIT 1;')
            data = cursor.fetchall()[0]

        # panel parameters
        dy = 0.17
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        ax.axis('off')
        
        # get header info
        if self.isGen3:
            path = self.butler.getURI('pfsMerged', visit=visit).geturl().replace('file://','')
        else:
            # this is gen2
            path = self.butler.getUri('pfsMerged', visit=visit)
        info = self.getHeaderInfo(path, visit, ['OBJECT', 'PROP-ID', 'HIERARCH VERSION_DRP_STELLA', 'HIERARCH VERSION_OBS_PFS', 'HIERARCH VERSION_DATAMODEL', 'W_PFDSNM'])
        ax.text(0.0, 0.90, f'visit={visit}, object={info["OBJECT"]}, prop-id={info["PROP-ID"]}', fontsize=15)
        ax.text(0.99, 0.90, f'fluxCalQA ver. {self.__version__}', fontsize=8, ha='right')
        ax.text(0.0, 0.90 - dy * 1, 'pfsDesignId=%x, pfsDesignName=%s' % (pfsConfig.pfsDesignId, info['W_PFDSNM']), color='#777777')
        ax.text(0.0, 0.90 - dy * 2, 'Exp_start=%s, Exptime=%3.0fsec' % (data[1], data[2]), color='#777777')
        ax.text(0.0, 0.90 - dy * 3, 'Altitude=%2.2fdeg, Azimuth=%3.2fdeg, Insrot=%+2.2fdeg, PA=%+3.1fdeg' % (data[3], data[4], data[5], pfsConfig.posAng), color='#777777')
        ax.text(0.0, 0.90 - dy * 4, 'R.A.=%sdeg, Dec.=%sdeg, Humidity=%s%%' % (data[6], data[7], data[8]), color='#777777')
        ax.text(0.0, 0.90 - dy * 5, f'collection={self.butlerRoot.replace("file:///work/datastore/","")}, calibRoot={self.butlerCalibRoot}, arms={pfsConfig.arms}', color='#AAAAAA')
        ax.text(0.0, 0.90 - dy * 6, f'DRP_STELLA={info["HIERARCH VERSION_DRP_STELLA"]}, OBS_PFS={info["HIERARCH VERSION_OBS_PFS"]}, DATAMODEL={info["HIERARCH VERSION_DATAMODEL"]}', color='#AAAAAA')
        
        # store insrot for other methods
        self.insrot = float(data[5])
            
            
    #-------------------------------------------------------------
    # show AG seeing and transparency    
    def drawAG(self, ax, visit):
        
        # qadb query to get seeing and transparency
        with psycopg2.connect('postgresql://pfs@pfsa-db:5436/qadb') as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT seeing_agc_exposure.pfs_visit_id, agc_exposure_id, transparency_median, seeing_median '
                           'FROM seeing_agc_exposure JOIN transparency_agc_exposure USING (agc_exposure_id) '
                           f'WHERE seeing_agc_exposure.pfs_visit_id = {visit} ORDER BY agc_exposure_id;')
            data = cursor.fetchall()

        # parse the db data
        expId = np.full(len(data), np.nan)
        transparency = np.full(len(data), np.nan)
        seeing = np.full(len(data), np.nan)
        
        for i in range(len(data)):
            expId[i] = data[i][1]
            transparency[i] = data[i][2]
            seeing[i] = data[i][3]
        
        dx = (expId[-1] - expId[0]) * 0.05
        xmin = expId[0] - dx
        xmax = expId[-1] + dx

        # plot transparency
        ax.plot(expId, transparency, 'o', color='blue', markersize=0.5)
        ax.set_xlabel('AGC exposure ID')
        ax.set_ylabel('transparency', color='blue')
        ax.set_ylim(0.0, 1.1)
        ax.set_xlim(xmin, xmax)
        
        # plot seeing
        ax2 = ax.twinx()
        ax2.plot(expId, seeing, 'o', color='red', markersize=0.5)
        ax2.set_ylabel('seeing', color='red')
        ax2.set_ylim(0.3, 1.5)
        ax2.set_xlim(xmin, xmax)
        
        # summary stats in the title
        quartile1 = np.percentile(transparency, [25,50,75])
        quartile2 = np.percentile(seeing, [25,50,75])
        ax.text(xmin + (xmax-xmin)*0.25, 1.2, 'transparency = %1.2f +/- %1.2f' %
                     (quartile1[1], (quartile1[2]- quartile1[0])/1.349), color='blue', ha='center')
        ax.text(xmax - (xmax-xmin)*0.25, 1.2, 'seeing = %1.2f +/- %1.2f arcsec' %
                     (quartile2[1], (quartile2[2]- quartile2[0])/1.349), color='red', ha='center')
    
    
    #-------------------------------------------------------------
    # show pfsMerged ref-mag difference from reference
    def drawPfsMerged(self, ax, pfsConfig, visit, SNMinWave=840., SNMaxWave=875.):  #originally 400-550

        merged = self.butler.get('pfsMerged', visit=visit)
        mag = np.full(len(pfsConfig),np.nan)
        SN = np.full(len(pfsConfig),np.nan)
    
        midres = False
        if 'm' in pfsConfig.arms:
            midres = True
        

            
        # loop over the fibers
        for i in range(len(pfsConfig)):
            
            if pfsConfig.targetType[i] != datamodel.TargetType.FLUXSTD:  # this can be modified to include, e.g., SCIENCE
                continue
            if pfsConfig.fiberStatus[i] != datamodel.FiberStatus.GOOD:
                continue        
                
            # NOTCONVERGED flag is not set for data taken before Oct 2024
            offset = pfsConfig.pfiNominal[i] - pfsConfig.pfiCenter[i]
            distance = np.sqrt(offset[0]**2 + offset[1]**2)
            if distance > self.fiberOffsetThres:
                continue
                
            index = np.where(merged.fiberId == pfsConfig.fiberId[i])
           
            # use good pixels
            bad = merged.mask[index] & merged.flags.get('BAD', 'CR', 'SAT', 'NO_DATA') != 0
            good = ~bad
                       
            # compute synthetic mags and difference from reference mags
            mag[i] = self.synthmag.getMag(merged.wavelength[index][good] * 10, merged.flux[index][good] / merged.norm[index][good],
                                          filterName=self.refFilterFile, verbose=False, fnu=True, midres=midres) - (-2.5*np.log10(pfsConfig.psfFlux[i][self.psfFluxIndex]) + 31.4 )
            
            if np.isnan(mag[i]):
                mag[i] = -99. # just to differentiate from non-FLUXSTDs
   
            index2 = np.where((merged.wavelength[index] > SNMinWave) & (merged.wavelength[index] < SNMaxWave) & good)[0]
            SN[i] = np.nanmedian(merged.flux[index][index2] / np.sqrt(merged.covar[index][0][index2]))
        
        
        scatter = np.nanpercentile(mag, (25,50,75))
        ax.set_title('%s, median=%2.2f, scatter=%1.3fmag' % (self.synthmag.filters[self.psfFluxIndex], scatter[1], (scatter[2]-scatter[0])/1.35), fontsize=10)
        mag -= scatter[1] # subtract the median
        ax.text(pfsConfig.raBoresight, pfsConfig.decBoresight + 0.7, 'the global median is subtracted',horizontalalignment='center', fontsize=7)
        
        # sanity check
        if scatter[1] == np.nan:
            print('the i-band merged difference is nan')
            return None, None, None

        # plot
        for i in range(len(pfsConfig)):
            if pfsConfig.targetType[i] != datamodel.TargetType.FLUXSTD:  # this can be modified to include, e.g., SCIENCE
                ax.plot(pfsConfig.ra[i], pfsConfig.dec[i], '.', color='lightgray', markersize=0.4)
                continue
            
            if (np.isnan(mag[i]) == True):
                ax.plot(pfsConfig.ra[i],pfsConfig.dec[i], marker='.', color='green', markersize=1)
            else:
                if SN[i] < 10:
                    circle = patches.Circle((pfsConfig.ra[i],pfsConfig.dec[i]), 0.035, edgecolor='green', facecolor='none', linewidth=0.5)
                    ax.add_patch(circle)
                    print('debug', pfsConfig.fiberId[i], SN[i])
                spectrograph = spectrographFromFiberId(pfsConfig.fiberId[i])
                sc = ax.scatter(pfsConfig.ra[i], pfsConfig.dec[i], marker=self.marker[spectrograph], vmin=-0.5, vmax=0.5, s=10, c=mag[i], cmap=cm.coolwarm_r)
                
        plt.colorbar(sc, ax=ax)
        ax.set_xlabel('R.A. [deg]')
        ax.set_ylabel('Dec. [deg]')
    
        dra = 0.8 / np.cos(np.radians(pfsConfig.decBoresight))
        ddec = 0.8    
        ax.set_xlim(pfsConfig.raBoresight + dra, pfsConfig.raBoresight - dra)
        ax.set_ylim(pfsConfig.decBoresight - ddec, pfsConfig.decBoresight + ddec)  

        # direction of gravity
        # tel -> pfi: (insrot + 90)
        # pfi -> sky: (PA + 90)
        # direction of gravity
        length = 0.1
        angle = np.deg2rad(270. + (self.insrot + 90.) + (pfsConfig.posAng + 90.))  # 270 deg is because gravity is downwards
        ax.arrow(pfsConfig.raBoresight + dra * 0.75, pfsConfig.decBoresight - ddec * 0.75,
                 length * np.cos(angle), length * np.sin(angle), width=0.02)

        return SN, SNMinWave, SNMaxWave
    
    
        
    #-------------------------------------------------------------
    # show S/N distribution of pfsMerged FLUXSTDs
    def drawMergedSN(self, ax, SN, SNMinWave, SNMaxWave):
        ax.hist(SN, bins=20)
        ax.set_xlabel('SN at %3.1f - %3.1f nm' % (SNMinWave, SNMaxWave))
        ax.set_ylabel('N')
        scatter = np.nanpercentile(SN, (25,50,75))
        ax.set_title('median=%2.2f, scatter=%1.3f' % (scatter[1], (scatter[2]-scatter[0])/1.35), fontsize=10)
        
        
        
    #-------------------------------------------------------------
    # sohw reference magnitude distribution of FLUXSTDs
    def drawMagDistrib(self, ax, pfsConfig):

        pfsConfigSel = pfsConfig.select(targetType = datamodel.TargetType.FLUXSTD)
        mag = np.full(len(pfsConfigSel), np.nan)
 
        for i in range(len(pfsConfigSel)):
            mag[i] = -2.5*np.log10(pfsConfigSel.psfFlux[i][self.psfFluxIndex]) + 31.4
        
        ax.hist(mag, bins=20)
        ax.set_xlabel(f'{self.refCat} {self.refFilter} magnitude')
        ax.set_ylabel('N')
        scatter = np.nanpercentile(mag, (25,50,75))
        ax.set_title('median=%2.2f, scatter=%1.3f' % (scatter[1], (scatter[2]-scatter[0])/1.35), fontsize=10)
        
        

    #-------------------------------------------------------------
    # show normalization of the flux calibration vector over the focal plane
    def drawFluxCalNorm(self, ax, pfsConfig, fluxCal):

        wavelengths = np.full((len(pfsConfig),1), 600)  # 600nm, but it does not matter
        positions = pfsConfig.pfiCenter
    
        fluxNorm = fluxCal.evaluate(wavelengths=wavelengths, positions=positions, fiberIds=None).values.reshape(1,len(pfsConfig))[0]
    
        median = np.nanmedian(fluxNorm)
        fluxNorm /= median

        scatter = np.nanpercentile(fluxNorm, (25,50,75))
        pcm = ax.scatter(pfsConfig.ra, pfsConfig.dec, marker='o', vmin=0.5, vmax=1.5, s=3, c=fluxNorm, cmap=cm.coolwarm)
        plt.colorbar(pcm, ax=ax)   
    
        ax.set_xlabel('R.A. [deg]')
        ax.set_ylabel('Dec. [deg]')
        ax.set_title('median=%1.2e, fractional scatter=%1.2f%%' % (median, (scatter[2]-scatter[0])/1.35*100.), fontsize=10)
    
        dra = 0.8 / np.cos(np.radians(pfsConfig.decBoresight))
        ddec = 0.8
        ax.set_xlim(pfsConfig.raBoresight + dra, pfsConfig.raBoresight - dra)
        ax.set_ylim(pfsConfig.decBoresight - ddec, pfsConfig.decBoresight + ddec)  
        
        
    #-------------------------------------------------------------
    # show pfsSingle difference from reference catalog
    def drawSingle(self, ax, pfsConfig, visit):
        
        mag = np.full((6,len(pfsConfig)),np.nan)
    
        midres = False
        if 'm' in pfsConfig.arms:
            midres = True

        # gen3 outputs pfsCalibrated
        if self.isGen3:
            calibrated = self.butler.get('pfsCalibrated', visit=visit)
        
        # compute magnitude difference between PFS and reference
        for i in range(len(pfsConfig)):
                       
            # select good FLUXSTDs
            if pfsConfig.targetType[i] != datamodel.TargetType.FLUXSTD:
                continue
            if pfsConfig.fiberStatus[i] != datamodel.FiberStatus.GOOD:
                continue
                
            # NOTCONVERGED flag is not set for data taken before Oct 2024
            offset = pfsConfig.pfiNominal[i] - pfsConfig.pfiCenter[i]
            distance = np.sqrt(offset[0]**2 + offset[1]**2)
            if distance > self.fiberOffsetThres:
                continue
                
            if self.isGen3:
                for target in calibrated:
                    if target.objId == pfsConfig.objId[i] and target.catId == pfsConfig.catId[i]:
                        single = calibrated[target]
                        break
            else:
                try:
                    single = self.butler.get('pfsSingle', dict(visit=visit, objId=pfsConfig.objId[i], tract=pfsConfig.tract[i], patch=pfsConfig.patch[i], catId=pfsConfig.catId[i]))
                except:
                    continue
            
            ref = -2.5*np.log10(pfsConfig.psfFlux[i])+31.4
            if self.refCat == 'PS1':
                ref = np.append(ref, ref[4]+0.08) #y-nJ ~ 0.08 +/- 0.04
            else:
                ref = np.append(ref, np.nan) # otherwise nan
        
            # use only good pixels
            bad = single.mask & single.flags.get('BAD', 'CR', 'SAT', 'NO_DATA') != 0
            good = ~bad
                
            # compute synthetic mags and differences from reference mags
            for j, filter in enumerate(self.synthmag.filters):
                mag[j][i] = self.synthmag.getMag(single.wavelength[good] * 10, single.flux[good], filterName=filter, verbose=False, fnu=True, midres=midres, medianFilterFine=True) - ref[j]
                
        
        # plot over the focal plane
        for i in range(len(pfsConfig)):
            if pfsConfig.targetType[i] != datamodel.TargetType.FLUXSTD:
                ax.plot(pfsConfig.ra[i],pfsConfig.dec[i],'.',color='lightgray', markersize=0.4)
                continue
            if pfsConfig.fiberStatus[i] != datamodel.FiberStatus.GOOD:
                ax.plot(pfsConfig.ra[i],pfsConfig.dec[i],'.',color='green', markersize=0.4)
                continue
                            
            spectrograph = spectrographFromFiberId(pfsConfig.fiberId[i])
            if np.isnan(mag[self.psfFluxIndex][i]) == True:
                ax.plot(pfsConfig.ra[i],pfsConfig.dec[i], marker=self.marker[spectrograph], color='green', markersize=1)
            else:
                sc = ax.scatter(pfsConfig.ra[i], pfsConfig.dec[i], marker=self.marker[spectrograph], vmin=-0.1, vmax=0.1, s=10, c=mag[self.psfFluxIndex][i], cmap=cm.coolwarm_r)
        
        plt.colorbar(sc, ax=ax)
        ax.set_xlabel('R.A. [deg]')
        ax.set_ylabel('Dec. [deg]')
        ax.text(pfsConfig.raBoresight, pfsConfig.decBoresight + 0.7, 'mag difference in %s' % self.synthmag.filters[self.psfFluxIndex],
                horizontalalignment='center', fontsize=9)

        scatter = np.nanpercentile(mag[self.psfFluxIndex], (25,50,75))
        ax.set_title('median=%1.3fmag, scatter=%1.3fmag' % (scatter[1], (scatter[2]-scatter[0])/1.35), fontsize=10)
    
        dra = 0.8 / np.cos(np.radians(pfsConfig.decBoresight))
        ddec = 0.8
        ax.set_xlim(pfsConfig.raBoresight + dra, pfsConfig.raBoresight - dra)
        ax.set_ylim(pfsConfig.decBoresight - ddec, pfsConfig.decBoresight + ddec)  

        return mag
    

    #-------------------------------------------------------------
    # show pfsSingle color difference from reference
    def drawSingleColor(self, ax, pfsConfig, mag):
        
        for i in range(len(pfsConfig)):
            if pfsConfig.targetType[i] != datamodel.TargetType.FLUXSTD:
                ax.plot(pfsConfig.ra[i],pfsConfig.dec[i],'.',color='lightgray', markersize=0.4)
                continue
            if pfsConfig.fiberStatus[i] != datamodel.FiberStatus.GOOD:
                ax.plot(pfsConfig.ra[i],pfsConfig.dec[i],'.',color='green', markersize=0.4)
                continue
            
            spectrograph = spectrographFromFiberId(pfsConfig.fiberId[i])
            if np.isnan(mag[self.psfColorIndex[0]][i]) == True or np.isnan(mag[self.psfColorIndex[1]][i]) == True:
                ax.plot(pfsConfig.ra[i],pfsConfig.dec[i], marker=self.marker[spectrograph], color='green', markersize=1)
                #print('debug', mag[self.psfColorIndex[0]][i], mag[self.psfColorIndex[1]][i])
            else:
                sc = ax.scatter(pfsConfig.ra[i], pfsConfig.dec[i], marker=self.marker[spectrograph], vmin=-0.1, vmax=0.1, s=10, c=mag[self.psfColorIndex[0]][i] - mag[self.psfColorIndex[1]][i], cmap=cm.coolwarm)
    
        plt.colorbar(sc, ax=ax)
        ax.set_xlabel('R.A. [deg]')
        ax.set_ylabel('Dec. [deg]')
        ax.text(pfsConfig.raBoresight, pfsConfig.decBoresight + 0.7,
                '%s-%s color difference' % (self.synthmag.filters[self.psfColorIndex[0]], self.synthmag.filters[self.psfColorIndex[1]]),
                horizontalalignment='center', fontsize=9)

        color = mag[self.psfColorIndex[0]] - mag[self.psfColorIndex[1]]
        index = np.where(color != np.nan)[0]
        scatter = np.nanpercentile(color[index], (25,50,75))
        ax.set_title('median=%1.3fmag, scatter=%1.3fmag' % (scatter[1], (scatter[2]-scatter[0])/1.35), fontsize=10)
    
        dra = 0.8 / np.cos(np.radians(pfsConfig.decBoresight))
        ddec = 0.8
        ax.set_xlim(pfsConfig.raBoresight + dra, pfsConfig.raBoresight - dra)
        ax.set_ylim(pfsConfig.decBoresight - ddec, pfsConfig.decBoresight + ddec)  
        
        
    #-------------------------------------------------------------
    # show pfsSingle color difference from reference
    def drawCaption(self, ax, text):
        rect = patches.Rectangle(xy=(0, 0), width=1, height=1, color='lightgray', fill=True)
        ax.add_patch(rect)
        
        ax.text(0.5, 0.5, text, ha='center', va='center')
        ax.axis('off')

        
    #-------------------------------------------------------------
    # main function to draw QA plots
    def caption(self, saveFigDir = None):
        
        fig = plt.figure(figsize=(10, 20))
            
        ax = fig.add_subplot(8, 1, 1)
        self.drawCaption(ax, "Visit information")
        
        ax = fig.add_subplot(8, 1, 2)
        self.drawCaption(ax, "Seeing (red) and transparency (blue) during the exposure.")
        
        ax = fig.add_subplot(4, 3, 6)
        self.drawCaption(ax, "magnitude distribution of\nFLUXSTDs in the reference\nfilter.")
        
        ax = fig.add_subplot(4, 3, 5)
        self.drawCaption(ax, "S/N of FLUXSTDs from pfsMerged\nwithin the specified\nwavelength window.\n(default is 840-880nm).")
        
        ax = fig.add_subplot(4, 3, 4)
        self.drawCaption(ax, "magnitude difference between\npfsMerged (i.e., before fluxCal)\nand reference pohtometry.\nThe green points are\nstars with S/N<10.\nThe arrow at bottom-left\nis the direction of gravity.")
        
        ax = fig.add_subplot(4, 3, 7)
        self.drawCaption(ax, "The normalization of\nthe fluxCal vector over\nthe focal plane at 600nm.\nThis is the result of\nfitFluxCal.py.")
        
        ax = fig.add_subplot(4, 3, 10)
        self.drawCaption(ax, "magnitude difference between\npfsSingle and reference\npohtometry. The colored\npoints are FLUXSTDs, and\ngray points are the other\nobjects. The green points\nare FLUXSTDs with flux=NaN.")
        
        ax = fig.add_subplot(4, 3, 11)
        self.drawCaption(ax, "color difference between\npfsSingle and reference.\nThe colored points are\nFLUXSTDs, and gray points\nare the other objects.\nThe green points are FLUXSTDs\nwith flux=NaN.")
        
        ax = fig.add_subplot(4, 3, 8)
        self.drawCaption(ax, 'The flux calibration vector.\nThe gray shade shows\nthe 68% interval, and\nthe black line is\nthe median.')
        
        ax = fig.add_subplot(4, 3, 9)
        self.drawCaption(ax, 'The color correction\napplied to the flux\ncalibration vector.')
        
        ax = fig.add_subplot(4, 3, 12)
        self.drawCaption(ax, 'pfsSingle of FLUXSTD\nwith the highest S/N.\nThe red points are\nreference fluxes, and\nthe blue points are PFS fluxes.')
        
        plt.subplots_adjust(hspace = 0.40, wspace=0.4, top=0.8, bottom=0.2)
            
        if saveFigDir == None:
            fig.show()
        else:
            fig.savefig('%s/fluxCalQA_caption.png' % (saveFigDir), bbox_inches='tight')
            
        if self.verbose:
            print('all done')
                
        
        
    #-------------------------------------------------------------
    # main function to draw QA plots
    def setFilterIndices(self, pfsConfig):
        # the assumption is that the filter set of the first FLUXSTD is the same as all the rest
        filterNames = pfsConfig.select(targetType = datamodel.TargetType.FLUXSTD).filterNames[0]
        filterSets = ['ps1', 'gaia', 'hsc']
        filterNums = [0, 0, 0]

        # scan all the filter names and look for ps1, gaia, and hsc
        for i, filterSet in enumerate(filterSets):
            for filterName in filterNames:
                if filterSet in filterName:
                    filterNums[i] += 1

        self.psfFluxIndex = -1
        self.psfColorIndex = -1
        
        # choose the filter set to use
        if filterNums[0] == 5:          # this is ps1 - filters are in the order of grizy
            self.psfFluxIndex = 2       # refers to PS1 i-band
            self.psfColorIndex = [0, 3] # refers to PS1 g-bband and z
            self.refCat = 'PS1'
            self.refFilter = 'i-band'
            self.refFilterFile = 'i_ps1'
        if filterNums[1] == 3:          # this is gaia - filters are disordered and we have to sort it
            self.psfColorIndex = [0,0]
            for i, filterName in enumerate(filterNames):
                #print(filterName)
                if filterName == 'g_gaia':
                    self.psfFluxIndex = i         # refers to Gaia G-band
                if filterName == 'bp_gaia':
                    self.psfColorIndex[0] = i     # refers to Gaia Bp-band
                if filterName == 'rp_gaia':
                    self.psfColorIndex[1] = i     # refers to Gaia Rp-band
            self.refCat = 'Gaia'
            self.refFilter = 'G-band'
            self.refFilterFile = 'g_gaia_trunc' 

        if self.psfFluxIndex < 0:
            print('no approriate filter set is found for fluxCalQA. Skipping this visit.')
            return False

        #print(self.psfFluxIndex, self.psfColorIndex)
        
        # now, import synthmag class to compute synthetic fluxes
        sys.path.append('/work/tanaka/processing/synthmag/')
        import synthmag
        self.synthmag = synthmag.synthmag(filterSet = 'custom', filterNames = filterNames, fakeJ=True)
        
        return True

    
    #-------------------------------------------------------------
    # dummy function
    def drawTest(self, ax, text = None, *args):
        x = [0., 1., 2.]
        y = x
        ax.plot(x,y,'o')        
        ax.text(0.5, 0.5, text, ha='center', va='center')

    
    #-------------------------------------------------------------
    # show the flux calibration vector
    def drawFluxCalVector(self, ax, pfsConfig, fluxCal):
        waveGrid = np.arange(380., 1261., 1)
        wavelengths = np.zeros((len(pfsConfig),len(waveGrid)))
        for i in range(len(pfsConfig)):
            wavelengths[i] = waveGrid
        positions = pfsConfig.pfiCenter
        fluxNorm = fluxCal.evaluate(wavelengths=wavelengths, positions=positions, fiberIds=None).values
        percentile = np.log10(np.nanpercentile(fluxNorm, (16, 50, 84), axis=0))
        
        for i in range(len(waveGrid)):
            x = np.full(2, waveGrid[i])
            y = [percentile[0][i], percentile[2][i]]
            ax.plot(x, y, '-', color='lightgray', linewidth=0.5, alpha=1)
        ax.plot(waveGrid, percentile[1], '-', color='black', linewidth=0.5)

        ax.set_xlabel('wavelength [nm]')
        ax.set_ylabel('fluxCal vector')

    
    #-------------------------------------------------------------
    # show the color stretch applied to flux calibration vector
    def drawFluxCalVectorColor(self, ax, pfsConfig, fluxCal):
        wavelengths = np.zeros((len(pfsConfig),2))
        for i in range(len(pfsConfig)):
            wavelengths[i] = self.fluxCalVectorColorWave
        positions = pfsConfig.pfiCenter
        fluxNorm = fluxCal.evaluate(wavelengths=wavelengths, positions=positions, fiberIds=None).values

        # take ratio between the two wavelength points
        ratio = np.full(len(pfsConfig), np.nan)
        for i in range(len(pfsConfig)):
            ratio[i] = fluxNorm[i][0] / fluxNorm[i][1]
        median = np.nanmedian(ratio)
        ratio /= median

        pcm = plt.scatter(pfsConfig.ra, pfsConfig.dec, marker='o', vmin=0.8, vmax=1.2, s=3, c=ratio, cmap=cm.coolwarm)
        plt.colorbar(pcm, ax=ax)   
    
        scatter = np.nanpercentile(ratio, (25,50,75))
        ax.set_xlabel('R.A. [deg]')
        ax.set_ylabel('Dec. [deg]')
        ax.set_title('median=%1.2e, fractional scatter=%1.2f%%' % (median, (scatter[2]-scatter[0])/1.35*100.), fontsize=10)
        ax.text(pfsConfig.raBoresight, pfsConfig.decBoresight + 0.7,
                'ratio between %3.1fnm and %3.1fnm' % (self.fluxCalVectorColorWave[0], self.fluxCalVectorColorWave[1]),
                horizontalalignment='center', fontsize=9)
    
        dra = 0.8 / np.cos(np.radians(pfsConfig.decBoresight))
        ddec = 0.8
        ax.set_xlim(pfsConfig.raBoresight + dra, pfsConfig.raBoresight - dra)
        ax.set_ylim(pfsConfig.decBoresight - ddec, pfsConfig.decBoresight + ddec)  


            
    #-------------------------------------------------------------
    # show the color stretch applied to flux calibration vector
    def drawExampleSpectrum(self, ax, pfsConfig, visit, SN, mag):
        index = np.nanargmax(SN)  # object with highest S/N
        
        # gen3 outputs pfsCalibrated
        if self.isGen3 == False:
            print('this function is not supposed in gen2')
            return
            
        calibrated = self.butler.get('pfsCalibrated', visit=visit)

        for target in calibrated:
            if target.objId == pfsConfig.objId[index] and target.catId == pfsConfig.catId[index]:
                single = calibrated[target]
            
                bad = single.mask & single.flags.get('BAD', 'CR', 'SAT', 'NO_DATA') != 0
                good = ~bad
                median = np.nanmedian(single.flux[good])

                ax.plot(single.wavelength[good], single.flux[good] / median, '-', linewidth=0.5, alpha=0.5)
                ax.set_ylim(0, 2)
                ax.set_xlabel('wavelength [nm]')
                ax.set_ylabel('flux in arbitrary units')
                ax.set_title('fiberId = %d, SN=%2.1f' % (pfsConfig.fiberId[index], SN[index]))

                # need to update to accept Gaia and HSC filters
                if self.refCat == 'PS1':
                    wave = [479.65, 616.09, 750.62, 865.83, 961.15, 1150.]
                    refMags = -2.5*np.log10(pfsConfig.psfFlux[index]) + 31.4
                    gr = refMags[0] - refMags[1]
                    nJMag = refMags[4] - ((-0.172811)+(0.362018)*gr+(0.00223607)*gr*gr)
                    nJFlux = np.power(10, -0.4*(nJMag-31.4))
                    refFluxes = np.append(pfsConfig.psfFlux[index], nJFlux)
                    refMags = np.append(refMags, nJMag)

                ax.plot(wave,  refFluxes / median, 'o', color='red')

                pfsMags = np.full(len(wave), np.nan)
                for i in range(len(wave)):
                    pfsMags[i] = mag[i][index] + refMags[i]
                ax.plot(wave, np.power(10., -0.4*(pfsMags-31.4)) / median, 'o', color='blue')
                
                return
                
        
    #-------------------------------------------------------------
    # main function to draw QA plots
    def plot(self, visits, saveFigDir = None):
        
        for visit in visits:
            fig = plt.figure(figsize=(15, 20))
            
            # draw caption
            if visit < 0:
                self.caption(saveFigDir)
                return
        
            # get pfsConfig
            if self.verbose:
                print(f'reading pfsConfig...', end='')
            pfsConfig = self.butler.get('pfsConfig', visit=visit)

            # set the filter indices
            if self.setFilterIndices(pfsConfig) == False:
                continue

            
            # observing info and AG info in the top two panels
            if self.verbose:
                print(f'generating plot for visit = {visit}...', end='')
            ax = fig.add_subplot(8, 1, 1)
            self.drawObsInfo(ax, pfsConfig, visit)
            #self.drawTest(ax, text='obsInfo')
            ax = fig.add_subplot(8, 1, 2)
            self.drawAG(ax, visit)
            #self.drawTest(ax, text='AG')
            
            # pfsMerged panels next
            if self.verbose:
                print('computing merged fluxes...', end='')
            ax = fig.add_subplot(4, 3, 6)
            SN, SNMinWave, SNMaxWave = self.drawPfsMerged(ax, pfsConfig, visit) 
            #self.drawTest(ax, text='pfsMerged')
            ax = fig.add_subplot(4, 3, 5)
            self.drawMergedSN(ax, SN, SNMinWave, SNMaxWave)
            #self.drawTest(ax, text='mergedSN')
            
            
            # FLUXSTD mag distribution and fluxCal normalization
            ax = fig.add_subplot(4, 3, 4)
            self.drawMagDistrib(ax, pfsConfig)
            #self.drawTest(ax, text='mag distrib')
            if self.verbose:
                print('computing focal plane normalization...', end='')
            ax = fig.add_subplot(4, 3, 7)
            fluxCal = self.butler.get('fluxCal', visit=visit)
            self.drawFluxCalNorm(ax, pfsConfig, fluxCal)
            #self.drawTest(ax, text='fluxcal')
            
            # more about the flux calibration vector
            ax = fig.add_subplot(4, 3, 8)
            self.drawFluxCalVector(ax, pfsConfig, fluxCal)
            #self.drawTest(ax, text='fluxcal')
            ax = fig.add_subplot(4, 3, 9)
            self.drawFluxCalVectorColor(ax, pfsConfig, fluxCal)
            #self.drawTest(ax, text='fluxcalcolor')
            
            # mag and color difference from pfsSingle
            if self.verbose:
                print('computing single fluxes...', end='')
            ax = fig.add_subplot(4, 3, 10)
            mag = self.drawSingle(ax, pfsConfig, visit)
            #self.drawTest(ax, text='single')
            ax = fig.add_subplot(4, 3, 11)
            self.drawSingleColor(ax, pfsConfig, mag)
            #self.drawTest(ax, text='singleColor')
            
            # example FLUXSTD spectrum
            if self.verbose:
                print('plotting a sample FLUXSTD spectrum...', end='')
            ax = fig.add_subplot(4, 3, 12)
            #with open('tmp1.pickle','wb') as file:
            #    pickle.dump(SN, file)
            #with open('tmp2.pickle','wb') as file:
            #    pickle.dump(mag, file)
            #with open('tmp1.pickle','rb') as file:
            #    SN = pickle.load(file)
            #with open('tmp2.pickle','rb') as file:
            #    mag = pickle.load(file)
                
            self.drawExampleSpectrum(ax, pfsConfig, visit, SN, mag)

            
            plt.subplots_adjust(hspace = 0.40, wspace=0.4, top=0.8, bottom=0.2)
            #fig.tight_layout()
            
            if saveFigDir == None:
                fig.show()
            else:
                fig.savefig('%s/fluxCalQA_v%s.png' % (saveFigDir, visit), bbox_inches='tight')
                #plt.close()
            
            if self.verbose:
                print('all done')
            
            
    
#-------------------------------------------------------------
def main():
    parser=argparse.ArgumentParser()
    # gen2
    #parser.add_argument('rerun')
    #parser.add_argument('calibRoot')
    #parser.add_argument('--repoDir', default = '/work/drp')
    parser.add_argument('collections')
    parser.add_argument('visit')
    parser.add_argument('--datastore', default = '/work/datastore')
    parser.add_argument('--saveFigDir', default = '/home/tanaka/')
    args=parser.parse_args()

    # gen2
    #rerun = os.path.join(args.repoDir, 'rerun', args.rerun)
    #butler = dafPersist.Butler(rerun, calibRoot=args.calibRoot)

    # gen3
    from lsst.daf.butler import Butler
    butler = Butler(args.datastore, collections=args.collections)
    
    fluxcalqa = fluxCalQA(butler, verbose = True)
    fluxcalqa.plot([int(args.visit)], saveFigDir = args.saveFigDir)
    
        
        
#-------------------------------------------------------------
if __name__ == "__main__":
    main()
    