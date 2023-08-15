import numpy as np
import scipy as sp

class bmObject:
    """Class object that handles breathing data

    Parameters
    ----------
    resp : Your reathing signal in the form of a list
    Must be a 1xN vector
    
    srate : sampling rate of the signal in Hz (int or float)
    sampling rates lower than 20 Hz or higher than 5000 hz are not supported.

    dataType : string that defines what kind of data the resp signal represents.
    Must be set to 'humanAirflow', 'rodentAirflow', 'humanBB', or 'rodentThermocouple'.
    
    Examples
    --------
    bm = bmObject(myResp,1000, 'humanAirflow')
    bm.estimateAllFeatures()
    """
    
    def __init__(self, resp, srate, dataType):
        self.resp = resp
        self.srate = srate
        self.dataType = dataType
        self.featureEstimationsComplete=0
        self.featuresManuallyEdited=0
        self.smoothedRespiration = []
        self.time = []
        self.baselineCorrectedRespiration = []
        
        self.inhalePeaks = []
        self.exhaleTroughs = []
        self.peakInspiratoryFlows = []
        self.troughExpiratoryFlows = []
        
        self.inhaleOnsets = []
        self.exhaleOnsets = []
        
        self.inhalePauseOnsets = []
        self.exhalePauseOnsets = []
        
        self.inhaleOffsets = []
        self.exhaleOffsets = []
        
        self.inhaleDurations = []
        self.exhaleDurations = []
        
        self.inhaleVolumes = []
        self.exhaleVolumes = []
        
        self.secondaryFeatures = []
        
        self.statuses = []
        
        # mean smoothing window is different for rodents and humans
        if dataType =='humanAirflow':
            self.smoothWinsize = 50

        elif dataType == 'humanBB':
            self.smoothWinsize = 50

        elif dataType == 'rodentAirflow':
            self.smoothWinsize = 10

        elif dataType == 'rodentThermocouple':
            self.smoothWinsize = 10
        else:
            print('todo error msg')
    
    

    
    # TODO
    #function validParams = checkClassInputs(Bm)


    # Preprocessing Methods
        
    def smoothResp(self):
        '''de-noises data using a sliding window
        '''
        srateCorrectedSmoothedWindow = int(np.floor((self.srate/1000) * self.smoothWinsize))
        self.smoothedRespiration = self.running_mean(self.resp, srateCorrectedSmoothedWindow);
            
        # time is like fieldtrip. All events are indexed by point in
        # time vector.
        self.time = np.arange(0,len(self.resp)) / self.srate
            

    def correctRespirationToBaseline(self, method='simple', zScore=0, verbose=0):
        """Corrects breathing to baseline using one of two methods

        Parameters
        ----------
        method : 'simple' subtracts the average of the breathing signal
                 'sliding' subtracts the sliding average in a minute-long window 

        zScore : 0 or 1, if zScore is set to 1, breathing signal is z-scored 
        
        """
        
        # default window size for sliding window mean is 60 seconds.
        swSize = 60; 
        
        detrendedResp=sp.signal.detrend(self.smoothedRespiration)
        respMean = self.smoothedRespiration - np.mean(self.smoothedRespiration)
    
        if method == 'simple':
            self.baselineCorrectedRespiration = respMean
            
        elif method == 'sliding':
            #for periodic drifts in data
            srateCorrectedSW = int(np.floor(self.srate * swSize));
        
            respSlidingMean=self.running_mean(self.smoothedRespiration, srateCorrectedSW);
                
            # subtract sliding mean from respiratory trace
            self.baselineCorrectedRespiration = self.smoothedRespiration - respSlidingMean
            
        if zScore == 1:
            self.baselineCorrectedRespiration = (self.baselineCorrectedRespiration - np.mean(self.baselineCorrectedRespiration)) / np.std(self.baselineCorrectedRespiration)
            
            
    def labelExtrema(self,simplify=0,verbose=0,customDecisionThreshold=0,swSizes=[]):
        """calls findExtrema to find and label peaks and troughs in a breathing signal

        Parameters
        ----------
        simplify : Sets nInhales to nExhales to simplify other computations. 
        Extra exhales at the end of the signal are removed.

        customDecisionThreshold : integer, set to adjust extrema identification threshold
        sampling rates lower than 20 Hz or higher than 5000 hz are not supported.

        swSizes : list of integers, custom windows may be used to change extrema identification sensitivity

        """
        
        # calls find extrema
        
        [putativePeaks, putativeTroughs] = self.findExtrema( customDecisionThreshold=customDecisionThreshold, swSizes=swSizes )
        
        # set nPeaks to nTroughs
        if simplify == 1:
            putativePeaks=putativePeaks[:len(putativeTroughs)]
            
        # The extrema represent the peak flow rates of inhales and exhales only in airflow recordings.
        if self.dataType == 'humanAirflow' or self.dataType == 'rodentAirflow':
            #indices of extrema
            self.inhalePeaks = putativePeaks
            self.exhaleTroughs = putativeTroughs

            # values of extrema
            self.peakInspiratoryFlows = self.baselineCorrectedRespiration[putativePeaks]
            self.troughExpiratoryFlows = self.baselineCorrectedRespiration[putativeTroughs]
            
            # In human breathing belt recordings, the peaks and troughs 
            # represent exhale and inhale onsets, respectively, because the
            # point where volume has maximized and instantaneously
            # decreases demarcates an exhale. This is unlike zero-crosses 
            # which demarcate breath onsets in airflow recordings.
            if self.dataType =='humanBB':
                
                self.inhaleOnsets = putativeTroughs;
                self.exhaleOnsets = putativePeaks;

            
            # In rodent thermocouple recordings, the peaks and troughs 
            # represent inhale and exhale onsets, respectively. Inhales
            # decrease the temperature in the nose and the onset is
            # demarcated by the first point this happens - the inflection
            # point of the peaks. Visa versa for exhales.
            elif self.dataType == 'rodentThermocouple':
                self.inhaleOnsets = putativePeaks;
                self.exhaleOnsets = putativeTroughs;
                
            
    # FEATURE EXTRACTION METHODS
    def findExtrema(self,customDecisionThreshold=0,swSizes=[]):
        '''Called by labelExtrema, identifies peaks and troughs in respiratory data
        
        Parameters
        ----------
        customDecisionThreshold : integer, set to adjust extrema identification threshold
        sampling rates lower than 20 Hz or higher than 5000 hz are not supported.

        swSizes : list of integers, custom windows may be used to change extrema identification sensitivity
        '''
        
        resp=self.baselineCorrectedRespiration
        srateAdjust = self.srate/1000
        
        if len(swSizes)==0:
            if self.dataType == 'humanAirflow' or self.dataType == 'humanBB':
                srateAdjust = self.srate/1000
                swSizes = [int(np.floor(100*srateAdjust)),int(np.floor(300*srateAdjust)), int(np.floor(700*srateAdjust)), int(np.floor(1000*srateAdjust)), int(np.floor(5000*srateAdjust))]
                
            elif self.dataType == 'rodentAirflow' or self.dataType == 'rodentThermocouple':
                srateAdjust = self.srate/1000;
                swSizes = [int(np.floor(5*srateAdjust)), int(np.floor(10 * srateAdjust)), int(np.floor(20 * srateAdjust)), int(np.floor(50 * srateAdjust))];
        

        # pad end with zeros to include tail of data otherwise missed by big windows 
        padInd = min([len(resp)-1,max(swSizes)*2])
        flipInds=[len(resp)-padInd,len(resp)]
        lrFlipped = resp[flipInds[1]:flipInds[0]:-1]
        paddedResp = np.append(resp,lrFlipped)

        # initializing vector of all points where there is a peak or trough.
        swPeakVect = np.zeros([1, len(paddedResp)])
        swTroughVect = np.zeros([1, len(paddedResp)])

        # peaks and troughs must exceed this value. Sometimes algorithm finds mini peaks in flat traces
        peakThreshold = np.mean(resp) + np.std(resp) / 2
        troughThreshold = np.mean(resp) - np.std(resp) / 2

        # shifting window to be unbiased by starting point
        SHIFTS = np.arange(3)
        nWindows=len(swSizes)*len(SHIFTS)


        # find maxes in each sliding window, in each shift, and return peaks that are agreed upon by majority windows.

        # find extrema in each window of the data using each window size and offset
        for win in range(len(swSizes)):

            sw = swSizes[win];
            # cut off end of data based on sw size
            nIters  = int(np.floor(len(paddedResp) / sw)-1); 

            for shift in SHIFTS:
                # store index of max and min airflow in each window
                argmaxVect = np.zeros([1, nIters]);
                argminVect = np.zeros([1, nIters]);

                # shift starting point of sliding window to get unbiased maxes
                windowInit = int((sw - np.floor(sw / (shift+ 1))));

                # iterate over this window and find all maxima and minima
                windowIter=windowInit

                for i in range(nIters):
                    thisWindow = paddedResp[windowIter:windowIter + sw - 1]
                    maxVal = np.max(thisWindow)
                    maxInd = np.argmax(thisWindow)

                    # make sure peaks and troughs are real.
                    if maxVal > peakThreshold:
                        # index in window + location of window in original resp time
                        argmaxVect[0,i]= windowIter + maxInd-1

                    minVal = np.min(thisWindow)
                    minInd = np.argmin(thisWindow)
                    if minVal < troughThreshold:
                        # index in window + location of window in original resp time
                        argminVect[0,i]= windowIter+minInd-1

                    windowIter = windowIter + sw

                # add 1 to consensus vector at time index where peak was found by this window
                for i in range(len(argmaxVect[0])):
                    if argmaxVect[0,i] > 0:
                        timeIndex=int(argmaxVect[0,i])
                        swPeakVect[0,timeIndex] = swPeakVect[0,timeIndex]+1

                for i in range(len(argminVect[0])):
                    if argminVect[0,i] > 0:
                        timeIndex=int(argminVect[0,i])
                        swTroughVect[0,timeIndex] = swTroughVect[0,timeIndex]+1


        # find threshold that makes minimal difference in number of extrema found similar to idea of knee method of k-means clustering

        nPeaksFound = np.zeros([1, nWindows])
        nTroughsFound = np.zeros([1, nWindows])
        for threshold_ind in range(nWindows):
            nPeaksFound[0, threshold_ind] = int(np.sum(swPeakVect > threshold_ind))
            nTroughsFound[0, threshold_ind] = int(np.sum(swTroughVect > threshold_ind))

        bestPeakDiff = np.argmax(np.diff(nPeaksFound))
        bestTroughDiff = np.argmax(np.diff(nTroughsFound))

        if customDecisionThreshold > 0:
            bestDecisionThreshold = customDecisionThreshold
        else:
            bestDecisionThreshold = int(np.floor(np.mean([bestPeakDiff, bestTroughDiff])))


        # temporary peak inds. Each point where there is a real peak or trough
        peakInds = [i for i in range(swPeakVect.shape[1]) if swPeakVect[0,i] > bestDecisionThreshold]
        troughInds = [i for i in range(swTroughVect.shape[1]) if swTroughVect[0,i] > bestDecisionThreshold]
        
        # sometimes there are multiple peaks or troughs in series which shouldn't be possible. This loop ensures the series alternates peaks and troughs.

        # first we must find the first peak
        offByN = True;
        tri = 1;
        while offByN:
            if peakInds[tri] > troughInds[tri]:
                # remove this trough
                troughInds = troughInds[tri + 1:len(troughInds)]
            else:
                offByN=False;

        correctedPeaks = []
        correctedTroughs = []


        pki=1 # peak ind
        tri=1 # trough ind

        # variable to decide whether to record peak and trough inds.
        proceedCheck = 1

        # find peaks and troughs that alternate
        while pki <len(peakInds)-1 and tri < len(troughInds)-1:

            # time difference between peak and next trough
            peakTroughDiff = troughInds[tri] - peakInds[pki]

            # check if two peaks in a row
            peakPeakDiff = peakInds[pki+1] - peakInds[pki]

            if peakPeakDiff < peakTroughDiff:
                # if two peaks in a row, take larger peak
                nxtPk = np.argmax([paddedResp[peakInds[pki]], paddedResp[peakInds[pki+1]]])
                if nxtPk == 2:
                    # forget this peak. keep next one.
                    pki = pki+1
                else:
                    # forget next peak. keep this one.
                    peakInds = np.setdiff1d(peakInds, peakInds[pki+1])

                # there still might be another peak to remove so go back and check
                # again
                proceedCheck=0
            #end

            # if the next extrema is a trough, check for trough series
            if proceedCheck == 1:

                # check if trough is after this trough.
                troughTroughDiff = troughInds[tri + 1] - troughInds[tri]
                troughPeakDiff = peakInds[pki + 1] - troughInds[tri]

                if troughTroughDiff < troughPeakDiff:
                    # if two troughs in a row, take larger trough
                    nxtTr = np.argmin([paddedResp[troughInds[tri]], paddedResp[troughInds[tri + 1]]])
                    if nxtTr == 2:
                        # take second trough
                        tri = tri + 1
                    else:
                        # remove second trough
                        troughInds = np.setdiff1d(troughInds, troughInds[tri+1])

                    # there still might be another trough to remove so go back and 
                    # check again
                    proceedCheck=0

            # if both of the above pass we can save values
            if proceedCheck == 1:
                # if peaks aren't ahead of troughs
                if peakTroughDiff > 0:
                    #time_diff_pt = [time_diff_pt peak_trough_diff*srate_adjust];
                    correctedPeaks.append(peakInds[pki])
                    correctedTroughs.append(troughInds[tri])

                    # step forward
                    tri=tri+1;
                    pki=pki+1;
                else:
        #             # peaks got ahead of troughs. This shouldn't ever happen.
        #             disp('Peaks got ahead of troughs. This shouldnt happen.');
        #             disp(strcat('Peak ind: ', num2str(peakInds(pki))));
        #             disp(strcat('Trough ind: ', num2str(troughInds(tri))));
                    raise('unexpected error. stopping');

            proceedCheck=1;

        # remove any peaks or troughs in padding
        correctedPeaks = [i for i in correctedPeaks if i < len(resp)]
        correctedTroughs = [i for i in correctedTroughs if i < len(resp)]
        
        return correctedPeaks, correctedTroughs
        
        
    def findRespiratoryPausesAndOnsets(self,nBins=100):
        '''finds each breath onset and respiratory pause in the data, given the peaks and troughs
        Parameters
        ----------
        nBins : number of bins to use to identify noise threshold of respiratory pauses
        
        Steps: 
        1. Find breath onsets for the first and last breath.
        2. For each trough-to-peak window, check if there is a respiratory pause.
        3. If no pause, then just find when the trace crosses baseline.
        4. if there is a pause, the amplitude range that it pauses over.
        5. Find onset and offset of pause.
        6. Find exhale onset in peak-to-trough window where it crosses baseline.
        7. Repeat.

        100 bins works well for data with sampling rates>100 Hz. 
        Use different nBINS for data with lower sampling rates sampling rate is too slow.
        '''
        
        if nBins >= 100:
            maxPauseBins=5
        else:
            maxPauseBins=2

        resp=self.baselineCorrectedRespiration
        peaks=self.inhalePeaks
        troughs=self.exhaleTroughs

        # free parameter for sensitivity of calling something a pause or not.
        MAXIMUM_BIN_THRESHOLD = 5;

        inhaleOnsets = np.zeros([1, len(peaks)])
        exhaleOnsets = np.zeros([1, len(troughs)])

        # inhale pause onsets happen after inhales but before exhales
        # exhale pause onsets happen after exhales but before inhales 
        inhalePauseOnsets = np.zeros([1,len(peaks)])
        exhalePauseOnsets = np.zeros([1,len(troughs)]) 
        inhalePauseOnsets[:]=np.nan
        exhalePauseOnsets[:]=np.nan

        # thresholds the mode bin (cycle zero cross) must fall between.
        # fast breathing sometimes looks like a square wave so the mode is at the
        # head or the tail and doesn't reflect a breath onset.
        UPPER_THRESHOLD = np.round(nBins*.7)
        LOWER_THRESHOLD = np.round(nBins*.3)

        # If this fails, use average of trace
        SIMPLE_ZERO_CROSS = np.mean(resp)

        # first onset is special because it's not between two peaks. If there is 
        # lots of time before the first peak, limit it to within 4000 samples.

        # head and tail onsets are hard to estimate without lims. use average
        # breathing interval in these cases.
        TAIL_ONSET_LIMS = int(np.floor(np.mean(np.diff(peaks))))

        if peaks[0] > TAIL_ONSET_LIMS:
            firstZeroCrossBoundary = peaks[0]-TAIL_ONSET_LIMS
        else:
            firstZeroCrossBoundary=0

        # Estimate last zero cross (inhale onset) before first peak
        # If there is not a full cycle before the first peak, this estimation will be unreliable.
        THIS_WINDOW = resp[firstZeroCrossBoundary:peaks[0]]

        CUSTOM_BINS = np.linspace(min(THIS_WINDOW),max(THIS_WINDOW), nBins)
        [AMPLITUDE_VALUES, WINDOW_BINS] = np.histogram(THIS_WINDOW, bins=CUSTOM_BINS)
        MODE_BIN = np.argmax(AMPLITUDE_VALUES)
        ZERO_CROSS_THRESHOLD = WINDOW_BINS[MODE_BIN]

        if MODE_BIN < LOWER_THRESHOLD or MODE_BIN > UPPER_THRESHOLD:
            ZERO_CROSS_THRESHOLD = SIMPLE_ZERO_CROSS


        POSSIBLE_INHALE_INDS = THIS_WINDOW < ZERO_CROSS_THRESHOLD

        if sum(POSSIBLE_INHALE_INDS) > 0:
            # get the last index where this is true
            possibleInhaleOnsets=np.where(POSSIBLE_INHALE_INDS==True)
            INHALE_ONSET = possibleInhaleOnsets[0][-1]
            inhaleOnsets[0,0] = firstZeroCrossBoundary + INHALE_ONSET
        else:
            inhaleOnsets[0,0] = firstZeroCrossBoundary


        # percent of amplitudes added by including a bin compared to max bin
        BINNING_THRESHOLD = .25

        # cycle through each peak-peak window of respiration data
        for THIS_BREATH in range(len(peaks)-1):

            # Find next inhale onset and pause

            INHALE_WINDOW = resp[troughs[THIS_BREATH]:peaks[THIS_BREATH+1]]
            CUSTOM_BINS = np.linspace(min(INHALE_WINDOW),max(INHALE_WINDOW),nBins)
            [AMPLITUDE_VALUES, WINDOW_BINS] = np.histogram(INHALE_WINDOW, CUSTOM_BINS)
            MODE_BIN = np.argmax(AMPLITUDE_VALUES)

            MAX_BIN_RATIO = AMPLITUDE_VALUES[MODE_BIN]/np.mean(AMPLITUDE_VALUES);

            isExhalePause = ~(MODE_BIN < LOWER_THRESHOLD or MODE_BIN > UPPER_THRESHOLD or MAX_BIN_RATIO < MAXIMUM_BIN_THRESHOLD)

            if ~isExhalePause:
                # data does not cluster in the middle, indicating no respiratory pause. So just use baseline crossing as inhale onset
                THIS_INHALE_THRESHOLD = SIMPLE_ZERO_CROSS;

                # points where the trace crosses zero
                POSSIBLE_INHALE_INDS = INHALE_WINDOW > THIS_INHALE_THRESHOLD

                # find last point in half-cycle below threshold
                possibleInhaleOnsets=np.where(POSSIBLE_INHALE_INDS==True)
                INHALE_ONSET = possibleInhaleOnsets[0][0]

                # this is the inhale onset for the next peak so add 1 to keep indexing consistant
                exhalePauseOnsets[0, THIS_BREATH] = np.nan
                inhaleOnsets[0, THIS_BREATH + 1] = troughs[THIS_BREATH] + INHALE_ONSET

            else:
                # add bins to find good range of variance for this respiratory pause
                MIN_PAUSE_RANGE = WINDOW_BINS[MODE_BIN]
                MAX_PAUSE_RANGE = WINDOW_BINS[MODE_BIN+1]
                MAX_BIN_TOTAL = AMPLITUDE_VALUES[MODE_BIN]

                # add bins in positive direction
                for ADDITIONAL_BIN in range(maxPauseBins):
                    THIS_BIN = MODE_BIN-ADDITIONAL_BIN
                    nVALS_ADDED = AMPLITUDE_VALUES[THIS_BIN]
                    if nVALS_ADDED > MAX_BIN_TOTAL * BINNING_THRESHOLD:
                        MIN_PAUSE_RANGE = WINDOW_BINS[THIS_BIN]


                # add bins in negative direction
                for ADDITIONAL_BIN in range(maxPauseBins):
                    THIS_BIN = MODE_BIN+ADDITIONAL_BIN;
                    nVALS_ADDED = AMPLITUDE_VALUES[THIS_BIN]
                    if nVALS_ADDED > MAX_BIN_TOTAL * BINNING_THRESHOLD:
                        MAX_PAUSE_RANGE = WINDOW_BINS[THIS_BIN]


                PUTATIVE_PAUSE_INDS = np.intersect1d(np.where(INHALE_WINDOW > MIN_PAUSE_RANGE), np.where(INHALE_WINDOW < MAX_PAUSE_RANGE))

                if len(PUTATIVE_PAUSE_INDS)==0:
#                     print('no exhale pause found')

                    # actually no pause
                    THIS_INHALE_THRESHOLD = SIMPLE_ZERO_CROSS;
                    POSSIBLE_INHALE_INDS = INHALE_WINDOW > THIS_INHALE_THRESHOLD
                    possibleInhaleOnsets=np.where(POSSIBLE_INHALE_INDS==True)
                    INHALE_ONSET = possibleInhaleOnsets[0][0]

                    exhalePauseOnsets[0, THIS_BREATH] = np.nan
                    inhaleOnsets[0, THIS_BREATH + 1] = troughs[THIS_BREATH] + INHALE_ONSET

                else:
                    PAUSE_ONSET = PUTATIVE_PAUSE_INDS[0]-1
                    INHALE_ONSET = PUTATIVE_PAUSE_INDS[-1]+1
                    exhalePauseOnsets[0, THIS_BREATH] = troughs[THIS_BREATH] + PAUSE_ONSET
                    inhaleOnsets[0, THIS_BREATH + 1] = troughs[THIS_BREATH] + INHALE_ONSET


            # Find Next Exhale

            # troughs always follow peaks
            EXHALE_WINDOW = resp[peaks[THIS_BREATH]:troughs[THIS_BREATH]]
            CUSTOM_BINS=np.linspace(min(EXHALE_WINDOW),max(EXHALE_WINDOW),nBins)
            [AMPLITUDE_VALUES, WINDOW_BINS] = np.histogram(EXHALE_WINDOW, CUSTOM_BINS)
            MODE_BIN = np.argmax(AMPLITUDE_VALUES)
            MAX_BIN_RATIO = AMPLITUDE_VALUES[MODE_BIN]/np.mean(AMPLITUDE_VALUES)

            isInhalePause = ~(MODE_BIN < LOWER_THRESHOLD or MODE_BIN > UPPER_THRESHOLD or MAX_BIN_RATIO < MAXIMUM_BIN_THRESHOLD)

            if ~isInhalePause:
                # data does not cluster in the middle, indicating no respiratory
                # pause. So just use baseline crossing as inhale onset
                THIS_EXHALE_THRESHOLD = SIMPLE_ZERO_CROSS

                #points where the trace crosses zero
                POSSIBLE_EXHALE_INDS = np.where(EXHALE_WINDOW > THIS_EXHALE_THRESHOLD)

                # find last point in half-cycle below threshold
                EXHALE_ONSET = POSSIBLE_EXHALE_INDS[0][-1]
                # this is the exhale onset for the next trough

                inhalePauseOnsets[0, THIS_BREATH] = np.nan
                exhaleOnsets[0, THIS_BREATH] = peaks[THIS_BREATH] + EXHALE_ONSET

            else:
                # add bins to find good range of variance for this respiratory pause
                MIN_PAUSE_RANGE = WINDOW_BINS[MODE_BIN]
                MAX_PAUSE_RANGE = WINDOW_BINS[MODE_BIN+1]
                MAX_BIN_TOTAL = AMPLITUDE_VALUES[MODE_BIN]


                # add bins in positive direction
                for ADDITIONAL_BIN in range(maxPauseBins):
                    THIS_BIN = MODE_BIN-ADDITIONAL_BIN
                    nVALS_ADDED = AMPLITUDE_VALUES[THIS_BIN]
                    if nVALS_ADDED > MAX_BIN_TOTAL * BINNING_THRESHOLD:
                        MIN_PAUSE_RANGE = WINDOW_BINS[THIS_BIN]

                # add bins in negative direction
                for ADDITIONAL_BIN in range(maxPauseBins):
                    THIS_BIN = MODE_BIN+ADDITIONAL_BIN;
                    nVALS_ADDED = AMPLITUDE_VALUES[THIS_BIN]
                    if nVALS_ADDED > MAX_BIN_TOTAL * BINNING_THRESHOLD:
                        MAX_PAUSE_RANGE = WINDOW_BINS[THIS_BIN]


                PUTATIVE_PAUSE_INDS = np.intersect1d(np.where(EXHALE_WINDOW > MIN_PAUSE_RANGE), np.where(EXHALE_WINDOW < MAX_PAUSE_RANGE))
                if len(PUTATIVE_PAUSE_INDS)==0:
#                     print('no inhale pause found')
                    THIS_EXHALE_THRESHOLD = SIMPLE_ZERO_CROSS
                    POSSIBLE_EXHALE_INDS = np.where(EXHALE_WINDOW > THIS_EXHALE_THRESHOLD)

                    EXHALE_ONSET = POSSIBLE_EXHALE_INDS[0][-1]

                    inhalePauseOnsets[0, THIS_BREATH] = np.nan
                    exhaleOnsets[0, THIS_BREATH] = peaks[THIS_BREATH] + EXHALE_ONSET
                else:
                    PAUSE_ONSET = PUTATIVE_PAUSE_INDS[0]-1
                    EXHALE_ONSET = PUTATIVE_PAUSE_INDS[-1]+1

                    inhalePauseOnsets[0, THIS_BREATH] = peaks[THIS_BREATH] + PAUSE_ONSET
                    exhaleOnsets[0, THIS_BREATH] = peaks[THIS_BREATH] + EXHALE_ONSET


        # last exhale onset is also special because it's not in a peak-peak cycle
        # treat it similar to first inhale
        if len(resp) - peaks[-1] > TAIL_ONSET_LIMS:
            LAST_ZERO_CROSS_BOUNDARY = peaks[-1] + TAIL_ONSET_LIMS
        else:
            LAST_ZERO_CROSS_BOUNDARY = len(resp);

        EXHALE_WINDOW = resp[peaks[-1]:LAST_ZERO_CROSS_BOUNDARY]
        ZERO_CROSS_THRESHOLD = SIMPLE_ZERO_CROSS;

        POSSIBLE_EXHALE_INDS = EXHALE_WINDOW < ZERO_CROSS_THRESHOLD

        # unlikely but possible that no values fit this criteria
        if sum(POSSIBLE_EXHALE_INDS)>0:
            EXHALE_BEST_GUESS = np.where(POSSIBLE_EXHALE_INDS==True)
            exhaleOnsets[0,-1] = peaks[-1] + EXHALE_BEST_GUESS[0][0];
        else:
            # about half a cycle of respiration.
            exhaleOnsets[0,-1] = LAST_ZERO_CROSS_BOUNDARY;

        correctedInhaleOnsets = [int(i) for i in inhaleOnsets[0,:]]
        correctedExhaleOnsets = [int(i) for i in exhaleOnsets[0,:]]

        self.inhaleOnsets = correctedInhaleOnsets
        self.exhaleOnsets = correctedExhaleOnsets
        self.inhalePauseOnsets = inhalePauseOnsets
        self.exhalePauseOnsets = exhalePauseOnsets
        
        
    def findRespiratoryOffsets(self):
        '''calculates when each breath ends
        '''

        inhaleOffsets = np.zeros([1,len(self.inhaleOnsets)])
        exhaleOffsets = np.zeros([1,len(self.exhaleOnsets)])

        # finding inhale offsets
        for bi in range(len(self.exhaleOnsets)):
            if np.isnan(self.inhalePauseOnsets[0,bi]):
                inhaleOffsets[0, bi] = self.exhaleOnsets[bi]-1
            else:
                inhaleOffsets[0, bi] = self.inhalePauseOnsets[0,bi]-1
        # finding exhale offsets
        for bi in range(len(self.exhaleOnsets) - 1):
            if np.isnan(self.exhalePauseOnsets[0,bi]):
                exhaleOffsets[0, bi] = self.inhaleOnsets[bi + 1]-1
            else:
                exhaleOffsets[0, bi] = self.exhalePauseOnsets[0,bi]-1

        # last exhale is different because there is no following inhale
        final_window = self.baselineCorrectedRespiration[self.exhaleOnsets[-1]:]
        putativeExhaleOffsets = np.where(final_window>0)

        # check that there is a real exhale end that it isn't artifact
        avgExhaleLen = np.mean(exhaleOffsets[0, :exhaleOffsets.shape[1]] - self.exhaleOnsets[:len(self.exhaleOnsets)])
        lowerLim = avgExhaleLen / 4;
        upperLim = avgExhaleLen * 1.75
        if putativeExhaleOffsets[0].size>0:
            if putativeExhaleOffsets[0][0] < lowerLim and putativeExhaleOffsets[0][0] >= upperLim:
                exhaleOffsets[0,-1] = self.exhaleOnsets[-1] + putativeExhaleOffsets[0][0] - 1

            else:
                # end of exhale cannot be calculated
                exhaleOffsets[0,-1] = np.nan
        else:
            exhaleOffsets[0,-1]= np.nan
            
        self.inhaleOffsets = inhaleOffsets
        self.exhaleOffsets = exhaleOffsets

        
    def findBreathDurations(self):
        '''calculates durations of breaths
        '''

        inhaleDurations = np.zeros([1,len(self.inhaleOnsets)])
        exhaleDurations = np.zeros([1,len(self.exhaleOnsets)])
        inhalePauseDurations = np.zeros([1,len(self.inhaleOnsets)]);
        exhalePauseDurations = np.zeros([1,len(self.exhaleOnsets)]);

        # calculate inhale durations
        for i in range(len(self.inhaleOnsets)):
            if not np.isnan(self.inhaleOffsets[0,i]):
                inhaleDurations[0,i] = self.inhaleOffsets[0,i]-self.inhaleOnsets[i]
            else:
                inhaleDurations[0,i] = np.nan

        # calculate exhale durations
        for e in range(len(self.exhaleOnsets)):
            if not np.isnan(self.exhaleOffsets[0,e]):
                exhaleDurations[0,e] = self.exhaleOffsets[0,e]-self.exhaleOnsets[e]
            else:
                exhaleDurations[0,e] = np.nan;


        # calculate inhale pause durations
        for i in range(self.inhalePauseOnsets.shape[1]):
            if not np.isnan(self.inhalePauseOnsets[0,i]):
                inhalePauseDurations[0,i] = self.exhaleOnsets[i] - self.inhalePauseOnsets[0,i]
            else:
                inhalePauseDurations[0,i] = np.nan

        # calculate exhale pause durations
        for e in range(self.exhalePauseOnsets.shape[1]):
            if not np.isnan(self.exhalePauseOnsets[0,e]):
                if e<len(self.inhaleOnsets):
                    exhalePauseDurations[0,e] = self.inhaleOnsets[e+1] - self.exhalePauseOnsets[0,e]
                else:
                    exhalePauseDurations[0,e] = np.nan
            else:
                exhalePauseDurations[0,e] = np.nan


        # normalize back into real time and store
        self.inhaleDurations = inhaleDurations / self.srate
        self.exhaleDurations = exhaleDurations / self.srate
        self.inhalePauseDurations = inhalePauseDurations / self.srate
        self.exhalePauseDurations = exhalePauseDurations / self.srate
        
    def estimateBreathVolumes(self):
        '''Estimates the volume of air displaced in all inhales and exhales by 
        calculating the integral of each breath.
        Only valid for airflow signals that have baseline corrected and for which breath onsets and offsets have been computed.
        '''

        inhaleVolumes = np.zeros([1,len(self.inhaleOnsets)]);
        exhaleVolumes = np.zeros([1,len(self.exhaleOnsets)]);

        for bi in range(len(self.inhaleOnsets)):
            if np.isnan(self.inhaleOffsets[0,bi]):
                inhaleVolumes[0,bi]=np.nan
            else:
                thisInhale = np.arange(self.inhaleOnsets[bi],int(self.inhaleOffsets[0,bi]))
                inhaleIntegral = sum(abs(self.baselineCorrectedRespiration[thisInhale]))
                inhaleVolumes[0,bi]=inhaleIntegral


        for bi in range(len(self.exhaleOnsets)):
            if np.isnan(self.exhaleOffsets[0,bi]):
                exhaleVolumes[0,bi]=np.nan
            else:
                thisExhale = np.arange(self.exhaleOnsets[bi],int(self.exhaleOffsets[0,bi]))
                exhaleIntegral = sum(abs(self.baselineCorrectedRespiration[thisExhale]))
                exhaleVolumes[0,bi]=exhaleIntegral



        # normalize for different sampling rates
        self.inhaleVolumes = (inhaleVolumes / self.srate) * 1000
        self.exhaleVolumes = (exhaleVolumes / self.srate) * 1000
        
        
       
    def estimateSecondaryFeatures(self):
        """Calculates breathing rate, average breath duration, etc.
        
        """
        
        nInhales=len(self.inhaleOnsets);
        nExhales=len(self.exhaleOnsets);
        if not self.statuses:
            validInhaleInds=np.arange(nInhales)
            validExhaleInds=np.arange(nExhales)
        #else:
        # will add when gui is complete

        nValidInhales=len(validInhaleInds)
        nValidExhales=len(validExhaleInds)

        # Breathing Rate
        # breathing rate is the sampling rate over the average number of samples in between breaths.

        # this is tricky when certain breaths have been rejected
        breathDiffs=[]

        for i in range(nValidInhales-1):
            thisBreath=validInhaleInds[i]
            nextBreath=validInhaleInds[i+1]
            # if there is no rejected breath between these breaths, they can be used to compute breathing rate.
            if nextBreath == thisBreath+1:
                breathDiffs.append(self.inhaleOnsets[nextBreath]-self.inhaleOnsets[thisBreath])
        breathingRate = self.srate/np.mean(breathDiffs)

        # Inter-Breath Interval
        # inter-breath interval is the inverse of breathing rate
        interBreathInterval = 1/breathingRate

        # Coefficient of Variation of Breathing Rate
        # this describes variability in time between breaths
        cvBreathingRate = np.std(breathDiffs)/np.mean(breathDiffs)

        if self.dataType == 'humanAirflow' or self.dataType == 'rodentAirflow':
            # the following features can only be computed for airflow data

            # Peak Flow Rates %%%
            # the maximum rate of airflow at each inhale and exhale

            # inhales
            validInhaleFlows=self.excludeOutliers(self.peakInspiratoryFlows, validInhaleInds)
            avgMaxInhaleFlow = np.mean(validInhaleFlows)

            # exhales
            validExhaleFlows=self.excludeOutliers(self.troughExpiratoryFlows, validExhaleInds)
            avgMaxExhaleFlow = np.mean(validExhaleFlows)

            # Breath Volumes
            # the volume of each breath is the integral of the airflow

            # inhales
            validInhaleVolumes=self.excludeOutliers(self.inhaleVolumes[0], validInhaleInds)
            avgInhaleVolume = np.mean(validInhaleVolumes)

            # exhales
            validExhaleVolumes=self.excludeOutliers(self.exhaleVolumes[0], validExhaleInds)
            avgExhaleVolume = np.mean(validExhaleVolumes)

            # Tidal volume
            # tidal volume is the total air displaced by inhale and exhale

            avgTidalVolume = avgInhaleVolume + avgExhaleVolume

            # Minute Ventilation
            # minute ventilation is the product of respiration rate and tidal volume
            minuteVentilation = breathingRate * avgTidalVolume

            # Duty Cycle
            # duty cycle is the percent of each breathing cycle that was spent in a phase

            # get avg duration of each phase
            avgInhaleDuration = np.nanmean(self.inhaleDurations)
            avgExhaleDuration = np.nanmean(self.exhaleDurations)

            # because pauses don't necessarily occur on every breath, multiply this value by total number that occured.
            pctInhalePause=(len(self.inhalePauseDurations[0])-sum(np.isnan(self.inhalePauseDurations[0])))/nValidInhales
            avgInhalePauseDuration = np.nanmean(self.inhalePauseDurations[0,validInhaleInds]) * pctInhalePause

            pctExhalePause=(len(self.exhalePauseDurations[0])-sum(np.isnan(self.exhalePauseDurations[0])))/nValidExhales
            avgExhalePauseDuration = np.nanmean(self.exhalePauseDurations[0,validExhaleInds]) * pctExhalePause

            inhaleDutyCycle = avgInhaleDuration / interBreathInterval
            inhalePauseDutyCycle = avgInhalePauseDuration / interBreathInterval
            exhaleDutyCycle = avgExhaleDuration / interBreathInterval
            exhalePauseDutyCycle = avgExhalePauseDuration / interBreathInterval

            CVInhaleDuration = np.nanstd(self.inhaleDurations)/avgInhaleDuration
            CVInhalePauseDuration = np.nanstd(self.inhalePauseDurations)/avgInhalePauseDuration
            CVExhaleDuration = np.nanstd(self.exhaleDurations)/avgExhaleDuration
            CVExhalePauseDuration = np.nanstd(self.exhalePauseDurations)/avgExhalePauseDuration

            # if there were no pauses, the average pause duration is 0, not nan
            if not avgInhalePauseDuration or np.isnan(avgInhalePauseDuration):
                avgInhalePauseDuration=0

            if not avgExhalePauseDuration or np.isnan(avgExhalePauseDuration):
                avgExhalePauseDuration=0

            #coefficient of variation in breath size describes variability of breath sizes
            CVTidalVolume = np.std(validInhaleVolumes)/np.mean(validInhaleVolumes)


            # assigning values for output

            if self.dataType == 'humanAirflow' or self.dataType == 'rodentAirflow':
                secondaryStats = {'Breathing Rate': breathingRate,
                                 'Average Inter-Breath Interval' : interBreathInterval,
                                 'Average Peak Inspiratory Flow' : avgMaxInhaleFlow,
                                 'Average Peak Expiratory Flow' : avgMaxExhaleFlow,
                                 'Average Inhale Volume' : avgInhaleVolume,
                                 'Average Exhale Volume' : avgExhaleVolume,
                                 'Average Tidal Volume' : avgTidalVolume,
                                 'Minute Ventilation' : minuteVentilation,
                                 'Duty Cycle of Inhale' : inhaleDutyCycle,
                                 'Duty Cycle of Inhale Pause' : inhalePauseDutyCycle,
                                 'Duty Cycle of Exhale' : exhaleDutyCycle,
                                 'Duty Cycle of Exhale Pause' : exhalePauseDutyCycle,
                                 'Coefficient of Variation of Inhale Duty Cycle' : CVInhaleDuration,
                                 'Coefficient of Variation of Inhale Pause Duty Cycle' : CVInhalePauseDuration,
                                 'Coefficient of Variation of Exhale Duty Cycle' : CVExhaleDuration,
                                 'Coefficient of Variation of Exhale Pause Duty Cycle' : CVExhalePauseDuration,
                                 'Average Inhale Duration' : avgInhaleDuration,
                                 'Average Inhale Pause Duration' : avgInhalePauseDuration,
                                 'Average Exhale Duration' : avgExhaleDuration,
                                 'Average Exhale Pause Duration' : avgExhalePauseDuration,
                                 'Percent of Breaths With Inhale Pause' : pctInhalePause,
                                 'Percent of Breaths With Exhale Pause' : pctExhalePause,
                                 'Coefficient of Variation of Breathing Rate' : cvBreathingRate,
                                 'Coefficient of Variation of Breath Volumes' : CVTidalVolume
                                }

            elif self.dataType == 'humanBB' or self.dataType == 'rodentThermocouple':
                secondaryStats = {'Breathing Rate': breathingRate,
                                  'Average Inter-Breath Interval' : interBreathInterval,
                                  'Coefficient of Variation of Breathing Rate' : cvBreathingRate
                                 }
                
            self.secondaryFeatures = secondaryStats
        
    # bookkeeping methods
    def running_mean(self, x, N):
        # calculates sliding window mean
        startInd=int(np.floor(N/2))
        cumsum = np.cumsum(np.insert(x, 0, 0)) 
        unPadded=(cumsum[N:] - cumsum[:-N]) / N
        smoothedData=np.zeros([1,len(x)])
        smoothedData[0,startInd:startInd+len(unPadded)]=unPadded
        smoothedData=smoothedData[0]

        return smoothedData
    
    def excludeOutliers(self,origVals,validBreathInds):
        # rejects values exceeding 2 stds from the mean

        upperBound=np.nanmean(origVals) + 2 * np.nanstd(origVals)
        lowerBound=np.nanmean(origVals) - 2 * np.nanstd(origVals)

        validValInds = np.intersect1d(np.where(origVals> lowerBound),np.where(origVals < upperBound))

        validVals = origVals[np.intersect1d(validValInds, validBreathInds)]

        return validVals

    def estimateAllFeatures(self, zScore=0, baselineCorrectionMethod='sliding', simplify=1, verbose=0 ):
        '''Calls all of the above functions in the correct order to estimate all features in the respiratory signal.
        
        Parameters
        ----------
        zScore : 0 or 1, respiratory signal is zscored if set to 1
        baselineCorrectionMethod : 'simple' subtracts the average of the signal to baseline
                                   'sliding' uses subtracts the average of minute long windows to baseline
        simplify : 0 or 1, nInhales is set to nExhales by removing the last exhale if set to 1
        
        '''
        if self.dataType == 'humanAirflow' or self.dataType == 'rodentAirflow':
            self.smoothResp()
            self.correctRespirationToBaseline(method=baselineCorrectionMethod,zScore=zScore)
            self.labelExtrema(simplify=simplify)
            self.findRespiratoryPausesAndOnsets()
            self.findRespiratoryOffsets()
            self.findBreathDurations()
            self.estimateBreathVolumes()
            self.estimateSecondaryFeatures()
        elif self.dataType == 'humanBB' or self.dataType == 'rodentThermocouple':
            self.correctRespirationToBaseline(method=baselineCorrectionMethod, zScore=zScore)
            self.labelExtrema(simplify=simplify)
            self.estimateSecondaryFeatures(verbose)
        
    