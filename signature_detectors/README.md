# Gatorsense hsitoolkit signature detectors Python version
hsi_toolkit_py/signature_detectors

Current suite of signature detectors:
- ace_detector: Squared Adaptive Cosine/Coherence Estimator (Squared ACE), cosine of vector angle between target and pixel spectra after whitening based on background statistics, squared
- ace_local_detector: Adaptive Cosine/Coherence Estimator where background statistics are estimated from local window
- ace_ss_detector: Squared Adaptive Cosine/Coherence Estimator Subspace Formulation
- acert_detector: Adaptive Cosine/Coherence Estimator (ACE), cosine of vector angle between target and pixel spectra after whitening based on background statistics
- fam_statistic: False Alarm Mitigation Statistic
- osp_detector: Orthogonal Subspace Projection Detector
- smf_detector: Spectral Matched Filter, inner product between target and pixel spectra after whitening based on background statistics
- smf_local_detector: Spectral Matched Filter using local background statistics


Suite of anomaly detectors in development:
?- acert_max_detector: ACE given multiple target signatures. Confidence value for each pixel is max ACE score over all target signatures.
!- amsd_detector: Adaptive Matched Subspace Detector
? - ccmf_detector: Class Conditional Matched Filter, Segment data using a Gaussian Mixture Model and then apply SMF to each component
?- cem_detector: Constrained Energy Minimization Detector
?- ctmf_detector: Cluster Tuned Matched Filter
!- ftmf_statistic: Finite Target Matched Filter
?- qmf_detector: Quadratic Spectral Matched Filter
?- sam_detector: Spectral Angle Mapper, calculates vector angle between target signature and each pixel spectrum
?- smf_max_detector: Spectral Matched Filter given multiple target signatures. Confidence value for each pixel is max SMF score over all target signatures.
?!- spsmf_detector: Subpixel Spectral Matched Filter
?- palm_detector: Pairwise Adaptive Linear Matched Filter


Suite of anomaly detectors to be implemented:
- abd_detector: Abudance of target signature when unmixed using target signature and background endmembers assuming the linear mixing model
- ha_detector: Hybrid Abundance Detector, unmix using background endmembers as well as using background and target endmembers, model proportions with a Gaussian mixture, compute pixel-wise likelihood ratios
- hs_detector: Hybrid Subpixel Detector
- hsa_detector: Hybrid Structured/Abundance Detector
- hsd_detector: Likelihood ratio after unmixing with background and unmixing with background and target signature (Broadwater and Chellappa's method)
- hsd_local_detector: Hybrid Structured Detector using local background estimation
- hua_detector: Hybrid Unstructured Abundance Detector
- hud_detector: Hybrid Unstructured Detector (Broadwater and Chellappa's method)
- mfmf_statistic: Mixture Tuned Matched Filter Infeasibility Statistic


<!-- Anomaly detectors can also be run in *Segmented* mode using segmented.m utility.
Segmented mode is where a detector is applied to segments of the imagery separately (i.e., background statistics computed from segment rather than full image).  
See segmented examples in demo code. -->

Review and references for many of these signature detectors can be found in the literature review in Taylor Glenn's Ph.D. thesis:

Glenn, Taylor C. Context-dependent detection in hyperspectral imagery. Diss. University of Florida, 2013.

Contact: Alina Zare, azare@ufl.edu
