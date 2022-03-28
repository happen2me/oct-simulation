# OCT Simulation
Simulate OCT eye scanning.

## (Potential) Datasets

Dataset analysis: [![Datasets Analysis](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1rVfWlkUjGyjN4Rw1B9GNANtr2Zdln7JO?usp=sharing)

- Segmentation of OCT images (DME), Chiu 2015 [[kaggle](https://www.kaggle.com/paultimothymooney/chiu-2015)] [[homepage@duke](https://people.duke.edu/~sf59/Chiu_BOE_2014_dataset.htm)] [[paper](https://opg.optica.org/boe/fulltext.cfm?uri=boe-6-4-1172&id=312754#)]
- Segmentation of OCT images (AMD), Farsiu 2013 [[kaggle](https://www.kaggle.com/paultimothymooney/farsiu-2014/home)] [[homepage@duke](https://people.duke.edu/~sf59/RPEDC_Ophth_2013_dataset.htm)] [[paper](http://people.duke.edu/~sf59/Farsiu_Ophthalmology_2013.pdf)]
- AROI dataset, Melinščak 2021 [[homepage](https://ipg.fer.hr/ipg/resources/oct_image_database)]
- OCTID: Optical Coherence Tomography Image Database, Peyman Gholami, Priyanka Roy, Mohana Kuppuswamy Parthasarathy, Vasudevan Lakshminarayanan [[data](https://dataverse.scholarsportal.info/dataverse/OCTID)]
- Real-Time Automatic Segmentation of Optical Coherence Tomography Volume Data of the Macular Region, Delia Cabrera Debuc [[data](http://www.plosone.org/article/fetchSingleRepresentation.action?uri=info:doi/10.1371/journal.pone.0133908.s002)][[paper](https://www.researchgate.net/publication/280944318_Real-Time_Automatic_Segmentation_of_Optical_Coherence_Tomography_Volume_Data_of_the_Macular_Region)]

## Train with Pix2pix Model

[![Train](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/15z0Wpn032z-PGBG-tObwSbYI9yht_xDW?usp=sharing)

## Terminology

- OCT: Optical Coherency Tomography
- En face: "En face" is an emerging imaging technique derived from spectral domain optical coherence tomography (OCT). It produces frontal sections of retinal layers, also called "C-scan OCT." Outer retinal tubulations (ORTs) in age-related macular degeneration (AMD) are a recent finding evidenced by spectral-domain OCT.
- B-scans: 
- Fundus (眼底): The interior surface of the eye opposite the lens and includes the retina, optic disc, macula, fovea, and posterior pole.  [[wiki](https://en.wikipedia.org/wiki/Fundus_(eye))]
- Macula (黄斑)
  ![Blausen 0389 EyeAnatomy 02.png](https://upload.wikimedia.org/wikipedia/commons/thumb/3/30/Blausen_0389_EyeAnatomy_02.png/250px-Blausen_0389_EyeAnatomy_02.png)

- AMD: Age-related macular degeneration [[nih](https://www.nei.nih.gov/learn-about-eye-health/eye-conditions-and-diseases/age-related-macular-degeneration)]
- DME: Diabetic macular edema (糖尿病性黄斑水肿)

