This readme file was generated on 2025-05-DD by Di Song


# GENERAL INFORMATION

* Title of Dataset: Data for egg hyperspectral image

## Author/Principal Investigator Information
Name: Di Song
ORCID:
Institution: University of Illinois Urbana-Champaign
Email: disong2@illinois.edu


* Date of data collection: *provide single date, range, or approximate date; suggested format YYYY-MM-DD)*
* Geographic location of data collection: Department of Agricultural and Biological Engineering, UIUC
* Information about funding sources that supported the collection of the data: This work was supported by the USDA National Institute of Food and Agriculture, Award # 2023-67015-39154.


# SHARING/ACCESS INFORMATION

* Licenses/restrictions placed on the data: None
* Links to publications that cite or use the data: 
Please refer to the Illinois Data Bank for up to date list of related materials.

* Links to other publicly accessible locations of the data: None
* Links/relationships to ancillary data sets: None
* Recommended citation for this dataset:
Please refer to the Illinois Data Bank for how to cite this dataset.


# DATA & FILE OVERVIEW

## File List: this dataset contains 2 files:
	1. all_data.rar: this folder is 219GB and contains 1228 egg hyperspectral images, the wavelength from 374 nm to 1015 nm (374.14, 376.18, 378.2, 380.24, 382.27, 384.3, 386.34, 388.37, 390.4, 392.44, 394.48, 396.51, 398.55, 400.59, 402.63, 404.67, 406.71, 408.75, 410.79, 412.84, 414.88, 416.92, 418.97, 421.02, 423.06, 425.11, 427.16, 429.21, 431.26, 433.31, 435.36, 437.42, 439.46, 441.52, 443.58, 445.63, 447.68, 449.74, 451.8, 453.86, 455.92, 457.98, 460.04, 462.1, 464.16, 466.22, 468.29, 470.36, 472.42, 474.49, 476.56, 478.62, 480.7, 482.76, 484.83, 486.9, 488.98, 491.05, 493.12, 495.2, 497.27, 499.34, 501.42, 503.5, 505.58, 507.66, 509.73, 511.82, 513.9, 515.98, 518.06, 520.14, 522.23, 524.31, 526.4, 528.48, 530.57, 532.66, 534.75, 536.84, 538.93, 541.02, 543.11, 545.2, 547.3, 549.4, 551.48, 553.58, 555.67, 557.78, 559.88, 561.98, 564.08, 566.17, 568.28, 570.38, 572.48, 574.58, 576.68, 578.79, 580.9, 583.0, 585.11, 587.22, 589.33, 591.43, 593.55, 595.66, 597.77, 599.88, 602.0, 604.11, 606.23, 608.34, 610.46, 612.58, 614.69, 616.81, 618.93, 621.05, 623.17, 625.3, 627.42, 629.54, 631.67, 633.79, 635.92, 638.05, 640.17, 642.3, 644.43, 646.56, 648.69, 650.82, 652.95, 655.08, 657.22, 659.36, 661.48, 663.62, 665.76, 667.9, 670.04, 672.18, 674.32, 676.46, 678.6, 680.74, 682.88, 685.02, 687.17, 689.32, 691.46, 693.6, 695.76, 697.9, 700.06, 702.2, 704.36, 706.5, 708.66, 710.81, 712.96, 715.12, 717.27, 719.43, 721.58, 723.74, 725.9, 728.06, 730.22, 732.38, 734.54, 736.7, 738.87, 741.03, 743.2, 745.36, 747.53, 749.7, 751.86, 754.03, 756.2, 758.37, 760.54, 762.71, 764.88, 767.06, 769.23, 771.4, 773.58, 775.76, 777.94, 780.11, 782.29, 784.46, 786.64, 788.82, 791.01, 793.2, 795.38, 797.56, 799.74, 801.93, 804.12, 806.3, 808.49, 810.68, 812.86, 815.06, 817.25, 819.44, 821.63, 823.82, 826.02, 828.21, 830.41, 832.6, 834.8, 837.0, 839.2, 841.4, 843.6, 845.8, 848.0, 850.2, 852.4, 854.61, 856.81, 859.02, 861.22, 863.43, 865.64, 867.85, 870.06, 872.27, 874.48, 876.69, 878.9, 881.12, 883.33, 885.54, 887.76, 889.97, 892.2, 894.42, 896.63, 898.84, 901.06, 903.3, 905.52, 907.73, 909.96, 912.18, 914.4, 916.64, 918.86, 921.08, 923.32, 925.54, 927.77, 930.0, 932.23, 934.46, 936.7, 938.93, 941.16, 943.4, 945.64, 947.87, 950.11, 952.35, 954.58, 956.82, 959.06, 961.3, 963.55, 965.79, 968.03, 970.28, 972.52, 974.77, 977.02, 979.26, 981.51, 983.76, 986.01, 988.26, 990.51, 992.76, 995.02, 997.27, 999.52, 1001.78, 1004.03, 1006.29, 1008.54, 1010.8, 1013.06, 1015.32).Open the each spectra with 'load('Sample ID.mat', 'spec')' in Matlab.
	2. spectra and reference parameters.csv: This file contains extracted average spectra (pixel intensity) from the hyperspectral images of egg and reference parameters.

* Relationship between files, if important: First file contains hyperspectral images where second file contains corresponding spectra and reference parameters.

* Additional related data collected that was not included in the current data package: None


# METHODOLOGICAL INFORMATION

## Description of methods used for collection/generation of data: 
Hyperspectral images were acquired using a line-scan benchtop imaging system (Resonon Inc., Bozeman, MT, USA) operated in a darkroom.
Reference parameters were measured / detected following standard methods.

## Methods for processing the data: 
Matlab was used for average spectra extraction from hyperspectal images.

## Instrument- or software-specific information needed to interpret the data: 
Matlab 2022a.



# DATA-SPECIFIC INFORMATION FOR: Egg dataset
1. all_data.rar: 1228 egg hyperspectral images 


2. spectra and reference parameters.csv 

* Number of variables: 308
* Number of cases/rows: 1228
* Variable List: 
Sample ID : The ID 'Egg-F-03Dec-0096' indicates the spectra of the 96th fertile egg collected on December 3rd.
Mass (g): Mass of intact egg in gram (g).
Major dia (mm):Major or highest diameter of egg in milimeter (mm).
Minor dia (mm):Minor or lowest diameter of egg in milimeter (mm).
Fertility status: Whether egg was fertile (which has male or female part) or infertile (which has only female part).
Thickness (mm): Egg shell thickness in milimeter (mm).
Yolk mass (g): The mass of yolk in gram (g).
Shell strength (N): The strength of intact egg shell in newton (N).

First row from Column I to column KV: The spectra variables as wavelength in nanometer (nm).
 
