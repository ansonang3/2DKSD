This toolbox was created on September 2019, 7th
by Anna SONG (anna.song.maths@gmail.com)

PLEASE REFER TO THIS WORK WHEN USING THIS CODE

Sparse Dictionary Learning for 2D Kendall Shapes
by Anna Song, Virginie Uhlmann, Julien Fageot and Michael Unser (2019)


2D KSD Toolbox contains :

	- settings.py : sets the complex framework, and the dataset is converted into pre-shapes 
	- splines_utils.py : auxiliary functions for handling splines
	- shapespace.py : standard operations in Kendall's shape space (not much used)
	- alignfirst_dico.py : "align-first" method, relying on the SPAMS toolbox
	- KSD.py : our algorithm (Method of Optimal Directions + Cholesky-optimized Order Recursive Matching Pursuit)

and 2 folders :

	- RESULTS : in which the results are saved ; already contains some examples of results.
	- DATA : contains some datasets (landmarks, open Hermite splines, closed Hermite splines, closed cubic B splines)

ABOUT THE DATA :

- datasets with landmarks :

	- 'worms' : C. elegans behavioural database
		Yemini et al., A database of C. elegans behavioral
		phenotypes, Nature Methods, pp. 877–879.
	- 'hands' : Stegmann and Gomez, A Brief Introduction to Statistical Shape Analysis, 2002
	- 'leaves_sym' : the Kaggle Leaf Dataset
	- 'mpeg7' and 'mpeg7_sampled100' : the MPEG7 dataset
	- 'horses' : the Weizmann Horses Dataset
	- '900_horses_leaves_worms' : a mix with 300 horses, 300 leaves and 300 worms

- datasets with Hermite splines :
	
	- open ones : '50_open_worms_6ctrlpts','200_open_worms_6ctrlpts','6376_open_worms_6ctrlpts'
	- closed ones : '200_closed_worms_8ctrlpts','100_closed_worms_10ctrlpts'
		Data courtesy of Dr. Laurent Mouchiroud (Laboratory of Integrative and Systems Physiology, EPFL, 			Switzerland) and Dr. Matteo Cornaglia (Microsystems Laboratory 2, EPFL, Switzerland).
		The two small datasets are extracts of the third one.

- datasets with closed cubic B-splines :
	- 'leaves_Bsplines' : the Kaggle Leaf Dataset
