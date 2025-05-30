# The location should be <your_location_to_the_dataset>, so you need to 
# make sure that all the groundtruth files can be found in that location, 
# a folder "words/" should also be there, which contains all the images
# without any folder or sub-folder.
baseDir_word = '../../DATASETS/IAM/striked_v3/'
baseDir_line = '/home/lkang/datasets/iam_final_lines/'
#dataset = 'CLEAN'

ignore_chars = ['#']

#probability of clean Vs striked and different striked types
# baseDir_stikeword = '../DATASETS/IAM/striked_v3/'
#baseDir_stikeword = '../DATASETS/IAM/strike_removed/'
# probability_clean = 0
# Probabilities must sum to 1
# striked_types = {
#     "CLEAN": 0.0, 
#     "MIXED": 0.0,
#     "CLEANED_CLEAN": 0.0,
#     "CLEANED_MIXED": 1.0,
#     "CROSS": 0.0, 
#     "DIAGONAL": 0.0, 
#     "DOUBLE_LINE": 0.0, 
#     "SCRATCH": 0.0, 
#     "SINGLE_LINE": 0.0, 
#     "WAVE": 0.0, 
#     "ZIG_ZAG": 0.0, 
#     "BLOT_1": 0.0, 
#     "BLOT_2": 0.0
# }

# striked_types = {
#     "CROSS": 0.1428, 
#     "DIAGONAL": 0.1428, 
#     "DOUBLE_LINE": 0.1428, 
#     "SCRATCH": 0.1428, 
#     "SINGLE_LINE": 0.1428, 
#     "WAVE": 0.1428, 
#     "ZIG_ZAG": 0.1428, 
#     "BLOT_1": 0.0, 
#     "BLOT_2": 0.0
# }