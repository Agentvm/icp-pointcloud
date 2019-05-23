def get_matching_filenames(filename):
    """
    finds matching filenames from one point cloud to another.
    Matches ALS16 to DIM16
    Inputs:
    filename: string; filename of the original file to split
    Outputs:
    [s1, s2, s3, s4]: list of string;
    s1: xmin and ymin
    s2: xmin and ymean
    s3: xmean and ymin
    s4: xmean and ymean
    """
    # get filenames
    s = filename.split('_')[0]

    # get minimal and mean values
    xmin = int(s[0:5] + '0')
    ymin = int(s[5:])
    xmean = xmin + 5
    ymean = ymin + 5
    
    # build file strings
    prep = 'DSM_Cloud_'
    ending = '.las'
    s1 = "".join([prep, str(xmin), '_', str(ymin), ending])
    s2 = "".join([prep, str(xmin), '_', str(ymean), ending])
    s3 = "".join([prep, str(xmean), '_', str(ymin), ending])
    s4 = "".join([prep, str(xmean), '_', str(ymean), ending])
    return [s1, s2, s3, s4]


# y >= 40 ###
# x = 30
#print (get_matching_filenames ("3331359940_1_2016-11-28.las" ))     # Y/YZ (Train)
#print (get_matching_filenames ("3331359950_1_2016-11-28.las" ))    # Forest
#print (get_matching_filenames ("3331359960_1_2016-11-28.las" ))

# x = 40
#print (get_matching_filenames ("3331459940_1_2016-11-28.las" ))
#print (get_matching_filenames ("3331459950_1_2016-11-28.las" ))    # DIM showcase, missing Building
#print (get_matching_filenames ("3331459960_1_2016-11-28.las" ))

# x = 50
#print (get_matching_filenames ("3331559940_1_2016-11-28.las" ))
#print (get_matching_filenames ("3331559950_1_2016-11-28.las" ))
#print (get_matching_filenames ("3331559960_1_2016-11-28.las" ))

# x = 60
#print (get_matching_filenames ("3331659940_1_2016-11-28.las" ))
print (get_matching_filenames ("3331659950_1_2016-11-28.las" ))    # XZ, everything, XYZ, Acker, XY, Fahrbahn
#print (get_matching_filenames ("3331659960_1_2016-11-28.las" ))
