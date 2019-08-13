'''
Contains reference translations determined by point picking alignment for all extracted regions (Old).
Format: (reference_cloud_path, aligned_cloud_path):(translation_x, translation_y, translation_z, RMS)
'''

reference_translations = {

    # ### reference: ALS16 <- DIM aligned ##################################

    # Color Houses
    ('clouds/Regions/Color Houses/ALS16_Cloud_reduced_normals_cleared.asc',
    'clouds/Regions/Color Houses/DSM_Cloud_reduced_normals.asc'):
    ((-0.041994094849, 0.044445037842, 0.017503738403), 0.110672),

    # DIM showcase
    ('clouds/Regions/DIM showcase/ALS16_Cloud_reduced_normals_cleared.asc',
    'clouds/Regions/DIM showcase/DSM_Cloud_reduced_normals.asc'):
    ((-0.003377914429, 0.093841552734, -0.133575439453),
    0.107081),

    # Everything
    ('clouds/Regions/Everything/ALS16_Cloud_reduced_normals_cleared.asc',
    'clouds/Regions/Everything/DSM_Cloud_reduced_normals.asc'):
    ((0.057880401611, 0.046981811523, -0.015285491943),
    0.0953088),

    # Field
    ('clouds/Regions/Field/ALS16_Cloud_reduced_normals_cleared.asc',
    'clouds/Regions/Field/DSM_Cloud_reduced_normals.asc'):
    ((0.002349853516, -0.057266235352, 0.240567699075),
    0.139969),

    # Forest
    ('clouds/Regions/Forest/ALS16_Cloud_reduced_normals_cleared.asc',
    'clouds/Regions/Forest/DSM_Cloud_reduced_normals.asc'):
    ((0.0, 0.0, 0.0),
    0.0),

    # Missing Building
    ('clouds/Regions/Missing Building/ALS16_Cloud_reduced_normals_cleared.asc',
    'clouds/Regions/Missing Building/DSM_Cloud_reduced_normals.asc'):
    ((-0.001159667969, 0.030044555664, 0.017412185669),
    0.131601),

    # Road
    ('clouds/Regions/Road/ALS16_Cloud_reduced_normals_cleared.asc',
    'clouds/Regions/Road/DSM_Cloud_reduced_normals.asc'):
    ((0.023117065430, 0.008197784424, 0.013129234314),
    0.122388),

    # Xy Tower
    ('clouds/Regions/Xy Tower/ALS16_Cloud_reduced_normals_cleared.asc',
    'clouds/Regions/Xy Tower/DSM_Cloud_reduced_normals.asc'):
    ((-0.827770233154, 0.162506103516, 0.191293716431),
    0.116207),

    # Xyz Square
    ('clouds/Regions/Xyz Square/ALS16_Cloud_reduced_normals_cleared.asc',
    'clouds/Regions/Xyz Square/DSM_Cloud_reduced_normals.asc'):
    ((-0.005023956299, 0.001205444336, 0.083156585693),
    0.0953),

    # Xz Hall
    ('clouds/Regions/Xz Hall/ALS16_Cloud_reduced_normals_cleared.asc',
    'clouds/Regions/Xz Hall/DSM_Cloud_reduced_normals.asc'):
    ((-0.077413558960, 0.121776580811, 0.030170440674),
    0.110266),

    # Yz Houses
    ('clouds/Regions/Yz Houses/ALS16_Cloud_reduced_normals_cleared.asc',
    'clouds/Regions/Yz Houses/DSM_Cloud_reduced_normals.asc'):
    ((0.314620971680, -0.019294738770, -0.035737037659),
    0.252714),

    # Yz Street
    ('clouds/Regions/Yz Street/ALS16_Cloud_reduced_normals_cleared.asc',
    'clouds/Regions/Yz Street/DSM_Cloud_reduced_normals.asc'):
    ((0.055110931396, -0.004455566406, -0.095468521118),
    0.0381034),

    # ### reference: ALS14 <- ALS16 aligned ##################################

    # Everything
    ('clouds/Regions/Everything/ALS14_Cloud_reduced_normals_cleared.asc',
    'clouds/Regions/Everything/ALS16_Cloud_reduced_normals_cleared.asc'):
    ((0.015129089355, -0.034282684326, 0.209009647369), 0.0862935),

    # Missing Building
    ('clouds/Regions/Missing Building/ALS14_Cloud_reduced_normals_cleared.asc',
    'clouds/Regions/Missing Building/ALS16_Cloud_reduced_normals_cleared.asc'):
    ((-0.039558410645, -0.025497436523, 0.008216857910), 0.282216),


    # ### reference: ALS14 <- DIM aligned ##################################

    # Everything
    ('clouds/Regions/Everything/ALS14_Cloud_reduced_normals_cleared.asc',
    'clouds/Regions/Everything/DSM_Cloud_reduced_normals.asc'):
    ((0.015129089355, -0.034282684326, 0.209009647369), 0.0862935),

    # Missing Building
    ('clouds/Regions/Missing Building/ALS14_Cloud_reduced_normals_cleared.asc',
    'clouds/Regions/Missing Building/DSM_Cloud_reduced_normals.asc'):
    ((-0.039558410645, -0.025497436523, 0.008216857910), 0.282216),

    # ### reference: ALS16_Scan_54 <- ALS16_Scan_55 aligned ##################################

    # Everything
    ('clouds/Regions/Everything/ALS16_Cloud _Scan54_reduced_normals.asc',
    'clouds/Regions/Everything/ALS16_Cloud _Scan55_reduced_normals.asc'):
    ((0.005805969238, -0.011207580566, -0.018814086914), 0.0452853)

}


# #########################################
# icp_translations                      ###
# #########################################

icp_translations = {

    # ### reference: ALS16 <- DIM aligned ##################################

    # Color Houses
    ('clouds/Regions/Color Houses/ALS16_Cloud_reduced_normals_cleared.asc',
    'clouds/Regions/Color Houses/DSM_Cloud_reduced_normals.asc'):
    ((0.09864286, 0.03564401, -0.23459308), 0.0),

    # DIM showcase
    ('clouds/Regions/DIM showcase/ALS16_Cloud_reduced_normals_cleared.asc',
    'clouds/Regions/DIM showcase/DSM_Cloud_reduced_normals.asc'):
    ((0.12792622, -0.24462657, -0.14004812), 0.0),

    # Everything
    ('clouds/Regions/Everything/ALS16_Cloud_reduced_normals_cleared.asc',
    'clouds/Regions/Everything/DSM_Cloud_reduced_normals.asc'):
    ((-0.01512663, 0.11508806, -0.09307337), 0.0),

    # Field
    ('clouds/Regions/Field/ALS16_Cloud_reduced_normals_cleared.asc',
    'clouds/Regions/Field/DSM_Cloud_reduced_normals.asc'):
    ((-0.02999802, 0.00536467, -0.14919334), 0.00000851),

    # Forest
    ('clouds/Regions/Forest/ALS16_Cloud_reduced_normals_cleared.asc',
    'clouds/Regions/Forest/DSM_Cloud_reduced_normals.asc'):
    ((-2.25365287, -3.36473759, -0.11304421), 0.0),

    # Missing Building
    ('clouds/Regions/Missing Building/ALS16_Cloud_reduced_normals_cleared.asc',
    'clouds/Regions/Missing Building/DSM_Cloud_reduced_normals.asc'):
    ((-0.05861624, -0.19376078, -0.13757500), 0.00013530),

    # Road
    ('clouds/Regions/Road/ALS16_Cloud_reduced_normals_cleared.asc',
    'clouds/Regions/Road/DSM_Cloud_reduced_normals.asc'):
    ((-1.04372880, 0.35911112, -0.13534037), 0.0),

    # Xy Tower
    ('clouds/Regions/Xy Tower/ALS16_Cloud_reduced_normals_cleared.asc',
    'clouds/Regions/Xy Tower/DSM_Cloud_reduced_normals.asc'):
    ((0.01669736, -0.00830808, 0.03436319), 0.0),

    # Xyz Square
    ('clouds/Regions/Xyz Square/ALS16_Cloud_reduced_normals_cleared.asc',
    'clouds/Regions/Xyz Square/DSM_Cloud_reduced_normals.asc'):
    ((-0.08984794, 0.17435764, 0.01461246), 0.0),

    # Xz Hall
    ('clouds/Regions/Xz Hall/ALS16_Cloud_reduced_normals_cleared.asc',
    'clouds/Regions/Xz Hall/DSM_Cloud_reduced_normals.asc'):
    ((-0.04716166, -0.07800126, -0.03992708), 0.0),

    # Yz Houses
    ('clouds/Regions/Yz Houses/ALS16_Cloud_reduced_normals_cleared.asc',
    'clouds/Regions/Yz Houses/DSM_Cloud_reduced_normals.asc'):
    ((0.35897160, 0.01135238, -0.10182163), 0.0),

    # Yz Street
    ('clouds/Regions/Yz Street/ALS16_Cloud_reduced_normals_cleared.asc',
    'clouds/Regions/Yz Street/DSM_Cloud_reduced_normals.asc'):
    ((-0.17405900, 0.14075322, -0.16364951), 0.0),

    # ### reference: ALS14 <- ALS16 aligned ##################################

    # Everything
    ('clouds/Regions/Everything/ALS14_Cloud_reduced_normals_cleared.asc',
    'clouds/Regions/Everything/ALS16_Cloud_reduced_normals_cleared.asc'):
    ((-8.11195657, 2.92718860, -0.31691730), 0.0),

    # Missing Building
    ('clouds/Regions/Missing Building/ALS14_Cloud_reduced_normals_cleared.asc',
    'clouds/Regions/Missing Building/ALS16_Cloud_reduced_normals_cleared.asc'):
    ((-0.01119095, -0.18303436, 0.80700484), 0.00002504),


    # ### reference: ALS14 <- DIM aligned ##################################

    # Everything
    ('clouds/Regions/Everything/ALS14_Cloud_reduced_normals_cleared.asc',
    'clouds/Regions/Everything/DSM_Cloud_reduced_normals.asc'):
    ((-8.47712255, 1.58571202, -0.61282978), 0.0),

    # Missing Building
    ('clouds/Regions/Missing Building/ALS14_Cloud_reduced_normals_cleared.asc',
    'clouds/Regions/Missing Building/DSM_Cloud_reduced_normals.asc'):
    ((0.48469832, -0.56654584, 0.66713481), 0.0),

    # ### reference: ALS16_Scan_54 <- ALS16_Scan_55 aligned ##################################

    # Everything
    ('clouds/Regions/Everything/ALS16_Cloud _Scan54_reduced_normals.asc',
    'clouds/Regions/Everything/ALS16_Cloud _Scan55_reduced_normals.asc'):
    ((0.34429442, -0.05473916, -0.00426233), 0.00009669),

}


# #########################################
# distance_consensus_translations       ###
# #########################################

distance_consensus_translations = {

    # ### reference: ALS16 <- DIM aligned ##################################

    # Color Houses
    ('clouds/Regions/Color Houses/ALS16_Cloud_reduced_normals_cleared.asc',
    'clouds/Regions/Color Houses/DSM_Cloud_reduced_normals.asc'):
    ((0.0, 0.6, 0.0), 0.0),

    # DIM showcase
    ('clouds/Regions/DIM showcase/ALS16_Cloud_reduced_normals_cleared.asc',
    'clouds/Regions/DIM showcase/DSM_Cloud_reduced_normals.asc'):
    ((0.3, -0.15, 0.0), 0.0),

    # Everything
    ('clouds/Regions/Everything/ALS16_Cloud_reduced_normals_cleared.asc',
    'clouds/Regions/Everything/DSM_Cloud_reduced_normals.asc'):
    ((-0.60000000, -0.30000000, 0.0), 0.0),

    # Field
    ('clouds/Regions/Field/ALS16_Cloud_reduced_normals_cleared.asc',
    'clouds/Regions/Field/DSM_Cloud_reduced_normals.asc'):
    ((-1.05000000, 0.15000000, -0.15000000), 0.0),

    # Forest
    ('clouds/Regions/Forest/ALS16_Cloud_reduced_normals_cleared.asc',
    'clouds/Regions/Forest/DSM_Cloud_reduced_normals.asc'):
    ((-0.30000000, 0.60000000, -0.60000000), 0.0),

    # Missing Building
    ('clouds/Regions/Missing Building/ALS16_Cloud_reduced_normals_cleared.asc',
    'clouds/Regions/Missing Building/DSM_Cloud_reduced_normals.asc'):
    ((0.15000000, -0.15000000, 0.0), 0.0),

    # Road
    ('clouds/Regions/Road/ALS16_Cloud_reduced_normals_cleared.asc',
    'clouds/Regions/Road/DSM_Cloud_reduced_normals.asc'):
    ((-0.15000000, 0.15000000, 0.0), 0.0),

    # Xy Tower
    ('clouds/Regions/Xy Tower/ALS16_Cloud_reduced_normals_cleared.asc',
    'clouds/Regions/Xy Tower/DSM_Cloud_reduced_normals.asc'):
    ((-0.90000000, 0.30000000, 0.0), 0.0),

    # Xyz Square
    ('clouds/Regions/Xyz Square/ALS16_Cloud_reduced_normals_cleared.asc',
    'clouds/Regions/Xyz Square/DSM_Cloud_reduced_normals.asc'):
    ((0.0, -0.75000000, 0.0), 0.0),

    # Xz Hall
    ('clouds/Regions/Xz Hall/ALS16_Cloud_reduced_normals_cleared.asc',
    'clouds/Regions/Xz Hall/DSM_Cloud_reduced_normals.asc'):
    ((0.75000000, -0.60000000, 0.0), 0.0),

    # Yz Houses
    ('clouds/Regions/Yz Houses/ALS16_Cloud_reduced_normals_cleared.asc',
    'clouds/Regions/Yz Houses/DSM_Cloud_reduced_normals.asc'):
    ((0.90000000, 0.15000000, 0.0), 0.0),

    # Yz Street
    ('clouds/Regions/Yz Street/ALS16_Cloud_reduced_normals_cleared.asc',
    'clouds/Regions/Yz Street/DSM_Cloud_reduced_normals.asc'):
    ((-0.30000000, -0.75000000, 0.0), 0.0),

    # ### reference: ALS14 <- ALS16 aligned ##################################

    # Everything
    ('clouds/Regions/Everything/ALS14_Cloud_reduced_normals_cleared.asc',
    'clouds/Regions/Everything/ALS16_Cloud_reduced_normals_cleared.asc'):
    ((-0.30000000, -0.75000000, 0.0), 0.0),

    # Missing Building
    ('clouds/Regions/Missing Building/ALS14_Cloud_reduced_normals_cleared.asc',
    'clouds/Regions/Missing Building/ALS16_Cloud_reduced_normals_cleared.asc'):
    ((0.15000000, -0.15000000, 0.0), 0.0),


    # ### reference: ALS14 <- DIM aligned ##################################

    # Everything
    ('clouds/Regions/Everything/ALS14_Cloud_reduced_normals_cleared.asc',
    'clouds/Regions/Everything/DSM_Cloud_reduced_normals.asc'):
    ((-0.30000000, 0.45000000, 0.0), 0.0),

    # Missing Building
    ('clouds/Regions/Missing Building/ALS14_Cloud_reduced_normals_cleared.asc',
    'clouds/Regions/Missing Building/DSM_Cloud_reduced_normals.asc'):
    ((0.60000000, 0.0, 0.0), 0.0),

    # ### reference: ALS16_Scan_54 <- ALS16_Scan_55 aligned ##################################

    # Everything
    ('clouds/Regions/Everything/ALS16_Cloud _Scan54_reduced_normals.asc',
    'clouds/Regions/Everything/ALS16_Cloud _Scan55_reduced_normals.asc'):
    ((0.0, 0.0, 0.0), 0.0),

}


# #########################################
# no_translations                       ###
# #########################################

no_translations = {

    # ### reference: ALS16 <- DIM aligned ##################################

    # Color Houses
    ('clouds/Regions/Color Houses/ALS16_Cloud_reduced_normals_cleared.asc',
    'clouds/Regions/Color Houses/DSM_Cloud_reduced_normals.asc'):
    ((-0.0, 0.0, 0.0), 0.0),

    # DIM showcase
    ('clouds/Regions/DIM showcase/ALS16_Cloud_reduced_normals_cleared.asc',
    'clouds/Regions/DIM showcase/DSM_Cloud_reduced_normals.asc'):
    ((-0.0, 0.0, 0.0), 0.0),

    # Everything
    ('clouds/Regions/Everything/ALS16_Cloud_reduced_normals_cleared.asc',
    'clouds/Regions/Everything/DSM_Cloud_reduced_normals.asc'):
    ((-0.0, 0.0, 0.0), 0.0),

    # Field
    ('clouds/Regions/Field/ALS16_Cloud_reduced_normals_cleared.asc',
    'clouds/Regions/Field/DSM_Cloud_reduced_normals.asc'):
    ((-0.0, 0.0, 0.0), 0.0),

    # Forest
    ('clouds/Regions/Forest/ALS16_Cloud_reduced_normals_cleared.asc',
    'clouds/Regions/Forest/DSM_Cloud_reduced_normals.asc'):
    ((-0.0, 0.0, 0.0), 0.0),

    # Missing Building
    ('clouds/Regions/Missing Building/ALS16_Cloud_reduced_normals_cleared.asc',
    'clouds/Regions/Missing Building/DSM_Cloud_reduced_normals.asc'):
    ((-0.0, 0.0, 0.0), 0.0),

    # Road
    ('clouds/Regions/Road/ALS16_Cloud_reduced_normals_cleared.asc',
    'clouds/Regions/Road/DSM_Cloud_reduced_normals.asc'):
    ((-0.0, 0.0, 0.0), 0.0),

    # Xy Tower
    ('clouds/Regions/Xy Tower/ALS16_Cloud_reduced_normals_cleared.asc',
    'clouds/Regions/Xy Tower/DSM_Cloud_reduced_normals.asc'):
    ((-0.0, 0.0, 0.0), 0.0),

    # Xyz Square
    ('clouds/Regions/Xyz Square/ALS16_Cloud_reduced_normals_cleared.asc',
    'clouds/Regions/Xyz Square/DSM_Cloud_reduced_normals.asc'):
    ((-0.0, 0.0, 0.0), 0.0),

    # Xz Hall
    ('clouds/Regions/Xz Hall/ALS16_Cloud_reduced_normals_cleared.asc',
    'clouds/Regions/Xz Hall/DSM_Cloud_reduced_normals.asc'):
    ((-0.0, 0.0, 0.0), 0.0),

    # Yz Houses
    ('clouds/Regions/Yz Houses/ALS16_Cloud_reduced_normals_cleared.asc',
    'clouds/Regions/Yz Houses/DSM_Cloud_reduced_normals.asc'):
    ((-0.0, 0.0, 0.0), 0.0),

    # Yz Street
    ('clouds/Regions/Yz Street/ALS16_Cloud_reduced_normals_cleared.asc',
    'clouds/Regions/Yz Street/DSM_Cloud_reduced_normals.asc'):
    ((-0.0, 0.0, 0.0), 0.0),

    # ### reference: ALS14 <- ALS16 aligned ##################################

    # Everything
    ('clouds/Regions/Everything/ALS14_Cloud_reduced_normals_cleared.asc',
    'clouds/Regions/Everything/ALS16_Cloud_reduced_normals_cleared.asc'):
    ((-0.0, 0.0, 0.0), 0.0),

    # Missing Building
    ('clouds/Regions/Missing Building/ALS14_Cloud_reduced_normals_cleared.asc',
    'clouds/Regions/Missing Building/ALS16_Cloud_reduced_normals_cleared.asc'):
    ((-0.0, 0.0, 0.0), 0.0),


    # ### reference: ALS14 <- DIM aligned ##################################

    # Everything
    ('clouds/Regions/Everything/ALS14_Cloud_reduced_normals_cleared.asc',
    'clouds/Regions/Everything/DSM_Cloud_reduced_normals.asc'):
    ((-0.0, 0.0, 0.0), 0.0),

    # Missing Building
    ('clouds/Regions/Missing Building/ALS14_Cloud_reduced_normals_cleared.asc',
    'clouds/Regions/Missing Building/DSM_Cloud_reduced_normals.asc'):
    ((-0.0, 0.0, 0.0), 0.0),

    # ### reference: ALS16_Scan_54 <- ALS16_Scan_55 aligned ##################################

    # Everything
    ('clouds/Regions/Everything/ALS16_Cloud _Scan54_reduced_normals.asc',
    'clouds/Regions/Everything/ALS16_Cloud _Scan55_reduced_normals.asc'):
    ((-0.0, 0.0, 0.0), 0.0),

}
