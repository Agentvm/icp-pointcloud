'''
Contains reference translations determined by point picking alignment for all extracted regions.
Format: (reference_cloud_path, aligned_cloud_path):(translation_x, translation_y, translation_z, RMS)
'''

translations = {

    # ### ALS16 -> DIM ##################################

    # # Color Houses
    ('clouds/Regions/Color Houses/ALS16_Cloud_reduced_normals_cleared.asc',
    'clouds/Regions/Color Houses/DSM_Cloud_reduced_normals.asc'):
    ((-0.041994094849, 0.044445037842, 0.017503738403), 0.110672),
    #
    # # DIM showcase
    # ('?reference?',
    # 'clouds/Regions/DIM showcase/ALS16_Cloud_reduced_normals_cleared.asc'):
    # ((0.0, 0.0, 0.0),
    # 0.0),
    #
    # ('?reference?',
    # 'clouds/Regions/DIM showcase/DSM_Cloud_reduced_normals.asc'):
    # ((0.0, 0.0, 0.0),
    # 0.0),

    # Everything
    # ('clouds/Regions/Everything/ALS16_Cloud_reduced_normals_cleared.asc',
    # 'clouds/Regions/Everything/DSM_Cloud_reduced_normals.asc'):
    # ((0.0, 0.0, 0.0), 0.0),   # too hard

    # # Field
    # ('?reference?',
    # 'clouds/Regions/Field/ALS16_Cloud_Scan53_reduced_normals.asc'):
    # ((0.0, 0.0, 0.0),
    # 0.0),
    # ('?reference?',
    # 'clouds/Regions/Field/ALS16_Cloud_Scan54_reduced_normals.asc'):
    # ((0.0, 0.0, 0.0),
    # 0.0),
    #
    # # Forest
    # ('?reference?',
    # 'clouds/Regions/Forest/ALS16_Cloud_Scan46_reduced_normals.asc'):
    # ((0.0, 0.0, 0.0),
    # 0.0),
    # ('?reference?',
    # 'clouds/Regions/Forest/ALS16_Cloud_Scan47_reduced_normals.asc'):
    # ((0.0, 0.0, 0.0),
    # 0.0),
    # ('?reference?',
    # 'clouds/Regions/Forest/DSM_Cloud_reduced_normals.asc'):
    # ((0.0, 0.0, 0.0),
    # 0.0),
    #
    # # Missing Building
    # ('?reference?',
    # 'clouds/Regions/Missing Building/ALS14_Cloud_reduced_normals_cleared.asc'):
    # ((0.0, 0.0, 0.0),
    # 0.0),
    # ('?reference?',
    # 'clouds/Regions/Missing Building/ALS16_Cloud_Scan48_reduced_normals.asc'):
    # ((0.0, 0.0, 0.0),
    # 0.0),
    # ('?reference?',
    # 'clouds/Regions/Missing Building/ALS16_Cloud_Scan49_reduced_normals.asc'):
    # ((0.0, 0.0, 0.0),
    # 0.0),
    #
    # # Road
    # ('?reference?',
    # 'clouds/Regions/Road/ALS16_Cloud_Scan54_reduced_normals.asc'):
    # ((0.0, 0.0, 0.0),
    # 0.0),
    # ('?reference?',
    # 'clouds/Regions/Road/ALS16_Cloud_Scan55_reduced_normals.asc'):
    # ((0.0, 0.0, 0.0),
    # 0.0),
    # ('?reference?',
    # 'clouds/Regions/Road/DSM_Cloud_reduced_normals.asc'):
    # ((0.0, 0.0, 0.0),
    # 0.0),
    # ('?reference?',
    #
    # # Test Xy
    # 'clouds/Regions/Test Xy/ALS14_Cloud_reduced_normals_cleared.asc'):
    # ((0.0, 0.0, 0.0),
    # 0.0),
    #
    # # Xy Tower
    # ('?reference?',
    # 'clouds/Regions/Xy Tower/ALS16_Cloud_Scan54_reduced_normals.asc'):
    # ((0.0, 0.0, 0.0),
    # 0.0),
    # ('?reference?',
    # 'clouds/Regions/Xy Tower/ALS16_Cloud_Scan55_reduced_normals.asc'):
    # ((0.0, 0.0, 0.0),
    # 0.0),
    # ('?reference?',
    # 'clouds/Regions/Xy Tower/DSM_Cloud_reduced_normals.asc'):
    # ((0.0, 0.0, 0.0),
    # 0.0),
    #
    # # Xyz Square
    # ('?reference?',
    # 'clouds/Regions/Xyz Square/ALS16_Cloud_Scan54_reduced_normals.asc'):
    # ((0.0, 0.0, 0.0),
    # 0.0),
    # ('?reference?',
    # 'clouds/Regions/Xyz Square/ALS16_Cloud_Scan55_reduced_normals.asc'):
    # ((0.0, 0.0, 0.0),
    # 0.0),
    # ('?reference?',
    # 'clouds/Regions/Xyz Square/DSM_Cloud_reduced_normals.asc'):
    # ((0.0, 0.0, 0.0),
    # 0.0),
    #
    # # Xz Hall
    # ('?reference?',
    # 'clouds/Regions/Xz Hall/ALS16_Cloud_Scan53_reduced_normals.asc'):
    # ((0.0, 0.0, 0.0),
    # 0.0),
    # ('?reference?',
    # 'clouds/Regions/Xz Hall/ALS16_Cloud_Scan54_reduced_normals.asc'):
    # ((0.0, 0.0, 0.0),
    # 0.0),
    # ('?reference?',
    # 'clouds/Regions/Xz Hall/DSM_Cloud_reduced_normals.asc'):
    # ((0.0, 0.0, 0.0),
    # 0.0),
    #
    # # Yz Houses
    # ('?reference?',
    # 'clouds/Regions/Yz Houses/ALS16_Cloud_reduced_normals_cleared.asc'):
    # ((0.0, 0.0, 0.0),
    # 0.0),
    #
    # # Yz Street
    # ('?reference?',
    # 'clouds/Regions/Yz Street/ALS16_Cloud_reduced_normals_cleared.asc'):
    # ((0.0, 0.0, 0.0),
    # 0.0),
    # ('?reference?',
    # 'clouds/Regions/Yz Street/DSM_Cloud_reduced_normals.asc'):
    # ((0.0, 0.0, 0.0),
    # 0.0

    # ### ALS14 -> ALS16 ##################################

    # Everything


    # Missing Building


    # ### ALS14 -> DIM ##################################

    # Everything
    ('clouds/Regions/Everything/ALS14_Cloud_reduced_normals_cleared.asc',
    'clouds/Regions/Everything/DSM_Cloud_reduced_normals.asc'):
    ((0.015129089355, -0.034282684326, 0.209009647369), 0.0862935),

    # Missing Building
    ('clouds/Regions/Missing Building/ALS14_Cloud_reduced_normals_cleared.asc',
    'clouds/Regions/Missing Building/DSM_Cloud_reduced_normals.asc'):
    ((-0.039558410645, -0.025497436523, 0.008216857910), 0.282216),

    # ### ALS16_Scan -> ALS16_Scan ##################################

    # Everything
    ('clouds/Regions/Everything/ALS16_Cloud _Scan54_reduced_normals.asc',
    'clouds/Regions/Everything/ALS16_Cloud _Scan55_reduced_normals.asc'):
    ((0.005805969238, -0.011207580566, -0.018814086914), 0.0452853),   # not yet done

}
