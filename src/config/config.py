import os


currentPath = os.path.dirname(os.path.abspath(__file__))
dataPath = (currentPath + '/../../data')

paths = dict (
    CRSPretPath = (dataPath + '/external/crspmret.csv'),
    CRSPinfoPath=(dataPath+'/external/crspminfo.csv'),
    FFPath = (dataPath + '/external/FamaFrenchData.csv'),
    SignalsPath = (dataPath + '/external/signed_predictors_all_wide.csv'),
    ProcessedDataPath = (dataPath + '/processed'),
    modelsPath = (currentPath + '/../../saved/models')
    )

ForcePreProcessing = False
ForceTraining = True