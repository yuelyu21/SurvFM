"""Show the common estimator interface across local backbones."""

from survfm import SurvFMRMSTRegressor, list_backbones


for item in list_backbones():
    print(f"{item.name:18} {item.access:12} install extra: {item.install_extra}")

# Every registered backbone uses the same outer API:
# model = SurvFMRMSTRegressor(backbone="tabpfn", tau=365)
# model = SurvFMRMSTRegressor(backbone="tabicl", tau=365)
# model = SurvFMRMSTRegressor(backbone="tabdpt", tau=365)
# model = SurvFMRMSTRegressor(backbone="tabh2o", tau=365)
# model = SurvFMRMSTRegressor(backbone="mitra", tau=365)
