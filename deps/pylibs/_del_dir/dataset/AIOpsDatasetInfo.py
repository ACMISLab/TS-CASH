import pandas as pd

"""
Get a AIOps KPIs info

af = AIOpsDatasetInfo()
# Get all kpi info
print(af.get_all_kpis())

# Get a kpi info by KPIID
print(af.get_kpi_info_by_kpi_id("ffb82d38-5f00-37db-abc0-5d2e4e4cb6aa"))

# Output: 
# n_anomaly   n_all   anomaly_rate                                kpi_id
# 9677        240180  4.03          ffb82d38-5f00-37db-abc0-5d2e4e4cb6aa
"""


class AIOpsDatasetInfo:
    KPIS_INFO = [{'n_anomaly': 2276,
                  'n_all': 295385,
                  'anomaly_rate': 0.77,
                  'kpi_id': '05f10d3a-239c-3bef-9bdc-a2feeb0037aa'}
        , {'n_anomaly': 84,
           'n_all': 17568,
           'anomaly_rate': 0.48,
           'kpi_id': '0efb375b-b902-3661-ab23-9a0bb799f4e3'}
        , {'n_anomaly': 1697,
           'n_all': 295410,
           'anomaly_rate': 0.5700000000000001,
           'kpi_id': '1c6d7a26-1f1a-3321-bb4d-7a9d969ec8f0'}
        , {'n_anomaly': 320,
           'n_all': 17568,
           'anomaly_rate': 1.82,
           'kpi_id': '301c70d8-1630-35ac-8f96-bc1b6f4359ea'}
        , {'n_anomaly': 3303,
           'n_all': 295414,
           'anomaly_rate': 1.1199999999999999,
           'kpi_id': '42d6616d-c9c5-370a-a8ba-17ead74f3114'}
        , {'n_anomaly': 6548,
           'n_all': 218700,
           'anomaly_rate': 2.9899999999999998,
           'kpi_id': '43115f2a-baeb-3b01-96f7-4ea14188343c'}
        , {'n_anomaly': 11508,
           'n_all': 240612,
           'anomaly_rate': 4.78,
           'kpi_id': '431a8542-c468-3988-a508-3afd06a218da'}
        , {'n_anomaly': 9981,
           'n_all': 240242,
           'anomaly_rate': 4.15,
           'kpi_id': '4d2af31a-9916-3d9f-8a8e-8a268a48c095'}
        , {'n_anomaly': 170,
           'n_all': 16482,
           'anomaly_rate': 1.03,
           'kpi_id': '54350a12-7a9d-3ca8-b81f-f886b9d156fd'}
        , {'n_anomaly': 9958,
           'n_all': 295337,
           'anomaly_rate': 3.37,
           'kpi_id': '55f8b8b8-b659-38df-b3df-e4a5a8a54bc9'}
        , {'n_anomaly': 151,
           'n_all': 240466,
           'anomaly_rate': 0.06,
           'kpi_id': '57051487-3a40-3828-9084-a12f7f23ee38'}
        , {'n_anomaly': 9491,
           'n_all': 239886,
           'anomaly_rate': 3.9600000000000004,
           'kpi_id': '6a757df4-95e5-3357-8406-165e2bd49360'}
        , {'n_anomaly': 1121,
           'n_all': 295361,
           'anomaly_rate': 0.38,
           'kpi_id': '6d1114ae-be04-3c46-b5aa-be1a003a57cd'}
        , {'n_anomaly': 8050,
           'n_all': 294283,
           'anomaly_rate': 2.74,
           'kpi_id': '6efa3a07-4544-34a0-b921-a155bd1a05e8'}
        , {'n_anomaly': 178,
           'n_all': 237417,
           'anomaly_rate': 0.06999999999999999,
           'kpi_id': '7103fa0f-cac4-314f-addc-866190247439'}
        , {'n_anomaly': 1911,
           'n_all': 295351,
           'anomaly_rate': 0.65,
           'kpi_id': '847e8ecc-f8d2-3a93-9107-f367a0aab37d'}
        , {'n_anomaly': 1367,
           'n_all': 295361,
           'anomaly_rate': 0.45999999999999996,
           'kpi_id': '8723f0fb-eaef-32e6-b372-6034c9c04b80'}
        , {'n_anomaly': 7084,
           'n_all': 218710,
           'anomaly_rate': 3.2399999999999998,
           'kpi_id': '9c639a46-34c8-39bc-aaf0-9144b37adfc8'}
        , {'n_anomaly': 7922,
           'n_all': 217782,
           'anomaly_rate': 3.64,
           'kpi_id': 'a07ac296-de40-3a7c-8df3-91f642cc14d0'}
        , {'n_anomaly': 476,
           'n_all': 16441,
           'anomaly_rate': 2.9000000000000004,
           'kpi_id': 'a8c06b47-cc41-3738-9110-12df0ee4c721'}
        , {'n_anomaly': 365,
           'n_all': 21918,
           'anomaly_rate': 1.67,
           'kpi_id': 'ab216663-dcc2-3a24-b1ee-2c3e550e06c9'}
        , {'n_anomaly': 3120,
           'n_all': 295409,
           'anomaly_rate': 1.06,
           'kpi_id': 'adb2fde9-8589-3f5b-a410-5fe14386c7af'}
        , {'n_anomaly': 9291,
           'n_all': 295258,
           'anomaly_rate': 3.15,
           'kpi_id': 'ba5f3328-9f3f-3ff5-a683-84437d16d554'}
        , {'n_anomaly': 67,
           'n_all': 17568,
           'anomaly_rate': 0.38,
           'kpi_id': 'c02607e8-7399-3dde-9d28-8a8da5e5d251'}
        , {'n_anomaly': 1580,
           'n_all': 295414,
           'anomaly_rate': 0.53,
           'kpi_id': 'c69a50cf-ee03-3bd7-831e-407d36c7ee91'}
        , {'n_anomaly': 16113,
           'n_all': 214884,
           'anomaly_rate': 7.5,
           'kpi_id': 'da10a69f-d836-3baa-ad40-3e548ecf1fbd'}
        , {'n_anomaly': 209,
           'n_all': 17568,
           'anomaly_rate': 1.1900000000000002,
           'kpi_id': 'e0747cad-8dc8-38a9-a9ab-855b61f5551d'}
        , {'n_anomaly': 10096,
           'n_all': 240938,
           'anomaly_rate': 4.19,
           'kpi_id': 'f0932edd-6400-3e63-9559-0a9860a1baa9'}
        , {'n_anomaly': 9677,
           'n_all': 240180,
           'anomaly_rate': 4.03,
           'kpi_id': 'ffb82d38-5f00-37db-abc0-5d2e4e4cb6aa'}]
    N_ANOMALY = "n_anomaly"
    N_COUNT = "n_all"
    ANOMALY_RATE = "anomaly_rate"
    KPI_ID = "kpi_id"

    def __init__(self):
        self.kpis_info = None

    def get_all_kpis(self) -> pd.DataFrame:
        if self.kpis_info is None:
            self.kpis_info = pd.DataFrame(self.KPIS_INFO)
        return self.kpis_info

    def get_kpi_info_by_kpi_id(self, kpi_id):
        all_kpis = self.get_all_kpis()
        return all_kpis.loc[all_kpis[self.KPI_ID] == kpi_id, :]

    def n_points(self, kpi_id):
        """
        Return the number of points (int number) of the given kpi
        Parameters
        ----------
        kpi_id :

        Returns
        -------

        """
        return self.get_kpi_info_by_kpi_id(kpi_id)[AIOpsDatasetInfo.N_COUNT].iloc[0]


if __name__ == '__main__':
    af = AIOpsDatasetInfo()
    print(af.get_all_kpis())
    print(af.get_kpi_info_by_kpi_id("ffb82d38-5f00-37db-abc0-5d2e4e4cb6aa"))
    # n_anomaly   n_all   anomaly_rate                                kpi_id
    # 9677        240180  4.03          ffb82d38-5f00-37db-abc0-5d2e4e4cb6aa
