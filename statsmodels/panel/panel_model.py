from statsmodels.base.model import LikelihoodModel
from statsmodels.panel.base.data import handle_panel_data


class PanelModel(LikelihoodModel):
    def __init__(self, y, X, time=None, panel=None, missing='none',
                       hasconst=None):
        #TODO: make time-series stuff more modular so we can check time
        # if need be -
        # check_dates(time)
        super(PanelModel, self).__init__(y, X, missing=missing,
                                         time=time, panel=panel)

    def _handle_data(self, y, X, missing, hasconst, **kwargs):
        return handle_panel_data(y, X, missing, hasconst, **kwargs)