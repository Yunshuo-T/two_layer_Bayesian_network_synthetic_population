import numpy as np
from src.utils import encoding

class Validation:
    """
    Arguments:
        syn_w(np.ndarray): The weights of synthetic populations.
        marginals(np.ndarray): The target marginals. 
    
    """
    def __init__(
            self, 
            syn_w: np.ndarray, 
            marginals: np.ndarray
    ): 
        self.w= syn_w 
        self.o_m = marginals
        self.e_m = np.sum(self.w,axis=0)

    def root_mean_squared_error(self):
        """RMSE for marginal distributions."""
        rms = np.sqrt(np.mean(np.square(self.o_m - self.e_m)))
        return rms
    
    def standardized_rmse(self):
        """
        Standardized RMSE for marginal distributions.\n
        Knudsen, D. C., & Fotheringham, A. S. (1986). 
        Matrix Comparison, Goodness-of-Fit, and Spatial Interaction Modeling. 
        _International Regional Science Review_, _10_(2), 127–147. 
        (https://doi.org/10.1177/016001768601000203)
        Returns:
            _type_: _description_
        """
        rms = self.root_mean_squared_error()
        srmse = rms / np.mean(self.o_m)
        return srmse
    
    def total_abs_error(self):
        """
        Total absolute error RMSE for marginal distributions.

        Returns:
            _type_: _description_
        """
        tae = np.sum(np.abs(self.o_m - self.e_m))
        return tae
    
    def relative_abs_error(self):
        """
        Relative absolute error for marginal distributions.
        
        Returns:
            _type_: _description_
        """
        tae = self.total_abs_error()
        rae = tae / np.sum(np.abs(self.o_m-np.mean(self.e_m)))
        return rae
    
    def mean_abs_error(self):
        mae = np.mean(np.abs(self.o_m - self.e_m))
        return mae
    
    
    def r_square_pearson(self):
        """
        Squared Pearson correlation

        Returns:
            _type_: _description_
        """
        mean_e = np.mean(self.e_m)
        mean_o = np.mean(self.o_m)
        nominator = np.dot((self.o_m - mean_o),(self.e_m - mean_e))
        sum_e_2 = np.sum((self.e_m - mean_e)**2)
        sum_o_2 = np.sum((self.o_m - mean_o)**2)
        denominator = np.sqrt(sum_o_2*sum_e_2)
        if denominator == 0:
            return 0
        return (nominator / denominator)**2
    
    
    def sum_square_modified_z_score(self):
        # TODO The zero cell problem
        """
        Williamson, P., Birkin, M., & Rees, P. H. (1998). 
        The estimation of population microdata by using data from small area statistics and samples of anonymised records. 
        Environment & Planning A, 30(5), 785–816. https://doi.org/10.1068/a300785
        Returns:
            _type_: _description_
        """
        total = np.sum(self.o_m)
        r = self.e_m / total
        p = self.o_m / total
        m_zscore = (r-p) / np.sqrt(p*(1-p)/total)
        return np.sum(m_zscore**2)

    @classmethod
    def _get_combination_counts(cls, synpop_df, origin_df, attributes, normalize=False):
        """Helper to group, encode, and align joint distributions for SRMSE calculations."""
        _, attr_to_code = encoding.cate_codes(origin_df[attributes], attributes)
        synpop_df = encoding.map_dataframe(attr_to_code, synpop_df)
        origin_df = encoding.map_dataframe(attr_to_code, origin_df)

        origin_counts = origin_df.groupby(attributes, observed=True).size()
        
        if 'integer' in synpop_df.columns:
            syn_counts = synpop_df.groupby(attributes, observed=True)['integer'].sum()
        else:
            syn_counts = synpop_df.groupby(attributes, observed=True).size()

        if normalize:
            origin_counts = origin_counts / len(origin_df)
            syn_counts = syn_counts / syn_counts.sum()
            
        origin_counts = origin_counts[origin_counts > 0]
        diff = origin_counts.sub(syn_counts, fill_value=0)
        return diff, origin_counts
    
    @classmethod
    def combination_srmse(cls, synpop_df, origin_df, attributes, normalize: bool = False):
        """Computes absolute combination SRMSE and difference distribution."""
        diff, origin_counts = cls._get_combination_counts(synpop_df, origin_df, attributes, normalize=normalize)
        
        rmse = np.sqrt((diff**2).sum() / len(diff))
        mean_origin = origin_counts.sum() / len(diff)
        srmse = rmse / mean_origin if mean_origin > 0 else 0

        return round(float(srmse), 4), diff  
    

def compute_metrics(name, h_weights, h_m_ar, p_weights, p_m_ar, h_filename, p_filename, is_converged):
    """Computes RMSE and Z-scores for both household and person levels."""
    h_valida = Validation(h_weights.T, h_m_ar)
    p_valida = Validation(p_weights.T, p_m_ar)
    
    return {
        name: {
            'filename_h': h_filename,
            'filename_p': p_filename,
            'h_rmse': h_valida.root_mean_squared_error(),
            'p_rmse': p_valida.root_mean_squared_error(),
            'h_zscore': h_valida.sum_square_modified_z_score(),
            'p_zscore': p_valida.sum_square_modified_z_score(),
            'is_converged': is_converged
        }
    }