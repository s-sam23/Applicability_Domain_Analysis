# import cupy as cp
# import numpy as np
# from rdkit import Chem
# from rdkit.ML.Descriptors import MoleculeDescriptors
# from rdkit.Chem import Descriptors
# from scipy.spatial import ConvexHull, Delaunay, QhullError
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import Ridge
# # from sklearn.decomposition import PCA  # Add PCA for dimensionality reduction
# from umap import UMAP
# import pandas as pd
# from concurrent.futures import ThreadPoolExecutor


# class CombinedApplicabilityDomain:
#     def __init__(self, training_smiles, target=None, model=None, num_workers=32, pca_components=20):
#         """
#         Initialize the combined Applicability Domain class, optimized for DGX, with PCA for dimensionality reduction.
#         :param training_smiles: List of SMILES strings used for training
#         :param target: Target values for leverage-based AD (optional)
#         :param model: Optional model for leverage-based AD (e.g., cuML Ridge)
#         :param num_workers: Number of threads to use for parallel processing on CPU (DGX optimized)
#         :param pca_components: Number of components to retain during PCA dimensionality reduction.
#         """
#         self.descriptor_names = [desc[0] for desc in Descriptors._descList if desc[0] != 'Ipc']
#         self.num_workers = num_workers

#         # Compute descriptors once and reuse across methods, using GPU arrays for post-processing
#         self.training_descriptors = cp.asarray(self._compute_descriptors_parallel(training_smiles))
#         self.training_descriptors = cp.asarray(self._clean_descriptors(self.training_descriptors))
#         # Leverage-based AD setup
#         self.scaler = StandardScaler()
#         self.training_descriptors = cp.asarray(self.scaler.fit_transform(cp.asnumpy(self.training_descriptors)))

#         # Apply PCA for dimensionality reduction
#         self.pca = UMAP(n_components=pca_components)
#         self.training_descriptors = cp.asarray(self.pca.fit_transform(cp.asnumpy(self.training_descriptors)))

#         # Range-based AD setup (using CuPy arrays)
#         self.min_values = cp.min(self.training_descriptors, axis=0)
#         self.max_values = cp.max(self.training_descriptors, axis=0)

#         # Convex Hull AD setup (on CPU as ConvexHull doesn't support GPU)
#         try:
#             self.hull = ConvexHull(cp.asnumpy(self.training_descriptors))  # Convert back to NumPy for ConvexHull
#             self.delaunay = Delaunay(cp.asnumpy(self.training_descriptors))
#         except QhullError as e:
#             print(f"ConvexHull failed: {e}")
#             self.hull = None  # Handle gracefully if Convex Hull fails

#   # Scale on CPU

#         # Use cuML Ridge regression for GPU acceleration
#         self.target = target
#         self.model = Ridge(alpha=1.0).fit(self.training_descriptors, cp.asarray(self.target)) if target is not None and model is None else model

#         # Compute Hat matrix on GPU and leverage threshold
#         self.hat_matrix = self._compute_hat_matrix()
#         self.leverage_threshold = 3 * (self.training_descriptors.shape[1] + 1) / self.training_descriptors.shape[0]

#     def _compute_descriptors_parallel(self, smiles_list):
#         """
#         Parallel computation of molecular descriptors using multi-threading for fast descriptor generation.
#         """
#         calc = MoleculeDescriptors.MolecularDescriptorCalculator(self.descriptor_names)
#         with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
#             descriptors = list(executor.map(self._calc_descriptor_for_smile, smiles_list))
#         return np.array([desc for desc in descriptors if desc is not None])

#     def _calc_descriptor_for_smile(self, smile):
#         """
#         Helper function to compute descriptors for a single SMILE.
#         """
#         mol = Chem.MolFromSmiles(smile)
#         if mol is not None:
#             calc = MoleculeDescriptors.MolecularDescriptorCalculator(self.descriptor_names)
#             return calc.CalcDescriptors(mol)
#         return None

#     def _clean_descriptors(self, descriptors):
#         """
#         Clean NaN/Inf values from descriptors, using CuPy for fast operations.
#         """
#         # Convert to NumPy before passing to pandas DataFrame
#         descriptors_cpu = descriptors if isinstance(descriptors, np.ndarray) else cp.asnumpy(descriptors)
        
#         df = pd.DataFrame(descriptors_cpu)
#         df.replace([np.inf, -np.inf], np.nan, inplace=True)
#         df.fillna(0, inplace=True)
        
#         # Return the cleaned data, and explicitly convert it back to CuPy
#         return df.to_numpy(dtype=np.float32)

#     def _compute_hat_matrix(self):
#         """
#         Compute the Hat matrix for leverage calculation, accelerated with CuPy.
#         """
#         X = self.training_descriptors_scaled
#         H = X @ cp.linalg.pinv(X.T @ X) @ X.T  # Compute using CuPy pseudo-inverse
#         return cp.diag(H)

#     def _compute_hat_matrix_for_test(self, test_descriptors):
#         """
#         Compute leverage for the test descriptors, using CuPy for fast matrix operations.
#         """
#         X_train = self.training_descriptors_scaled
#         H_test = cp.diag(test_descriptors @ cp.linalg.pinv(X_train.T @ X_train) @ test_descriptors.T)
#         return H_test

#     # Method 1: Descriptor Range AD
#     def is_within_range_domain(self, test_smiles):
#         """
#         Check if test SMILES are within the applicability domain based on descriptor ranges.
#         """
#         test_descriptors = cp.asarray(self._compute_descriptors_parallel(test_smiles))
#         test_descriptors = cp.asarray(self._clean_descriptors(test_descriptors))
#         test_descriptors = cp.asarray(self.scaler.transform(test_descriptors))

#         # Apply PCA to test descriptors
#         test_descriptors = cp.asarray(self.pca.transform(cp.asnumpy(test_descriptors)))

#         within_domain = cp.all((test_descriptors >= self.min_values) & (test_descriptors <= self.max_values), axis=1)
#         return within_domain

#     # Method 2: Convex Hull AD
#     def is_within_convex_hull_domain(self, test_smiles):
#         """
#         Check if test SMILES are within the applicability domain based on Convex Hull (CPU-bound for now).
#         """
#         test_descriptors = cp.asnumpy(self._compute_descriptors_parallel(test_smiles))  # Convert to NumPy for ConvexHull
#         test_descriptors = self._clean_descriptors(test_descriptors)
#         test_descriptors = cp.asarray(self.scaler.transform(test_descriptors))

#         # Apply PCA to test descriptors
#         test_descriptors = self.pca.transform(test_descriptors)

#         if self.hull:
#             return self.delaunay.find_simplex(test_descriptors) >= 0
#         else:
#             return np.array([False] * len(test_smiles))  # If Convex Hull failed, return False

#     # Method 3: Leverage-based AD
#     def is_within_leverage_domain(self, test_smiles):
#         """
#         Check if test SMILES are within the applicability domain based on leverage.
#         """
#         test_descriptors = cp.asarray(self._compute_descriptors_parallel(test_smiles))
#         test_descriptors = cp.asarray(self._clean_descriptors(test_descriptors))
#         test_descriptors = cp.asarray(self.scaler.transform(test_descriptors))

#         # Apply PCA to test descriptors
#         test_descriptors_scaled = cp.asarray(self.scaler.transform(cp.asnumpy(self.pca.transform(cp.asnumpy(test_descriptors)))))

#         H_test = self._compute_hat_matrix_for_test(test_descriptors_scaled)
#         return H_test <= self.leverage_threshold



 
 

import cupy as cp
import numpy as np
from rdkit import Chem
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors
from scipy.spatial import ConvexHull, Delaunay, QhullError
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


# from umap import UMAP
import umap.umap_ as UMAP


import pandas as pd
from concurrent.futures import ThreadPoolExecutor


class CombinedApplicabilityDomain:
    def __init__(self, training_smiles, target=None, model=None, num_workers=32, umap_components=20):
        """
        Initialize the combined Applicability Domain class, optimized for DGX, with UMAP for dimensionality reduction.
        :param training_smiles: List of SMILES strings used for training
        :param target: Target values for leverage-based AD (optional)
        :param model: Optional model for leverage-based AD (e.g., Ridge)
        :param num_workers: Number of threads to use for parallel processing on CPU (DGX optimized)
        :param umap_components: Number of components to retain during UMAP dimensionality reduction.
        """
        self.descriptor_names = [desc[0] for desc in Descriptors._descList if desc[0] != 'Ipc']
        self.num_workers = num_workers

        # Compute descriptors once and reuse across methods, using GPU arrays for post-processing
        training_descriptors = self._compute_descriptors_parallel(training_smiles)
        cleaned_descriptors = self._clean_descriptors(training_descriptors)

        # Standard scaling
        self.scaler = StandardScaler()
        scaled_descriptors = self.scaler.fit_transform(cleaned_descriptors)

        # Apply UMAP for dimensionality reduction
        self.umap = UMAP.UMAP(n_components=umap_components)
        reduced_descriptors = self.umap.fit_transform(scaled_descriptors)

        # Move to GPU for CuPy-based operations
        self.training_descriptors = cp.asarray(reduced_descriptors)

        # Range-based AD setup (using CuPy arrays)
        self.min_values = cp.min(self.training_descriptors, axis=0)
        self.max_values = cp.max(self.training_descriptors, axis=0)

        # Convex Hull AD setup (on CPU as ConvexHull doesn't support GPU)
        try:
            self.hull = ConvexHull(cleaned_descriptors)  # ConvexHull requires NumPy
            self.delaunay = Delaunay(cleaned_descriptors)
        except QhullError as e:
            print(f"ConvexHull failed: {e}")
            self.hull = None  # Handle gracefully if Convex Hull fails

        # Ridge regression (target should be on GPU)
        if target is not None and model is None:
            self.target = cp.asarray(target)
            self.model = Ridge(alpha=1.0).fit(self.training_descriptors.get(), self.target.get())
        else:
            self.model = model

        # Compute Hat matrix and leverage threshold
        self.hat_matrix = self._compute_hat_matrix()
        self.leverage_threshold = 3 * (self.training_descriptors.shape[1] + 1) / self.training_descriptors.shape[0]

    def _compute_descriptors_parallel(self, smiles_list):
        """
        Parallel computation of molecular descriptors using multi-threading for fast descriptor generation.
        """
        calc = MoleculeDescriptors.MolecularDescriptorCalculator(self.descriptor_names)
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            descriptors = list(executor.map(self._calc_descriptor_for_smile, smiles_list))
        return np.array([desc for desc in descriptors if desc is not None])

    def _calc_descriptor_for_smile(self, smile):
        """
        Helper function to compute descriptors for a single SMILE.
        """
        mol = Chem.MolFromSmiles(smile)
        if mol:
            calc = MoleculeDescriptors.MolecularDescriptorCalculator(self.descriptor_names)
            return calc.CalcDescriptors(mol)
        return None

    def _clean_descriptors(self, descriptors):
        """
        Clean NaN/Inf values from descriptors.
        """
        df = pd.DataFrame(descriptors)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)
        return df.to_numpy(dtype=np.float32)

    def _compute_hat_matrix(self):
        """
        Compute the Hat matrix for leverage calculation.
        """
        X = self.training_descriptors
        XTX_inv = cp.linalg.pinv(X.T @ X)  # Compute pseudo-inverse once
        H = X @ XTX_inv @ X.T
        return cp.diag(H)

    def _compute_hat_matrix_for_test(self, test_descriptors):
        """
        Compute leverage for the test descriptors.
        """
        X_train = self.training_descriptors
        XTX_inv = cp.linalg.pinv(X_train.T @ X_train)  # Reuse precomputed pinv
        H_test = cp.diag(test_descriptors @ XTX_inv @ test_descriptors.T)
        return H_test

    # Method 1: Descriptor Range AD
    def is_within_range_domain(self, test_smiles):
        """
        Check if test SMILES are within the applicability domain based on descriptor ranges.
        """
        test_descriptors = self._compute_descriptors_parallel(test_smiles)
        cleaned_descriptors = self._clean_descriptors(test_descriptors)

        # Apply UMAP and scale
        test_descriptors = self.scaler.transform(cleaned_descriptors)
        test_descriptors = self.umap.transform(test_descriptors)

        test_descriptors_gpu = cp.asarray(test_descriptors)

        # Check if test descriptors are within the min-max range
        within_domain = cp.all((test_descriptors_gpu >= self.min_values) & (test_descriptors_gpu <= self.max_values), axis=1)
        return within_domain

    # Method 2: Convex Hull AD
    def is_within_convex_hull_domain(self, test_smiles):
        """
        Check if test SMILES are within the applicability domain based on Convex Hull (CPU-bound for now).
        """
        test_descriptors = self._compute_descriptors_parallel(test_smiles)
        cleaned_descriptors = self._clean_descriptors(test_descriptors)

        # Apply UMAP and scale
        test_descriptors = self.scaler.transform(cleaned_descriptors)
        test_descriptors = self.umap.transform(test_descriptors)

        # Check if Convex Hull exists and apply Delaunay
        if self.hull:
            return self.delaunay.find_simplex(test_descriptors) >= 0
        else:
            return np.array([False] * len(test_smiles))  # If Convex Hull failed, return False

    # Method 3: Leverage-based AD
    def is_within_leverage_domain(self, test_smiles):
        """
        Check if test SMILES are within the applicability domain based on leverage.
        """
        test_descriptors = self._compute_descriptors_parallel(test_smiles)
        cleaned_descriptors = self._clean_descriptors(test_descriptors)

        # Apply UMAP and scale
        test_descriptors = self.scaler.transform(cleaned_descriptors)
        test_descriptors = self.umap.transform(test_descriptors)

        # Move to GPU for computation
        test_descriptors_gpu = cp.asarray(test_descriptors)

        # Compute the Hat matrix for the test data
        H_test = self._compute_hat_matrix_for_test(test_descriptors_gpu)
        return H_test <= self.leverage_threshold
