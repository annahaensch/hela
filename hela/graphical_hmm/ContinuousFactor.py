import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

from pgmpy.factors.base import BaseFactor
import numbers

class ContinuousFactor(BaseFactor):
    """
    Defines the weight and covariance matrix for an edge in an FHMM graph.

    This class is very similar to pgmpy.factors.discrete.TabularCPD and is
    used to define edges in the graphical model as the respective weight or
    mean matrix and covariance matrix.  
    ----------
    variable: int, string (any hashable python object)
        The variable whose distribution is defined by the weight matrix.
    variable_card: integer
        cardinality of variable
    weights: 2d array, 2d list or 2d tuple
        weight matrix for evidence
    covariance: 2d array, 2d list or 2d tuple
        covariance matrix for the evidence
    evidence: array-like
        evidences(if any) w.r.t. the linear gaussian is defined,
        typically a latent node.
    evidence_card: integer, array-like
        cardinality of evidences (if any)
    """

    def __init__(
        self,
        variable,
        variable_card,
        weights,
        covariance,
        evidence=None,
        evidence_card=None,
        state_names={},
    ):

        self.variable = variable
        self.variable_card = None

        variables = [variable]
        if not isinstance(variable_card, numbers.Integral):
            raise TypeError("Event cardinality must be an integer")
        self.variable_card = variable_card

        cardinality = [variable_card]
        if evidence_card is not None:
            if isinstance(evidence_card, numbers.Real):
                raise TypeError("Evidence card must be a list of numbers")
            cardinality.extend(evidence_card)
            
        if evidence is not None:
            if isinstance(evidence, str):
                raise TypeError("Evidence must be list, tuple or array of strings.")
            variables.extend(evidence)
            if not len(evidence_card) == len(evidence):
                raise ValueError(
                    "Length of evidence_card doesn't match length of evidence"
                )

        weights = np.array(weights)
        if weights.ndim != 2:
            raise TypeError("Weights must be a 2D list/array")
        covariance = np.array(covariance)
        if covariance.ndim != 2:
            raise TypeError("Covariance must be a 2D list/array")
            
        expected_shape = (variable_card, np.product(evidence_card))
        if weights.shape != expected_shape:
            raise ValueError(
                f"values must be of shape {expected_shape}. Got shape: {weights.shape}"
            )
            
        self.variables = list(variables)
        self.cardinality = np.array(cardinality, dtype=int)
        self.weights = weights.flatten("C").reshape(self.cardinality)
        self.covariance = covariance

        # dictionary of {node:cardinality} corresponding to the
        # edge the factor is defined for
        self.state_names = {
                var: list(range(int(cardinality[index])))
                for index, var in enumerate(variables)
            }
        self.name_to_no = {
                var: {i: i for i in range(int(cardinality[index]))}
                for index, var in enumerate(variables)
            }

    def get_mean_value(self, hs):
        #TODO(isalju)
        pass

    def copy(self):
        """
        Returns a copy of the ContinuousFactor object.
        """
        evidence = self.variables[1:] if len(self.variables) > 1 else None
        evidence_card = self.cardinality[1:] if len(self.variables) > 1 else None
        return ContinuousFactor(
            self.variable,
            self.variable_card,
            self.weights,
            self.covariance,
            evidence,
            evidence_card,
            state_names=self.state_names.copy(),
        )


