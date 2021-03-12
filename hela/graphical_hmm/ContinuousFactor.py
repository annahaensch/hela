import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

from pgmpy.factors.base import BaseFactor
import numbers

class ContinuousFactor(BaseFactor):
    """
    Defines the weight and covariance matrix for an edge in an FHMM graph
    ----------
    variable: int, string (any hashable python object)
        The variable whose distribution is defined by the weight matrix.
    variable_card: integer
        cardinality of variable
    values: 2d array, 2d list or 2d tuple
        weight matrix
    evidence: array-like
        evidences(if any) w.r.t. which distribution is defined
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