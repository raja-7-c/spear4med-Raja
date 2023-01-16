from typing import Any, Callable, List, Mapping, Optional

from ..data_types import DataPoint
from ..preprocess import BasePreprocessor




class LabelingFunction:
    """Base class for labeling function

    Args:
        name (str): name for this LF object
        f (Callable[..., int]): core function which labels the input
        pre (Optional[List[BasePreprocessor]], optional): Preprocessors to apply on input before labeling. Defaults to None.
        
    """    
    def __init__(
        self,
        name: str,
        f: Callable[..., int],                              
        pre: Optional[List[BasePreprocessor]] = None,
        
    ) -> None:
        """Instatiates LabelingFunction class object
        """
        self.name = name
        self._f = f
        self._pre = pre or []
       

    def _preprocess_data_point(self, x: DataPoint) -> DataPoint:
        """Preprocesses input by applying each preprocessing function in succession

        Args:
            x (DataPoint): Single datapoint

        Raises:
            ValueError: When a preprocessing function returns None

        Returns:
            DataPoint: Preprocessed datapoint
        """
        for preprocessor in self._pre:
            x = preprocessor(x)
            if x is None:
                raise ValueError("Preprocessor should not return None")
        return x

    def __call__(self, x: DataPoint):                                                                           # -> (Enum, float)
        """Applies core labeling function and continuous scorer on datapoint and returns label and confidence

        Args:
            x (DataPoint): Datapoint 

        Returns:
            (Enum, float): Label enum object and confidence for the datapoint

        """
        x = self._preprocess_data_point(x)                                   
        
    def __repr__(self) -> str:
        """Represents class object as string

        Returns:
            str: string representation of the class object
        """
        preprocessor_str = f", Preprocessors: {self._pre}"
        return f"{type(self).__name__} {self.name}{preprocessor_str}"


class roi_selection:
    """Decorator class for a labeling function
    
    Args:
        name (Optional[str], optional): Name for this labeling function. Defaults to None.
        pre (Optional[List[BasePreprocessor]], optional): Preprocessors to apply on input before labeling . Defaults to None.
       

    Raises:
        ValueError: If the decorator is missing parantheses
    """    
    def __init__(
        self,
        name: Optional[str] = None,
        pre: Optional[List[BasePreprocessor]] = None,
        
    ) -> None:
        """Instatiates decorator for labeling function
        """
        if callable(name):
            raise ValueError("Looks like this decorator is missing parentheses!")
        self.name = name
        self.pre = pre

    def __call__(self, f: Callable[..., int]) -> LabelingFunction:
        """Creates and returns a LabelingFunction object for labeling Datapoint

        Args:
            f (Callable[..., int]): core function which labels the input

        Returns:
            LabelingFunction: a callable LabelingFunction object 
        """
        name = self.name or f.__name__
        return LabelingFunction(name=name, f=f, pre=self.pre)
