# Spatial Reasoning Evaluation Framework

## Overview
A comprehensive framework for evaluating spatial reasoning capabilities in Large Language Models, focusing on 2D/3D navigation and route planning tasks.

## Implementation

```python
import numpy as np
from typing import List, Tuple

class SpatialReasoningEvaluator:
    def __init__(self):
        self.dimensions = {
            '2d': 2,
            '3d': 3
        }
    
    def evaluate_point_plotting(self, 
                              predicted_points: List[Tuple], 
                              ground_truth: List[Tuple]) -> float:
        """
        Evaluates accuracy of spatial point plotting
        
        Args:
            predicted_points: List of predicted coordinate points
            ground_truth: List of actual coordinate points
        
        Returns:
            float: Accuracy score between 0 and 1
        """
        if len(predicted_points) != len(ground_truth):
            return 0.0
        
        total_distance = sum(
            np.linalg.norm(
                np.array(pred) - np.array(true)
            ) 
            for pred, true in zip(predicted_points, ground_truth)
        )
        return 1.0 / (1.0 + total_distance)

    def evaluate_route_planning(self,
                              predicted_path: List[Tuple],
                              optimal_path: List[Tuple],
                              dimension: str = '2d') -> dict:
        """
        Evaluates route planning efficiency
        
        Args:
            predicted_path: List of coordinates in predicted route
            optimal_path: List of coordinates in optimal route
            dimension: '2d' or '3d'
            
        Returns:
            dict: Evaluation metrics including path length and efficiency
        """
        dim = self.dimensions[dimension]
        
        pred_length = self._calculate_path_length(predicted_path, dim)
        opt_length = self._calculate_path_length(optimal_path, dim)
        
        return {
            'path_efficiency': opt_length / pred_length if pred_length > 0 else 0,
            'predicted_length': pred_length,
            'optimal_length': opt_length
        }
    
    def _calculate_path_length(self, path: List[Tuple], dim: int) -> float:
        """Calculate total path length"""
        return sum(
            np.linalg.norm(
                np.array(path[i][:dim]) - np.array(path[i-1][:dim])
            ) 
            for i in range(1, len(path))
        )

# Example Usage
evaluator = SpatialReasoningEvaluator()

# Test 2D point plotting
pred_points = [(1, 1), (2, 3), (4, 2)]
true_points = [(1, 2), (2, 3), (4, 1)]
accuracy = evaluator.evaluate_point_plotting(pred_points, true_points)

# Test 2D route planning
pred_path = [(0, 0), (1, 1), (2, 2), (3, 2)]
opt_path = [(0, 0), (2, 1), (3, 2)]
results = evaluator.evaluate_route_planning(pred_path, opt_path, '2d')
