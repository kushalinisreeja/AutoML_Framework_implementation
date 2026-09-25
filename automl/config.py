from dataclasses import dataclass, field
from typing import List, Optional, Union, Dict, Any

@dataclass
class AutoMLConfig:
    """
    Configuration options for the AutoML engine.
    """
    # Task specification: 'auto', 'classification', 'regression'
    task: str = "auto"
    
    # Primary optimization metric. If None, default metric is picked based on task:
    # classification -> 'f1_weighted' (multiclass) or 'roc_auc'/'f1' (binary)
    # regression -> 'r2'
    primary_metric: Optional[str] = None
    
    # Validation strategy
    n_splits: int = 5
    test_size: float = 0.2
    random_state: int = 42
    
    # Optimization budget
    time_budget_secs: Optional[int] = 300  # Total time budget in seconds (None for unlimited)
    n_iter_per_model: int = 10             # Number of hyperparameter trials per candidate model
    
    # Feature engineering & preprocessing
    extract_datetime: bool = True
    impute_strategy: str = "auto"          # 'auto', 'mean', 'median', 'most_frequent'
    scale_numeric: bool = True
    scaling_method: str = "standard"       # 'standard', 'robust', 'minmax'
    encode_categorical: bool = True
    max_one_hot_cardinality: int = 15      # Use one-hot if unique values <= this, else ordinal/target
    feature_selection: bool = True
    max_features: Optional[Union[int, float]] = 0.9  # Select top fraction or number of features
    
    # Models to include or exclude
    include_models: Optional[List[str]] = None
    exclude_models: Optional[List[str]] = None
    
    # Ensembling options
    ensemble: bool = True
    ensemble_type: str = "both"            # 'voting', 'stacking', 'both'
    top_k_models: int = 3                  # Top K models to combine in ensembles
    
    # Verbosity
    verbose: int = 1                       # 0 = silent, 1 = progress summary, 2 = detailed debug
