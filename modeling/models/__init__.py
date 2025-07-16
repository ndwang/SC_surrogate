"""
Model registry and factory for Space Charge Surrogate models.

This module provides a centralized way to register and instantiate different
model architectures. To add a new model, simply import it here and add it
to the MODEL_REGISTRY.
"""

from typing import Dict, Any, Type
import torch.nn as nn
import logging

# Handle both relative and absolute imports for testing
try:
    from .cnn3d import CNN3D
except ImportError:
    # Fallback for direct execution
    from cnn3d import CNN3D

logger = logging.getLogger(__name__)

# Registry mapping model names to their classes
MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {
    'cnn3d': CNN3D,
}


def get_model(name: str, config: Dict[str, Any]) -> nn.Module:
    """
    Get a model instance by name from the registry.
    
    Args:
        name: Name of the model architecture (must be in MODEL_REGISTRY)
        config: Configuration dictionary containing model hyperparameters
        
    Returns:
        Instantiated model
        
    Raises:
        ValueError: If model name is not found in registry
        
    Example:
        >>> config = {'model': {'architecture': 'cnn3d', ...}}
        >>> model = get_model('cnn3d', config)
    """
    name = name.lower().strip()
    
    if name not in MODEL_REGISTRY:
        available_models = list(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown model architecture: '{name}'. "
            f"Available models: {available_models}"
        )
    
    model_class = MODEL_REGISTRY[name]
    
    try:
        model = model_class(config)
        logger.info(f"Successfully created {name} model")
        return model
    except Exception as e:
        logger.error(f"Failed to create {name} model: {e}")
        raise


def register_model(name: str, model_class: Type[nn.Module]) -> None:
    """
    Register a new model class in the registry.
    
    Args:
        name: Name to register the model under
        model_class: Model class that takes config as __init__ parameter
        
    Example:
        >>> register_model('my_new_model', MyNewModelClass)
    """
    name = name.lower().strip()
    
    if name in MODEL_REGISTRY:
        logger.warning(f"Overwriting existing model registration for '{name}'")
    
    MODEL_REGISTRY[name] = model_class
    logger.info(f"Registered model '{name}' -> {model_class.__name__}")


def list_available_models() -> list[str]:
    """
    Get a list of all available model names.
    
    Returns:
        List of registered model names
    """
    return list(MODEL_REGISTRY.keys())


def get_model_info() -> Dict[str, str]:
    """
    Get information about all registered models.
    
    Returns:
        Dictionary mapping model names to their class names
    """
    return {name: cls.__name__ for name, cls in MODEL_REGISTRY.items()}


# Convenience function that automatically gets model name from config
def create_model_from_config(config: Dict[str, Any]) -> nn.Module:
    """
    Create a model using the architecture specified in config.
    
    Args:
        config: Configuration dictionary that must contain model.architecture
        
    Returns:
        Instantiated model
        
    Example:
        >>> config = {'model': {'architecture': 'cnn3d', ...}}
        >>> model = create_model_from_config(config)
    """
    model_config = config.get('model', {})
    architecture = model_config.get('architecture')
    
    if architecture is None:
        raise ValueError(
            "Config must contain 'model.architecture' field. "
            f"Available architectures: {list_available_models()}"
        )
    
    return get_model(architecture, config)


# Export main interface
__all__ = [
    'get_model',
    'register_model', 
    'list_available_models',
    'get_model_info',
    'create_model_from_config',
    'MODEL_REGISTRY'
]


# Example usage and testing
if __name__ == "__main__":
    # Test the model registry
    print(f"Available models: {list_available_models()}")
    print(f"Model info: {get_model_info()}")
    
    # Test model creation
    test_config = {
        'model': {
            'architecture': 'cnn3d',
            'input_channels': 1,
            'output_channels': 3,
            'hidden_channels': [32, 64, 128, 64, 32],
            'kernel_size': 3,
            'padding': 1,
            'activation': 'relu',
            'batch_norm': True,
            'dropout_rate': 0.1,
            'weight_init': 'kaiming_normal'
        }
    }
    
    # Test both ways to create models
    model1 = get_model('cnn3d', test_config)
    model2 = create_model_from_config(test_config)
    
    print(f"Model 1 type: {type(model1).__name__}")
    print(f"Model 2 type: {type(model2).__name__}")
    print("Model registry test completed successfully!") 