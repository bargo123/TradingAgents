class Mt5ProviderError(RuntimeError):
    """Base class for read-only MT5 provider failures."""

class Mt5DependencyError(Mt5ProviderError):
    pass
class Mt5InitializationError(Mt5ProviderError):
    pass
class Mt5NotConnectedError(Mt5ProviderError):
    pass
class Mt5TimeframeError(Mt5ProviderError):
    pass
class Mt5SymbolError(Mt5ProviderError):
    pass
class Mt5SymbolNotFoundError(Mt5SymbolError):
    pass
class Mt5SymbolAmbiguousError(Mt5SymbolError):
    pass
class Mt5AccountError(Mt5ProviderError):
    pass
class Mt5AccountDisconnectedError(Mt5AccountError):
    pass
class Mt5DataError(Mt5ProviderError):
    pass


class Mt5BrokerClockError(Mt5ProviderError):
    """The terminal clock could not be calibrated or is no longer usable."""
