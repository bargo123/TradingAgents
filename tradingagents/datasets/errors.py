class DatasetError(Exception):
    """Base class for bounded, local dataset-factory failures."""


class DatasetConfigError(DatasetError, ValueError):
    pass


class SourceIntegrityError(DatasetError):
    pass


class SourceReadError(DatasetError):
    pass


class DatasetSchemaError(DatasetError, ValueError):
    pass


class DatasetOutputError(DatasetError):
    pass
