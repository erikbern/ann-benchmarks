import os


def get_bool_env_var(env_var_name: str, default_value: bool) -> bool:
    """
    Interpret the given environment variable's value as a boolean flag. If it
    is not specified or empty, return the given default value.
    """
    str_value = os.getenv(env_var_name)
    if str_value is None:
        return default_value
    str_value = str_value.strip().lower()
    if len(str_value) == 0:
        return default_value
    return str_value in ['y', 'yes', '1', 'true', 't', 'on']
