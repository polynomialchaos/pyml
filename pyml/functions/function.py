class Function:
    """Function (base) class."""

    def function(self, *args):
        """Function callable."""
        raise NotImplementedError

    def function_derive(self, *args):
        """Function derivative callable."""
        raise NotImplementedError
