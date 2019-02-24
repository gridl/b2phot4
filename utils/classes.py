import os
from configparser import ConfigParser


class Configuration(ConfigParser):
    """Configuration class to read experiment and distribute parameters"""
    def __init__(self, configuration_file):
        """
        Parameters
        ----------
        configuration_file : str
            Path to the config.ini file for the experiment to run.
        """
        ConfigParser.__init__(self)
        self.file = configuration_file
        self.read(configuration_file)
        self.check_configuration_file()

    def check_configuration_file(self):
        for section in ['model', 'loss', 'optimizer', 'experiment']:
            if not self.has_section(section):
                raise Exception("Configuration file does not have the required section: {}".format(section))

    def get_path(self):
        """ Get path of the configuration file"""
        return os.path.dirname(self.file)

    def get_section(self, section):
        """Get all parameters from a section as a dict

        Parameters
        ----------
        section : str
            Section of the configuration file.

        Returns
        -------
        section : dict
            Dictionary containing keys and values of section.
        """
        if not self.has_section(section):
            raise KeyError("The section name {} does not exist in the configuration file".format(section))
        return dict(self.items(section))
