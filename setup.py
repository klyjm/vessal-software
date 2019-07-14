import sys
from cx_Freeze import setup, Executable

additional_mods = ['numpy.core._methods', 'numpy.lib.format']
setup(name = "test",
      version = "0.1",
      description = "ui",
      options = {'build_exe': {'includes': additional_mods}},
      executables = [Executable("mainui.py")])