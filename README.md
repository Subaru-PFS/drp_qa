drp_qa
======

To actually run the gen2 executable, add an entry `command_name_config` in the PfsMapper yaml file should be added.


File description below:

   -`SConstruct`
     Configuring scons (a replacement for make)

   - `mypy.ini`
     mypy (type checker) configuration

   - `pyproject.toml`
     black (shaping tool) settings

   - `setup.cfg`
     Configuring flake8 (Coding Convention Checker)

   - `bin.src/`
     gen2 executables
     There are no meaningful files in this directory to read.

     Copied to bin/ by running `scons`. For platforms that require shebang changes, shebang changes are made and copied to bin/

   - `pipelines/`
     gen3 pipeline definition files run like:
     `pipetask run -p '$PFS_QA_DIR/pipelines/qa.yaml'`

   - `python/`
     Script entities

   - `ups/`
     Information files for the setup command (EUPS)
