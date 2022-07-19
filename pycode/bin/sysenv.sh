#!/bin/bash

#
#                   COPYRIGHT (c) 2020 BY AMELIORAT TECH INC.
#
# This software is furnished under a license and may be used and  copied
# only  in  accordance  with  the  terms  of  such  license and with the
# inclusion of the above copyright notice.  This software or  any  other
# copies  thereof may not be provided or otherwise made available to any
# other person.  No title to and ownership of  the  software  is  hereby
# transferred.
#
# The information in this software is subject to change  without  notice.
#

# set executable path and postgresql client load library
export PATH=/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin:$PATH
# export DYLD_FALLBACK_LIBRARY_PATH=/Library/PostgreSQL/11/lib/:$DYLD_FALLBACK_LIBRARY_PATH

# set application root path
export MODEL_ROOT=/Users/andreasmarx/Projects/DataBauhaus/CIBC/PBB/development/pycode

# set python environment
export PYTHONPATH=$MODEL_ROOT:$PYTHONPATH
export PYEXEC=python3

# set MLflow model tracking URI
export MLFLOW_TRACKING_URI=http://localhost:5000
